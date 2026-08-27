from flask import Flask, request, jsonify, g
import threading
import time
import importlib
from typing import Dict, List, Tuple, Optional, Any, Type
from tracerigor.env import REGISTERED_ENV
from tracerigor.env.base.base_service import BaseService
from tracerigor.env.base.base_service_config import BaseServiceConfig
import hydra
from omegaconf import DictConfig
from tracerigor.server.llm_as_judge import wandb_run_context

# NEW imports
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

class BatchEnvServer:
    """
    A unified server for handling batch environment operations through HTTP requests.
    Uses environment services to handle operations and properly handle serialization.
    Exposes only the standard BaseService interface.
    """

    def __init__(self, config):
        """
        Initialize the BatchEnvServer.
        """
        self.host = config.server.host
        self.port = config.server.port
        self.debug = config.server.debug
        self.config = config
        self.wandb_context = None

        # Dictionary to store services by environment type
        self.services = {}
        # Dictionary to track which service manages which environment ID
        self.env_to_service = {}

        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger("BatchEnvServer")

        # ThreadPool executor for running service calls with timeout
        # Configurable via hydra config: server.executor_workers
        executor_workers = getattr(self.config.server, "executor_workers", 8)
        self.executor = ThreadPoolExecutor(max_workers=executor_workers)
        # Server-side timeout for service calls (seconds). Default < client's 300s to fail fast server-side.
        self.request_timeout = getattr(self.config.server, "request_timeout", 240)

        # Create Flask app
        self.app = Flask(__name__)
        self._setup_routes()

        # Server state
        self.is_running = False
        self.server_thread = None

    # --- helper to run service calls with timeout ---
    def _run_with_timeout(self, fn, *args, timeout: Optional[float] = None, **kwargs):
        """
        Submit fn(*args, **kwargs) to executor and wait up to timeout seconds.
        On timeout, cancel the future and raise FutureTimeoutError.
        """
        fut = self.executor.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=(timeout if timeout is not None else self.request_timeout))
        except FutureTimeoutError:
            # best-effort cancel
            try:
                fut.cancel()
            except Exception:
                pass
            raise
        except Exception:
            # re-raise so caller can handle logging/translation to HTTP error
            raise

    def _setup_routes(self):
        """Set up HTTP routes for the Flask app"""
        # request id + logging
        @self.app.before_request
        def _before_request():
            g.request_start_time = time.time()
            g.request_id = uuid.uuid4().hex
            # remote_addr may be None in some infra; guard it
            remote = request.remote_addr or "local"
            self.logger.info(f"[{g.request_id}] -> {request.method} {request.path} from {remote}")

        @self.app.after_request
        def _after_request(response):
            elapsed = time.time() - g.get("request_start_time", time.time())
            response.headers['X-Request-ID'] = g.get("request_id", "")
            self.logger.info(f"[{g.request_id}] <- {request.method} {request.path} status={response.status_code} elapsed={elapsed:.3f}s")
            return response

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "ok",
                "message": "Environment server is running",
                "registered_envs": list(REGISTERED_ENV.keys()),
                "active_services": list(self.services.keys()),
                "active_environments": len(self.env_to_service)
            }), 200

        @self.app.route('/environments', methods=['POST'])
        def create_environments_batch():
            """Create environments endpoint - implements BaseService interface"""
            data = request.json
            if not data or 'ids2configs' not in data:
                return jsonify({"error": "Missing required parameter: ids2configs"}), 400

            ids2configs = data['ids2configs']
            # run create in executor with timeout
            try:
                self._run_with_timeout(self._create_environments_batch, ids2configs)
            except FutureTimeoutError:
                self.logger.exception("create_environments_batch timed out")
                return jsonify({"error": "create_environments_batch_timeout"}), 504
            except Exception:
                self.logger.exception("create_environments_batch failed")
                return jsonify({"error": "create_environments_batch_error"}), 500

            return jsonify({"success": True}), 200

        @self.app.route('/batch/reset', methods=['POST'])
        def reset_batch():
            """Reset multiple environments endpoint"""
            data = request.json
            if not data or 'ids2seeds' not in data:
                return jsonify({"error": "Missing required parameter: ids2seeds"}), 400

            ids2seeds = data['ids2seeds']
            try:
                results = self._run_with_timeout(self._reset_batch, ids2seeds)
            except FutureTimeoutError:
                self.logger.exception("reset_batch timed out")
                return jsonify({"error": "reset_batch_timeout"}), 504
            except Exception:
                self.logger.exception("reset_batch failed")
                return jsonify({"error": "reset_batch_error"}), 500

            return jsonify({"results": results}), 200

        @self.app.route('/batch/step', methods=['POST'])
        def step_batch():
            """Step multiple environments endpoint"""
            data = request.json
            if not data or 'ids2actions' not in data:
                return jsonify({"error": "Missing required parameter: ids2actions"}), 400

            ids2actions = data['ids2actions']
            try:
                results = self._run_with_timeout(self._step_batch, ids2actions)
            except FutureTimeoutError:
                self.logger.exception("step_batch timed out")
                # produce a structured error per env id so caller can handle it gracefully
                # mark all as timed out
                error_results = {eid: (None, 0.0, True, {"error": "step_timeout"}) for eid in ids2actions.keys()}
                return jsonify({"results": error_results}), 504
            except Exception:
                self.logger.exception("step_batch failed")
                error_results = {eid: (None, 0.0, True, {"error": "step_error"}) for eid in ids2actions.keys()}
                return jsonify({"results": error_results}), 500

            return jsonify({"results": results}), 200

        @self.app.route('/batch/reward', methods=['POST'])
        def compute_reward_batch():
            """Compute reward for multiple environments endpoint"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400

            env_ids = data['env_ids']
            try:
                rewards = self._run_with_timeout(self._compute_reward_batch, env_ids)
            except FutureTimeoutError:
                self.logger.exception("compute_reward_batch timed out")
                # return partial with error tags
                error_rewards = {eid: None for eid in env_ids}
                return jsonify({"rewards": error_rewards, "error": "compute_reward_batch_timeout"}), 504
            except Exception:
                self.logger.exception("compute_reward_batch failed")
                return jsonify({"error": "compute_reward_batch_error"}), 500

            return jsonify({"rewards": rewards}), 200

        @self.app.route('/batch/system_prompt', methods=['POST'])
        def get_system_prompts_batch():
            """Get system prompts for multiple environments endpoint"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400

            env_ids = data['env_ids']
            try:
                prompts = self._run_with_timeout(self._get_system_prompts_batch, env_ids)
            except FutureTimeoutError:
                self.logger.exception("get_system_prompts_batch timed out")
                error_prompts = {eid: "" for eid in env_ids}
                return jsonify({"system_prompts": error_prompts, "error": "get_system_prompts_batch_timeout"}), 504
            except Exception:
                self.logger.exception("get_system_prompts_batch failed")
                return jsonify({"error": "get_system_prompts_batch_error"}), 500

            return jsonify({"system_prompts": prompts}), 200

        @self.app.route('/batch/close', methods=['POST'])
        def close_batch():
            """Close multiple environments endpoint"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400

            env_ids = data['env_ids']
            try:
                self._run_with_timeout(self._close_batch, env_ids)
            except FutureTimeoutError:
                self.logger.exception("close_batch timed out")
                return jsonify({"error": "close_batch_timeout"}), 504
            except Exception:
                self.logger.exception("close_batch failed")
                return jsonify({"error": "close_batch_error"}), 500

            return jsonify({"status": "success"}), 200

        # Single-env endpoints reuse the batch implementations and thus benefit from the timeout wrapper above
        @self.app.route('/reset/<env_id>', methods=['POST'])
        def reset_environment(env_id):
            data = request.json or {}
            seed = data.get('seed')
            results = self._reset_batch({env_id: seed})
            if env_id not in results:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
            obs, info = results[env_id]
            return jsonify({"observation": obs, "info": info}), 200

        @self.app.route('/step/<env_id>', methods=['POST'])
        def step_environment(env_id):
            data = request.json
            if not data or 'action' not in data:
                return jsonify({"error": "Missing required parameter: action"}), 400

            action = data['action']
            results = self._step_batch({env_id: action})
            if env_id not in results:
                return jsonify({"error": f"Environment {env_id} not found"}), 404

            obs, reward, done, info = results[env_id]
            return jsonify({
                "observation": obs,
                "reward": reward,
                "done": done,
                "info": info
            }), 200

        @self.app.route('/reward/<env_id>', methods=['GET'])
        def compute_reward(env_id):
            rewards = self._compute_reward_batch([env_id])
            if env_id not in rewards:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
            return jsonify({"reward": rewards[env_id]}), 200

        @self.app.route('/system_prompt/<env_id>', methods=['GET'])
        def get_system_prompt(env_id):
            prompts = self._get_system_prompts_batch([env_id])
            if env_id not in prompts:
                return jsonify({"error": f"Environment {env_id} not found"}), 404
            return jsonify({"system_prompt": prompts[env_id]}), 200

        @self.app.route('/close/<env_id>', methods=['DELETE'])
        def close_environment(env_id):
            self._close_batch([env_id])
            return jsonify({"status": "success"}), 200

    # --- rest of your original helper methods, mostly unchanged, but we keep the pattern of using services ---
    def _get_service_for_env_name(self, env_name: str) -> BaseService:
        if env_name not in self.services:
            if env_name not in REGISTERED_ENV:
                raise ValueError(f"Unknown environment type: {env_name}")
            if "service_cls" not in REGISTERED_ENV[env_name]:
                raise ValueError(f"No service class registered for environment type: {env_name}")
            service_class = REGISTERED_ENV[env_name]["service_cls"]
            service_config = REGISTERED_ENV[env_name].get("service_config_cls", BaseServiceConfig)(**self.config.get(env_name, {}))
            self.services[env_name] = service_class(service_config)
        return self.services[env_name]

    def _get_service_for_env(self, env_id: str) -> Tuple[BaseService, str]:
        if env_id not in self.env_to_service:
            raise ValueError(f"Environment {env_id} not found")
        env_name = self.env_to_service[env_id]
        service = self.services[env_name]
        return service, env_name

    def _create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        for env_id, config in ids2configs.items():
            env_name = config.get("env_name")
            if not env_name:
                raise ValueError(f"Config for environment {env_id} is missing 'env_name'")
            if env_name not in self.services:
                self.services[env_name] = self._get_service_for_env_name(env_name)
            self.env_to_service[env_id] = env_name

        service_to_configs = {}
        for env_id, config in ids2configs.items():
            env_name = self.env_to_service[env_id]
            service_to_configs.setdefault(env_name, {})[env_id] = config

        for env_name, configs in service_to_configs.items():
            service = self.services[env_name]
            # call without extra wrapping here, outer call to _run_with_timeout covers timeout
            service.create_environments_batch(configs)


    def _reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        service_groups = {}
        for env_id, seed in ids2seeds.items():
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, {})
            service_groups[env_name][1][env_id] = seed

        results = {}
        for env_name, (service, group_ids2seeds) in service_groups.items():
            try:
                service_results = service.reset_batch(group_ids2seeds)
                results.update(service_results)
            except Exception as e:
                self.logger.exception("reset_batch failed for service %s", env_name)
                for eid in group_ids2seeds.keys():
                    results[eid] = (None, {"error": "reset_failed", "msg": str(e)})
        return results

    def _step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        service_groups = {}
        for env_id, action in ids2actions.items():
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, {})
            service_groups[env_name][1][env_id] = action

        results = {}
        for env_name, (service, group_ids2actions) in service_groups.items():
            try:
                # run service.step_batch within executor with timeout
                service_results = self._run_with_timeout(service.step_batch, group_ids2actions)
                results.update(service_results)
            except FutureTimeoutError:
                self.logger.exception("step_batch timed out for service %s", env_name)
                # indicate timeout per env
                for eid in group_ids2actions.keys():
                    results[eid] = (None, 0.0, True, {"error": "step_timeout"})
            except Exception as e:
                self.logger.exception("step_batch failed for service %s", env_name)
                for eid in group_ids2actions.keys():
                    results[eid] = (None, 0.0, True, {"error": "step_error", "msg": str(e)})

        return results

    def _compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        service_groups = {}
        for env_id in env_ids:
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, [])
            service_groups[env_name][1].append(env_id)

        results = {}
        for env_name, (service, group_env_ids) in service_groups.items():
            try:
                service_results = service.compute_reward_batch(group_env_ids)
                results.update(service_results)
            except Exception as e:
                self.logger.exception("compute_reward_batch failed for service %s", env_name)
                for eid in group_env_ids:
                    results[eid] = None
        return results

    def _get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        service_groups = {}
        for env_id in env_ids:
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, [])
            service_groups[env_name][1].append(env_id)

        results = {}
        for env_name, (service, group_env_ids) in service_groups.items():
            try:
                service_results = service.get_system_prompts_batch(group_env_ids)
                results.update(service_results)
            except Exception as e:
                self.logger.exception("get_system_prompts_batch failed for service %s", env_name)
                for eid in group_env_ids:
                    results[eid] = ""
        return results

    def _close_batch(self, env_ids: List[str]) -> None:
        service_groups = {}
        for env_id in env_ids:
            service, env_name = self._get_service_for_env(env_id)
            if env_name not in service_groups:
                service_groups[env_name] = (service, [])
            service_groups[env_name][1].append(env_id)
            # Remove from tracking
            try:
                del self.env_to_service[env_id]
            except KeyError:
                pass

        for env_name, (service, group_env_ids) in service_groups.items():
            try:
                service.close_batch(group_env_ids)
            except Exception as e:
                self.logger.exception("close_batch failed for service %s", env_name)

    def _generate_env_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def start(self, background: bool = True) -> None:
        if self.is_running:
            print("Server is already running")
            return

        if self.config.get("use_state_reward", False):
            self.wandb_context = wandb_run_context()
            self.wandb_context.__enter__()
            print("Initialized wandb for LLM Judge")

        if background:
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True

            # Wait for server to start
            max_retries = 5
            retry_delay = 0.5
            import requests
            for _ in range(max_retries):
                time.sleep(retry_delay)
                try:
                    response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                    if response.status_code == 200:
                        print(f"Server started on http://{self.host}:{self.port}")
                        break
                except Exception:
                    pass
            else:
                print("Server may not have started properly")
        else:
            self.is_running = True
            self._run_server()

    def _run_server(self) -> None:
        """Run the Flask server"""
        # run Flask in threaded mode to allow concurrent connections (lightweight)
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False, threaded=True)

    def stop(self) -> None:
        """Stop the server and clean up resources"""
        if not self.is_running:
            return

        # Close all environments
        env_ids = list(self.env_to_service.keys())
        self._close_batch(env_ids)

        # Shut down the Flask server
        self.is_running = False
        if self.server_thread and self.server_thread.is_alive():
            import requests
            try:
                requests.post(f"http://{self.host}:{self.port}/shutdown")
            except Exception:
                pass

        if self.wandb_context:
            self.wandb_context.__exit__(None, None, None)
            self.wandb_context = None
            print("Closed wandb for LLM Judge")

        print("Server stopped")


@hydra.main(version_base=None, config_path="config", config_name="server")
def main(cfg: DictConfig):
    """
    Main function to start the batch environment server.
    Uses Hydra for configuration management.

    Args:
        cfg: Configuration object from Hydra
    """
    # Create and start server with configuration
    print(cfg)
    server = BatchEnvServer(cfg)
    print(f"Starting Batch Environment Server on http://{cfg.server.host}:{cfg.server.port}")
    server.start(background=False)


if __name__ == "__main__":
    main()
