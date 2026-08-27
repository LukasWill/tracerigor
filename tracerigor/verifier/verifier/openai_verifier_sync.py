# verifier/openai_verifier_sync.py
import asyncio
from typing import Any, Dict, List
from verifier.openai_verifier import run_openai_verifier  # adjust import to your path

def run_openai_verifier_sync(
    input_data: List[Dict[str, Any]],
    rubric: str,
    model_name: str,
    model_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Safe sync entrypoint for Flask / regular Python code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # unlikely in Flask, but keep it robust
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(asyncio.run, run_openai_verifier(input_data, rubric, model_name, model_params))
            return fut.result()
    else:
        return asyncio.run(run_openai_verifier(input_data, rubric, model_name, model_params))