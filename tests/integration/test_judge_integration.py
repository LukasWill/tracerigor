#!/usr/bin/env python3
"""
Integration tests for the tracerigor/judge/ package.

Tests cover:
  1. Schema (TurnJudgePacket, JudgeResponse, RubricResult, JSON schemas)
  2. Config (JudgeConfig defaults, from_dict, rubric naming)
  3. Heuristics (format, invalid action, repetition, empty trace)
  4. Reward (compute_process_reward, rubric_scores_dict)
  5. Prompt/template (env template registration, message building)
  6. Env templates (ground truth builders, template content)
  7. Packet builder (build_turn_packet, tag extraction)
  8. Router parsing (_parse_llm_response, _fallback_parse, via public methods)
"""
import json
import sys
import traceback
from dataclasses import asdict

# ---- Test utilities ----
_results = {"pass": 0, "fail": 0}


def _run(name, fn):
    try:
        fn()
        _results["pass"] += 1
        print(f"  PASS  {name}")
    except Exception as e:
        _results["fail"] += 1
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ========================================================================
# 1. Schema tests
# ========================================================================
def test_turn_judge_packet_defaults():
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(env_id="env_0", episode_step=1, task_name="sokoban")
    assert p.format_correct is True
    assert p.action_feedback == ""
    assert p.ground_truth_state == {}
    assert p.task_description == ""
    assert p.valid_actions == ""
    assert p.post_action_observation == ""
    assert p.agent_modality == "text"


def test_rubric_result_defaults():
    from tracerigor.judge.schema import RubricResult
    r = RubricResult()
    assert r.label == "uncertain"
    assert r.score == 0.5
    assert r.confidence == 0.5
    assert r.evidence == []


def test_judge_response_defaults():
    from tracerigor.judge.schema import JudgeResponse
    r = JudgeResponse()
    assert r.short_circuited is False
    assert r.rubrics == {}
    assert r.process_reward == 0.0


def test_json_schema_keys():
    from tracerigor.judge.schema import JUDGE_OUTPUT_JSON_SCHEMA
    props = JUDGE_OUTPUT_JSON_SCHEMA["properties"]
    assert "observation_grounding" in props
    assert "action_coherence" in props
    assert "temporal_consistency" in props
    # Must NOT have old "grounding" key
    assert "grounding" not in props


# ========================================================================
# 2. Config tests
# ========================================================================
def test_config_defaults_rubric_names():
    from tracerigor.judge.config import JudgeConfig
    cfg = JudgeConfig()
    assert "observation_grounding" in cfg.rubrics
    assert "action_coherence" in cfg.rubrics
    assert "temporal_consistency" in cfg.rubrics
    assert "grounding" not in cfg.rubrics


def test_config_reward_weights():
    from tracerigor.judge.config import RewardConfig
    r = RewardConfig()
    assert "observation_grounding" in r.rubric_weights
    assert "grounding" not in r.rubric_weights
    assert abs(sum(r.rubric_weights.values()) - 1.0) < 1e-6


def test_config_from_dict():
    from tracerigor.judge.config import JudgeConfig
    cfg = JudgeConfig.from_dict({
        "enabled": False,
        "rubrics": ["observation_grounding", "action_coherence"],
        "provider": {"model": "test-model", "temperature": 0.5},
        "heuristics": {"min_trace_tokens": 16},
    })
    assert cfg.enabled is False
    assert cfg.rubrics == ["observation_grounding", "action_coherence"]
    assert cfg.provider.model == "test-model"
    assert cfg.provider.temperature == 0.5
    assert cfg.heuristics.min_trace_tokens == 16


# ========================================================================
# 3. Heuristics tests
# ========================================================================
def test_heuristic_format_violation():
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import HeuristicConfig
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(
        env_id="e0", episode_step=1, task_name="sokoban",
        format_correct=False,
    )
    result = run_heuristics(p, HeuristicConfig())
    assert result["should_short_circuit"] is True
    assert "format" in result["reason"].lower()


def test_heuristic_no_action_parsed():
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import HeuristicConfig
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(
        env_id="e0", episode_step=1, task_name="sokoban",
        format_correct=True,
        chosen_action="",
        action_tokens="",
    )
    result = run_heuristics(p, HeuristicConfig())
    assert result["should_short_circuit"] is True
    assert "no action" in result["reason"].lower()


def test_heuristic_invalid_action_feedback():
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import HeuristicConfig
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(
        env_id="e0", episode_step=1, task_name="sciworld",
        chosen_action="dance around",
        action_tokens="<action>dance around</action>",
        action_feedback="No known action matches that input.",
    )
    result = run_heuristics(p, HeuristicConfig())
    assert result["should_short_circuit"] is True
    assert "rejected" in result["reason"].lower()


def test_heuristic_invalid_action_admissible():
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import HeuristicConfig
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(
        env_id="e0", episode_step=1, task_name="sokoban",
        chosen_action="Teleport",
        action_tokens="<action>Teleport</action>",
        admissible_actions=["Up", "Down", "Left", "Right"],
    )
    result = run_heuristics(p, HeuristicConfig())
    assert result["should_short_circuit"] is True
    assert "not in valid" in result["reason"].lower()


def test_heuristic_valid_action_passes():
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import HeuristicConfig
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(
        env_id="e0", episode_step=2, task_name="sokoban",
        chosen_action="Up",
        action_tokens="<action>Up</action>",
        admissible_actions=["Up", "Down", "Left", "Right"],
        reasoning_tokens="<think>I need to push the box up.</think>",
        raw_trace="<think>I need to push the box up.</think><action>Up</action>",
        history=[{"obs": "grid state", "action_tokens": "<action>Down</action>"}],
    )
    result = run_heuristics(p, HeuristicConfig())
    assert result["should_short_circuit"] is False


def test_heuristic_repetition_detection():
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import HeuristicConfig
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(
        env_id="e0", episode_step=3, task_name="sokoban",
        chosen_action="Up",
        action_tokens="<action>Up</action>",
        reasoning_tokens="<think>Go up again.</think>",
        admissible_actions=["Up", "Down", "Left", "Right"],
        current_observation_text="Player at (3,2), box at (3,1)",
        post_action_observation="Player at (3,2), box at (3,1)",  # unchanged
        history=[{
            "observation_text": "Player at (3,2), box at (3,1)",
            "action_tokens": "<action>Up</action>",
        }],
    )
    result = run_heuristics(p, HeuristicConfig())
    # Repetition is informational, not critical — should NOT short-circuit
    assert result["should_short_circuit"] is False
    assert result["checks"]["repetition"]["fired"] is True


def test_heuristic_empty_trace():
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import HeuristicConfig
    from tracerigor.judge.schema import TurnJudgePacket
    p = TurnJudgePacket(
        env_id="e0", episode_step=1, task_name="sokoban",
        chosen_action="Up",
        action_tokens="<action>Up</action>",
        reasoning_tokens="<think>ok</think>",  # very short
        admissible_actions=["Up", "Down", "Left", "Right"],
    )
    cfg = HeuristicConfig(min_trace_tokens=50)
    result = run_heuristics(p, cfg)
    assert result["checks"]["empty_trace"]["fired"] is True


# ========================================================================
# 4. Reward tests
# ========================================================================
def test_reward_all_pass():
    from tracerigor.judge.reward import compute_process_reward, rubric_scores_dict
    from tracerigor.judge.config import RewardConfig
    from tracerigor.judge.schema import JudgeResponse, RubricResult
    resp = JudgeResponse(rubrics={
        "observation_grounding": RubricResult(score=1.0, label="pass", confidence=0.9),
        "action_coherence": RubricResult(score=1.0, label="pass", confidence=0.9),
        "temporal_consistency": RubricResult(score=1.0, label="pass", confidence=0.9),
    })
    r = compute_process_reward(resp, RewardConfig())
    # r_raw = 1.0, mapped = 2*1 - 1 = 1.0, clamped to 0.25
    assert r == 0.25, f"Expected 0.25, got {r}"


def test_reward_all_fail():
    from tracerigor.judge.reward import compute_process_reward
    from tracerigor.judge.config import RewardConfig
    from tracerigor.judge.schema import JudgeResponse, RubricResult
    resp = JudgeResponse(rubrics={
        "observation_grounding": RubricResult(score=0.0, label="fail", confidence=0.9),
        "action_coherence": RubricResult(score=0.0, label="fail", confidence=0.9),
        "temporal_consistency": RubricResult(score=0.0, label="fail", confidence=0.9),
    })
    r = compute_process_reward(resp, RewardConfig())
    # r_raw = 0.0, mapped = -1.0, clamped to -0.25
    assert r == -0.25, f"Expected -0.25, got {r}"


def test_reward_short_circuit():
    from tracerigor.judge.reward import compute_process_reward
    from tracerigor.judge.config import RewardConfig
    from tracerigor.judge.schema import JudgeResponse
    resp = JudgeResponse(short_circuited=True)
    r = compute_process_reward(resp, RewardConfig())
    assert r < 0, f"Short-circuit reward should be negative, got {r}"


def test_rubric_scores_dict_keys():
    from tracerigor.judge.reward import rubric_scores_dict
    from tracerigor.judge.schema import JudgeResponse, RubricResult
    resp = JudgeResponse(
        overall_confidence=0.85,
        process_reward=0.1,
        rubrics={
            "observation_grounding": RubricResult(score=1.0, label="pass", confidence=0.9),
            "action_coherence": RubricResult(score=0.0, label="fail", confidence=0.8),
        },
    )
    d = rubric_scores_dict(resp)
    assert "judge_observation_grounding_score" in d
    assert "judge_action_coherence_score" in d
    assert d["judge_observation_grounding_score"] == 1.0
    assert d["judge_action_coherence_score"] == 0.0
    assert d["judge_overall_confidence"] == 0.85


# ========================================================================
# 5. Prompt / template tests
# ========================================================================
def test_env_template_registration():
    from tracerigor.judge.prompt import available_templates, _TEMPLATE_REGISTRY
    from tracerigor.judge.env_templates import register_all_templates
    # Clear and re-register
    _TEMPLATE_REGISTRY.clear()
    register_all_templates()
    templates = available_templates()
    assert "sciworld" in templates
    assert "sokoban" in templates
    assert "universal" in templates["sciworld"]
    assert "universal" in templates["sokoban"]


def test_build_messages_sokoban():
    from tracerigor.judge.prompt import build_messages, _TEMPLATE_REGISTRY
    from tracerigor.judge.env_templates import register_all_templates
    from tracerigor.judge.schema import TurnJudgePacket
    _TEMPLATE_REGISTRY.clear()
    register_all_templates()

    p = TurnJudgePacket(
        env_id="e0", episode_step=2, task_name="sokoban",
        current_observation_text="##\n#P_#\n#_X#\n#_O#\n##",
        reasoning_tokens="<think>Push box right</think>",
        action_tokens="<action>Right</action>",
        chosen_action="Right",
        admissible_actions=["Up", "Down", "Left", "Right"],
        ground_truth_state={"state_sentences": ["Player at row 1 col 1", "Box at row 2 col 1"]},
    )
    msgs = build_messages(p, rubric="universal")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # System prompt should contain Sokoban-specific content
    assert "Sokoban" in msgs[0]["content"]
    assert "observation_grounding" in msgs[0]["content"].lower() or "Observation Grounding" in msgs[0]["content"]
    # User prompt should contain the observation and action
    assert "Push box right" in msgs[1]["content"]
    assert "Right" in msgs[1]["content"]
    # Ground truth should be included
    assert "Player at row 1 col 1" in msgs[1]["content"]


def test_build_messages_sciworld():
    from tracerigor.judge.prompt import build_messages, _TEMPLATE_REGISTRY
    from tracerigor.judge.env_templates import register_all_templates
    from tracerigor.judge.schema import TurnJudgePacket
    _TEMPLATE_REGISTRY.clear()
    register_all_templates()

    p = TurnJudgePacket(
        env_id="e1", episode_step=5, task_name="sciworld",
        current_observation_text="This room is called the kitchen. In it, you see: a table, a stove.",
        reasoning_tokens="<reflection>I am in the kitchen. I need to find the thermometer.</reflection>",
        action_tokens="<action>look around</action>",
        chosen_action="look around",
        task_description="Heat water to 100 degrees",
        ground_truth_state={"ground_truth_location": "kitchen", "ground_truth_inventory": ["thermometer"]},
    )
    msgs = build_messages(p, rubric="universal")
    assert len(msgs) == 2
    assert "SciWorld" in msgs[0]["content"]
    # Ground truth section should mention kitchen
    assert "kitchen" in msgs[1]["content"]
    assert "thermometer" in msgs[1]["content"]
    # Task description should appear in user prompt
    assert "Heat water" in msgs[1]["content"]


def test_build_messages_default_fallback():
    from tracerigor.judge.prompt import build_messages, _TEMPLATE_REGISTRY
    from tracerigor.judge.schema import TurnJudgePacket
    _TEMPLATE_REGISTRY.clear()  # No templates registered

    p = TurnJudgePacket(
        env_id="e2", episode_step=1, task_name="unknown_env",
        current_observation_text="some obs",
        reasoning_tokens="some thinking",
        action_tokens="some action",
        chosen_action="do X",
    )
    msgs = build_messages(p, rubric="universal")
    assert len(msgs) == 2
    # Falls back to default prompts
    assert "trace verifier" in msgs[0]["content"].lower()


# ========================================================================
# 6. Env templates (ground truth builders)
# ========================================================================
def test_sciworld_ground_truth_builder():
    from tracerigor.judge.env_templates import build_sciworld_ground_truth
    gt = {"ground_truth_location": "kitchen", "ground_truth_inventory": ["flask", "thermometer"]}
    section = build_sciworld_ground_truth(gt)
    assert "kitchen" in section
    assert "flask" in section
    assert "thermometer" in section


def test_sokoban_ground_truth_builder():
    from tracerigor.judge.env_templates import build_sokoban_ground_truth
    gt = {"state_sentences": ["Player at row 1 col 2", "Box A at row 3 col 4"]}
    section = build_sokoban_ground_truth(gt)
    assert "Player at row 1 col 2" in section
    assert "Box A at row 3 col 4" in section


def test_ground_truth_dispatch():
    from tracerigor.judge.env_templates import build_ground_truth_section
    # Sokoban
    s = build_ground_truth_section("sokoban", {"state_sentences": ["Player at (1,1)"]})
    assert "Player at (1,1)" in s
    # SciWorld
    s = build_ground_truth_section("sciworld", {"ground_truth_location": "lab"})
    assert "lab" in s
    # Unknown env → empty
    s = build_ground_truth_section("frozenlake", {"some": "data"})
    assert s == ""
    # Empty state → empty
    s = build_ground_truth_section("sokoban", {})
    assert s == ""


# ========================================================================
# 7. Packet builder tests
# ========================================================================
def test_build_turn_packet_sokoban():
    from tracerigor.judge.packet_builder import build_turn_packet
    p = build_turn_packet(
        env_id="env_0",
        task_name="sokoban",
        episode_step=3,
        obs_text="##\n#PX#\n##",
        obs_images=[],
        history=[],
        raw_trace="<think>The box is to my right. I'll push it right.</think><action>Right</action>",
        info={
            "action_content": "Right",
            "format_correct": True,
            "action_feedback": "",
            "ground_truth_state": {"state_sentences": ["Player at row 1 col 1"]},
            "available_actions": ["Up", "Down", "Left", "Right"],
        },
        admissible_actions=["Up", "Down", "Left", "Right"],
    )
    assert p.task_name == "sokoban"
    assert p.episode_step == 3
    assert p.chosen_action == "Right"
    assert "<think>" in p.reasoning_tokens
    assert "<action>" in p.action_tokens
    assert p.format_correct is True
    assert p.ground_truth_state["state_sentences"] == ["Player at row 1 col 1"]
    assert p.agent_modality == "text"


def test_build_turn_packet_sciworld():
    from tracerigor.judge.packet_builder import build_turn_packet
    p = build_turn_packet(
        env_id="env_1",
        task_name="sciworld",
        episode_step=5,
        obs_text="This room is called the kitchen.",
        obs_images=[],
        history=[],
        raw_trace="<reflection>I am in the kitchen.</reflection><action>look around</action>",
        info={
            "action_content": "look around",
            "format_correct": True,
            "task_description": "Heat water",
            "valid_actions": "look around, pick up flask, go to lab",
        },
    )
    assert p.task_name == "sciworld"
    assert "<reflection>" in p.reasoning_tokens
    assert "<action>" in p.action_tokens
    assert p.task_description == "Heat water"
    assert p.valid_actions == "look around, pick up flask, go to lab"


def test_build_turn_packet_format_error():
    from tracerigor.judge.packet_builder import build_turn_packet
    p = build_turn_packet(
        env_id="env_2",
        task_name="sokoban",
        episode_step=1,
        obs_text="grid",
        obs_images=[],
        history=[],
        raw_trace="I don't know what to do",  # no tags
        info={"format_correct": False, "action_feedback": ""},
    )
    assert p.format_correct is False
    assert p.reasoning_tokens == ""
    assert p.action_tokens == ""


# ========================================================================
# 8. Router parsing tests (testing parse logic via internal helpers)
# ========================================================================
def _make_router_no_client():
    """Create a JudgeRouter without requiring openai/httpx by mocking the client."""
    from tracerigor.judge.config import JudgeConfig
    from tracerigor.judge.router import JudgeRouter
    cfg = JudgeConfig()
    # Bypass __init__ to avoid JudgeClient import requirement
    router = object.__new__(JudgeRouter)
    router.cfg = cfg
    router._client = None  # not needed for parsing
    from tracerigor.judge.audit import AuditQueue
    router._audit = AuditQueue(cfg.audit)
    router._train_step = 0
    return router


def test_router_parse_yes_no_json():
    """Test that the router correctly parses the yes_no JSON format."""
    from tracerigor.judge.schema import TurnJudgePacket

    router = _make_router_no_client()
    packet = TurnJudgePacket(env_id="e0", episode_step=1, task_name="sokoban")

    raw = {
        "success": True,
        "response": json.dumps({
            "observation_grounding": {"yes_no": "YES", "evidence": "Positions match the grid"},
            "action_coherence": {"yes_no": "YES", "evidence": "Action follows reasoning"},
            "temporal_consistency": {"yes_no": "NO", "evidence": "Contradicts prior step"},
        }),
        "model": "test-model",
    }
    resp = router._parse_llm_response(packet, raw)
    assert resp.query_success is True
    assert resp.parse_success is True
    assert resp.rubrics["observation_grounding"].score == 1.0
    assert resp.rubrics["observation_grounding"].label == "pass"
    assert resp.rubrics["action_coherence"].score == 1.0
    assert resp.rubrics["temporal_consistency"].score == 0.0
    assert resp.rubrics["temporal_consistency"].label == "fail"
    assert resp.overall_confidence > 0


def test_router_parse_with_aliases():
    """Test that aliases (e.g. 'grounding' → 'observation_grounding') work."""
    from tracerigor.judge.schema import TurnJudgePacket

    router = _make_router_no_client()
    packet = TurnJudgePacket(env_id="e0", episode_step=1, task_name="sokoban")

    # LLM returns the alias keys (old style) instead of canonical names
    raw = {
        "success": True,
        "response": json.dumps({
            "grounding": {"yes_no": "YES", "evidence": "ok"},
            "action_coherence": {"yes_no": "NO", "evidence": "bad"},
            "history_consistency": {"yes_no": "YES", "evidence": "ok"},
        }),
        "model": "test-model",
    }
    resp = router._parse_llm_response(packet, raw)
    assert resp.parse_success is True
    # "grounding" should match via alias to "observation_grounding"
    assert "observation_grounding" in resp.rubrics
    assert resp.rubrics["observation_grounding"].score == 1.0
    # "history_consistency" should match via alias to "temporal_consistency"
    assert "temporal_consistency" in resp.rubrics
    assert resp.rubrics["temporal_consistency"].score == 1.0


def test_router_fallback_parse():
    """Test that <answer>YES|NO</answer> fallback works."""
    from tracerigor.judge.schema import TurnJudgePacket

    router = _make_router_no_client()
    packet = TurnJudgePacket(env_id="e0", episode_step=1, task_name="sokoban")

    raw = {
        "success": True,
        "response": "<think>The agent is correct.</think><answer>YES</answer>",
        "model": "test-model",
    }
    resp = router._parse_llm_response(packet, raw)
    assert resp.parse_success is True
    # All rubrics should get the binary answer
    for rubric_name in router.cfg.rubrics:
        assert rubric_name in resp.rubrics
        assert resp.rubrics[rubric_name].score == 1.0


def test_router_parse_failure():
    """Test handling of unparseable responses."""
    from tracerigor.judge.schema import TurnJudgePacket

    router = _make_router_no_client()
    packet = TurnJudgePacket(env_id="e0", episode_step=1, task_name="sokoban")

    raw = {
        "success": True,
        "response": "I don't understand the question.",
        "model": "test-model",
    }
    resp = router._parse_llm_response(packet, raw)
    # All rubrics should be filled with defaults
    for rubric_name in router.cfg.rubrics:
        assert rubric_name in resp.rubrics
        assert resp.rubrics[rubric_name].label == "uncertain"


def test_router_parse_query_failure():
    """Test handling when the LLM query itself failed."""
    from tracerigor.judge.schema import TurnJudgePacket

    router = _make_router_no_client()
    packet = TurnJudgePacket(env_id="e0", episode_step=1, task_name="sokoban")

    raw = {"success": False, "error": "timeout", "response": ""}
    resp = router._parse_llm_response(packet, raw)
    assert resp.query_success is False
    assert resp.parse_success is False


# ========================================================================
# 9. YAML config consistency
# ========================================================================
def test_yaml_config_rubric_names():
    """Verify default_config.yaml uses consistent rubric naming."""
    import yaml
    with open("tracerigor/judge/default_config.yaml") as f:
        data = yaml.safe_load(f)
    rubrics = data.get("rubrics", [])
    assert "observation_grounding" in rubrics
    assert "grounding" not in rubrics

    rw = data.get("reward", {}).get("rubric_weights", {})
    assert "observation_grounding" in rw
    assert "grounding" not in rw


# ========================================================================
# 10. End-to-end: heuristics → reward
# ========================================================================
def test_e2e_short_circuit_reward():
    """Test: format violation → short-circuit → negative reward."""
    from tracerigor.judge.heuristics import run_heuristics
    from tracerigor.judge.config import JudgeConfig
    from tracerigor.judge.reward import compute_process_reward
    from tracerigor.judge.schema import TurnJudgePacket, JudgeResponse

    cfg = JudgeConfig()
    p = TurnJudgePacket(
        env_id="e0", episode_step=1, task_name="sokoban",
        format_correct=False,
    )
    heur = run_heuristics(p, cfg.heuristics)
    assert heur["should_short_circuit"] is True

    resp = JudgeResponse(
        env_id=p.env_id,
        episode_step=p.episode_step,
        short_circuited=True,
        short_circuit_reason=heur["reason"],
    )
    reward = compute_process_reward(resp, cfg.reward)
    assert reward < 0


def test_e2e_all_pass_reward():
    """Test: all rubrics YES → positive reward."""
    from tracerigor.judge.reward import compute_process_reward
    from tracerigor.judge.config import RewardConfig
    from tracerigor.judge.schema import JudgeResponse, RubricResult

    resp = JudgeResponse(rubrics={
        "observation_grounding": RubricResult(score=1.0, label="pass", confidence=0.9),
        "action_coherence": RubricResult(score=1.0, label="pass", confidence=0.9),
        "temporal_consistency": RubricResult(score=1.0, label="pass", confidence=0.9),
    })
    r = compute_process_reward(resp, RewardConfig())
    assert r > 0


# ========================================================================
# Run all tests
# ========================================================================
def main():
    print("\n=== Judge Package Integration Tests ===\n")

    tests = [
        # Schema
        ("schema.TurnJudgePacket defaults", test_turn_judge_packet_defaults),
        ("schema.RubricResult defaults", test_rubric_result_defaults),
        ("schema.JudgeResponse defaults", test_judge_response_defaults),
        ("schema.JSON schema keys", test_json_schema_keys),
        # Config
        ("config.defaults rubric names", test_config_defaults_rubric_names),
        ("config.reward weights", test_config_reward_weights),
        ("config.from_dict", test_config_from_dict),
        # Heuristics
        ("heuristics.format_violation", test_heuristic_format_violation),
        ("heuristics.no_action_parsed", test_heuristic_no_action_parsed),
        ("heuristics.invalid_action_feedback", test_heuristic_invalid_action_feedback),
        ("heuristics.invalid_action_admissible", test_heuristic_invalid_action_admissible),
        ("heuristics.valid_action_passes", test_heuristic_valid_action_passes),
        ("heuristics.repetition_detection", test_heuristic_repetition_detection),
        ("heuristics.empty_trace", test_heuristic_empty_trace),
        # Reward
        ("reward.all_pass", test_reward_all_pass),
        ("reward.all_fail", test_reward_all_fail),
        ("reward.short_circuit", test_reward_short_circuit),
        ("reward.scores_dict_keys", test_rubric_scores_dict_keys),
        # Prompt / templates
        ("prompt.template_registration", test_env_template_registration),
        ("prompt.build_messages_sokoban", test_build_messages_sokoban),
        ("prompt.build_messages_sciworld", test_build_messages_sciworld),
        ("prompt.build_messages_fallback", test_build_messages_default_fallback),
        # Env templates
        ("env_templates.sciworld_ground_truth", test_sciworld_ground_truth_builder),
        ("env_templates.sokoban_ground_truth", test_sokoban_ground_truth_builder),
        ("env_templates.dispatch", test_ground_truth_dispatch),
        # Packet builder
        ("packet_builder.sokoban", test_build_turn_packet_sokoban),
        ("packet_builder.sciworld", test_build_turn_packet_sciworld),
        ("packet_builder.format_error", test_build_turn_packet_format_error),
        # Router parsing
        ("router.parse_yes_no_json", test_router_parse_yes_no_json),
        ("router.parse_with_aliases", test_router_parse_with_aliases),
        ("router.fallback_parse", test_router_fallback_parse),
        ("router.parse_failure", test_router_parse_failure),
        ("router.parse_query_failure", test_router_parse_query_failure),
        # YAML config
        ("yaml.rubric_names", test_yaml_config_rubric_names),
        # End-to-end
        ("e2e.short_circuit_reward", test_e2e_short_circuit_reward),
        ("e2e.all_pass_reward", test_e2e_all_pass_reward),
    ]

    for name, fn in tests:
        _run(name, fn)

    print(f"\n--- Results: {_results['pass']} passed, {_results['fail']} failed ---\n")
    sys.exit(0 if _results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
