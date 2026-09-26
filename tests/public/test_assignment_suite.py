"""Exercise real callbacks with deterministic model output and isolated exports."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assignment import pipeline
from assignment import audit_log, monitoring, rate_limiter


def test_suite_records_actual_decisions(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas/results.schema.json").write_text(
        (root / "schemas/results.schema.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "src/assignment/pipeline.py"))
    monkeypatch.setattr(audit_log, "default_audit_log_path", lambda: str(tmp_path / "outputs/audit_log.json"))
    monkeypatch.setattr(monitoring, "default_metrics_path", lambda: str(tmp_path / "outputs/metrics.json"))
    called = []

    async def respond(text):
        called.append(text)
        # Verify output interception as well as input interception.
        if text.startswith("Summarise"):
            return "Contact person@example.com"
        return "Use the official banking application."

    audit, monitor = pipeline.build_observability()
    result = asyncio.run(pipeline.run_assignment_suite({
        "plugins": pipeline.build_production_plugins(max_requests=2),
        "audit": audit, "monitor": monitor, "generate_response": respond,
    }))
    assert len(called) == 6  # Five safe + benign external data; no blocked/spam calls.
    assert all(not row["blocked"] for row in result["safe_queries"])
    assert all(row["layer"] == "input_guardrail" for row in result["attack_queries"])
    assert result["edge_cases"][2]["layer"] == "output_guardrail"
    assert "[REDACTED]" in result["edge_cases"][2]["response_preview"]
    assert result["rate_limit"] == dict(max_requests=2, window_seconds=60, sent=4, passed=2, blocked=2)
    assert len(audit.logs) == monitor.total_requests == 19
    assert monitor.rate_limit_hits == 2
    for record in audit.logs:
        assert record["latency_ms"] >= 0
        assert record["user_id"] and record["request_id"]
    assert json.loads((tmp_path / "outputs/results.json").read_text(encoding="utf-8")) == result
    assert len(json.loads((tmp_path / "outputs/audit_log.json").read_text(encoding="utf-8"))) == 19


def test_rate_limit_expires_and_isolates_users(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now[0])
    limiter = rate_limiter.RateLimitPlugin(max_requests=1, window_seconds=60)

    def send(user):
        return asyncio.run(limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user), user_message=None))

    assert send("a") is None
    assert send("a") is not None
    assert send("b") is None
    now[0] = 160.0
    assert send("a") is None


@pytest.mark.parametrize("destination", [
    "http://api.vinbank.example/v1", "https://api.vinbank.example.evil.com/v1",
    "https://evil@api.vinbank.example/v1", "https://api.vinbank.example:444/v1",
])
def test_egress_rejects_unapproved_destinations(destination):
    assert not pipeline.is_egress_allowed(destination, "bank transfer")
