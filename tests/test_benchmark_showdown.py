"""Tests for Memanto Benchmarking Showdown (Issue #639)."""

import json
import pytest
from memanto.cli.analyze.benchmark_showdown import (
    ASSUMPTIONS,
    ScenarioAResult,
    ScenarioBResult,
    run_scenario_a,
    run_scenario_b,
    compute_metrics,
    build_llm_prompt,
    build_report_markdown,
)


class TestScenarioA:
    def test_default_parameters(self):
        result = run_scenario_a()
        assert isinstance(result, ScenarioAResult)
        assert result.sessions == ASSUMPTIONS["dense_sessions"]
        assert result.total_messages == result.sessions * ASSUMPTIONS["messages_per_session"]
        assert result.latency_speedup_x > 1.0
        assert result.token_savings_per_turn > 0
        assert result.total_token_savings > 0
        assert result.write_availability_saved_ms > 0

    def test_custom_parameters(self):
        result = run_scenario_a(sessions=10, messages_per_session=50, avg_chars_per_message=500)
        assert result.sessions == 10
        assert result.total_messages == 500
        assert result.total_tokens > 0

    def test_memanto_faster_than_competitor(self):
        result = run_scenario_a()
        assert result.memanto_read_ms < result.competitor_read_ms
        assert result.memanto_write_ms < result.competitor_write_ms
        assert result.memanto_retrieval_tokens_per_turn < result.competitor_retrieval_tokens_per_turn

    def test_token_savings_positive(self):
        result = run_scenario_a()
        assert result.token_savings_per_turn == (
            result.competitor_retrieval_tokens_per_turn
            - result.memanto_retrieval_tokens_per_turn
        )


class TestScenarioB:
    def test_default_parameters(self):
        result = run_scenario_b()
        assert isinstance(result, ScenarioBResult)
        assert result.sessions == ASSUMPTIONS["preference_sessions"]
        assert result.mutations > 0
        assert 0 <= result.memanto_precision <= 1.0
        assert 0 <= result.competitor_precision <= 1.0

    def test_memanto_better_precision(self):
        result = run_scenario_b()
        assert result.memanto_precision > result.competitor_precision
        assert result.memanto_staleness_rate > result.competitor_staleness_rate

    def test_context_pollution_savings(self):
        result = run_scenario_b()
        assert result.memanto_context_facts < result.competitor_context_facts
        assert result.context_savings_facts > 0

    def test_custom_parameters(self):
        result = run_scenario_b(sessions=16, mutations_per_session=5, total_facts=100)
        assert result.sessions == 16
        assert result.total_facts == 100
        assert result.mutations == 80


class TestComputeMetrics:
    def test_complete_metrics_structure(self):
        metrics = compute_metrics()
        assert "scenario_a" in metrics
        assert "scenario_b" in metrics
        assert "ingestion_tax" in metrics
        assert "meta" in metrics

        sa = metrics["scenario_a"]
        assert sa["name"] == "Context-Overhead & Latency Sprint"
        assert "memanto" in sa
        assert "competitor" in sa
        assert "deltas" in sa
        assert sa["deltas"]["latency_speedup_x"] > 1.0

        sb = metrics["scenario_b"]
        assert sb["name"] == "Shifting Persona & Temporal Tracking"
        assert sb["deltas"]["precision_improvement"] > 0
        assert sb["deltas"]["staleness_improvement"] > 0

    def test_ingestion_cost_savings(self):
        metrics = compute_metrics()
        tax = metrics["ingestion_tax"]
        assert tax["mem0_cost_usd"] > 0
        assert tax["memanto_cost_usd"] == 0.0
        assert tax["cost_saved_usd"] == tax["mem0_cost_usd"]

    def test_with_precomputed_scenarios(self):
        a = run_scenario_a(sessions=3, messages_per_session=10)
        b = run_scenario_b(sessions=4, mutations_per_session=2, total_facts=30)
        metrics = compute_metrics(scenario_a=a, scenario_b=b)
        assert metrics["scenario_a"]["sessions"] == 3
        assert metrics["scenario_b"]["sessions"] == 4

    def test_meta_includes_assumptions(self):
        metrics = compute_metrics()
        assert "assumptions" in metrics["meta"]
        assert metrics["meta"]["assumptions"]["chars_per_token"] == 4


class TestBuildLLMPrompt:
    def test_prompt_contains_key_numbers(self):
        metrics = compute_metrics()
        prompt = build_llm_prompt(metrics)
        assert "Scenario A" in prompt
        assert "Scenario B" in prompt
        assert "tokens" in prompt.lower()
        assert "latency" in prompt.lower()

    def test_prompt_not_empty(self):
        metrics = compute_metrics()
        prompt = build_llm_prompt(metrics)
        assert len(prompt) > 500


class TestBuildReport:
    def test_report_markdown_structure(self):
        metrics = compute_metrics()
        report = build_report_markdown(
            metrics=metrics,
            narrative="The Memanto architecture demonstrates...",
            competitor_name="Mem0",
        )
        assert "# 🐜 Memanto Benchmarking Showdown" in report
        assert "Scenario A" in report
        assert "Scenario B" in report
        assert "Ingestion Tax" in report
        assert "AI Analysis" in report
        assert "Method & Assumptions" in report
        assert "Mem0" in report

    def test_report_contains_table_headers(self):
        metrics = compute_metrics()
        report = build_report_markdown(
            metrics=metrics, narrative="Test.", competitor_name="Zep"
        )
        assert "| Metric | Memanto | Competitor | Delta |" in report
        assert "Zep" in report

    def test_report_includes_assumptions(self):
        metrics = compute_metrics()
        report = build_report_markdown(
            metrics=metrics, narrative="Test.", competitor_name="Test"
        )
        assert "chars_per_token" in report
        assert "dense_sessions" in report

    def test_report_contains_delta_values(self):
        metrics = compute_metrics()
        report = build_report_markdown(
            metrics=metrics, narrative="Test.", competitor_name="Test"
        )
        sa = metrics["scenario_a"]
        assert f"{sa['deltas']['latency_speedup_x']}x faster" in report
