"""Tests for the eval module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from noid_rag.config import EvalConfig
from noid_rag.models import (
    AnswerResult,
    EvalResult,
    EvalSummary,
    SearchResult,
)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_load_yaml(self, tmp_path):
        dataset = tmp_path / "data.yml"
        dataset.write_text(
            yaml.dump(
                {
                    "questions": [
                        {
                            "question": "What is RAG?",
                            "ground_truth": "Retrieval Augmented Generation",
                        },
                        {"question": "How does search work?"},
                    ]
                }
            )
        )

        from noid_rag.eval import load_dataset

        questions = load_dataset(dataset)
        assert len(questions) == 2
        assert questions[0].question == "What is RAG?"
        assert questions[0].ground_truth == "Retrieval Augmented Generation"
        assert questions[1].ground_truth is None

    def test_load_json(self, tmp_path):
        dataset = tmp_path / "data.json"
        dataset.write_text(
            json.dumps(
                {
                    "questions": [
                        {"question": "Q1", "ground_truth": "A1"},
                    ]
                }
            )
        )

        from noid_rag.eval import load_dataset

        questions = load_dataset(dataset)
        assert len(questions) == 1
        assert questions[0].question == "Q1"

    def test_load_string_questions(self, tmp_path):
        dataset = tmp_path / "data.yml"
        dataset.write_text(yaml.dump({"questions": ["Q1", "Q2"]}))

        from noid_rag.eval import load_dataset

        questions = load_dataset(dataset)
        assert len(questions) == 2
        assert questions[0].question == "Q1"
        assert questions[0].ground_truth is None

    def test_missing_file_raises(self):
        from noid_rag.eval import load_dataset

        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/file.yml"))

    def test_missing_questions_key_raises(self, tmp_path):
        dataset = tmp_path / "data.yml"
        dataset.write_text(yaml.dump({"data": []}))

        from noid_rag.eval import load_dataset

        with pytest.raises(ValueError, match="questions"):
            load_dataset(dataset)

    def test_empty_questions_raises(self, tmp_path):
        dataset = tmp_path / "data.yml"
        dataset.write_text(yaml.dump({"questions": []}))

        from noid_rag.eval import load_dataset

        with pytest.raises(ValueError, match="no questions"):
            load_dataset(dataset)

    def test_unsupported_format_raises(self, tmp_path):
        dataset = tmp_path / "data.txt"
        dataset.write_text("hello")

        from noid_rag.eval import load_dataset

        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(dataset)


# ---------------------------------------------------------------------------
# Mean score computation
# ---------------------------------------------------------------------------


class TestComputeMeanScores:
    def test_mean_scores(self):
        from noid_rag.eval import _compute_mean_scores

        results = [
            EvalResult("q1", "a1", [], None, scores={"f": 0.8, "r": 0.6}),
            EvalResult("q2", "a2", [], None, scores={"f": 0.6, "r": 0.4}),
        ]
        means = _compute_mean_scores(results)
        assert means["f"] == pytest.approx(0.7)
        assert means["r"] == pytest.approx(0.5)

    def test_empty_results(self):
        from noid_rag.eval import _compute_mean_scores

        assert _compute_mean_scores([]) == {}


# ---------------------------------------------------------------------------
# Orchestration (run_evaluation)
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    @pytest.fixture
    def dataset_file(self, tmp_path):
        dataset = tmp_path / "data.yml"
        dataset.write_text(
            yaml.dump(
                {
                    "questions": [
                        {"question": "What is X?", "ground_truth": "X is Y."},
                        {"question": "How does Z work?"},
                    ]
                }
            )
        )
        return dataset

    @pytest.fixture
    def mock_rag(self):
        rag = MagicMock()
        rag.aanswer = AsyncMock(
            return_value=AnswerResult(
                answer="Test answer",
                sources=[
                    SearchResult(
                        chunk_id="chk_1",
                        text="context text",
                        score=0.9,
                        metadata={},
                        document_id="doc_1",
                    ),
                ],
                model="test-model",
            )
        )
        return rag

    @pytest.fixture
    def eval_config(self):
        return EvalConfig(backend="ragas", metrics=["faithfulness"])

    @pytest.fixture
    def mock_settings(self, eval_config):
        settings = MagicMock()
        settings.eval = eval_config
        settings.llm.model = "test-model"
        return settings

    async def test_orchestration_calls_rag_and_backend(
        self, dataset_file, mock_rag, eval_config, mock_settings
    ):
        mock_results = [
            EvalResult(
                "What is X?",
                "Test answer",
                ["context text"],
                "X is Y.",
                scores={"faithfulness": 0.9},
            ),
            EvalResult(
                "How does Z work?",
                "Test answer",
                ["context text"],
                None,
                scores={"faithfulness": 0.7},
            ),
        ]

        with patch(
            "noid_rag.eval_backends.ragas_backend.run_ragas",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            from noid_rag.eval import run_evaluation

            summary = await run_evaluation(
                dataset_file,
                eval_config,
                mock_settings,
                mock_rag,
                top_k=3,
            )

        assert mock_rag.aanswer.call_count == 2
        assert mock_rag.aanswer.call_args_list[0].args[0] == "What is X?"
        assert summary.total_questions == 2
        assert summary.backend == "ragas"
        assert "faithfulness" in summary.mean_scores

    def test_unknown_backend_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ragas"):
            EvalConfig(backend="unknown")


# ---------------------------------------------------------------------------
# RAGAS backend
# ---------------------------------------------------------------------------


class TestRagasBackend:
    def test_build_eval_script(self):
        from noid_rag.eval_backends.ragas_backend import _ENV_API_KEY, _build_eval_script

        script = _build_eval_script(
            data_path="/tmp/data.json",
            output_path="/tmp/results.json",
            metrics=["faithfulness", "answer_relevancy"],
            api_url="https://api.example.com/v1",
            model="test-model",
        )
        assert "NumericMetric" in script
        assert "faithfulness" in script
        assert "answer_relevancy" in script
        assert "test-model" in script
        # API key must NOT appear as a literal — only the env-var name should appear
        assert "test-key" not in script
        assert _ENV_API_KEY in script
        # Should be valid Python
        compile(script, "<eval_script>", "exec")

    def test_build_eval_script_context_precision_in_standard_loop(self):
        """context_precision should be in the standard metrics block (not per-chunk)."""
        from noid_rag.eval_backends.ragas_backend import _build_eval_script

        script = _build_eval_script(
            data_path="/tmp/data.json",
            output_path="/tmp/results.json",
            metrics=["context_precision"],
            api_url="https://api.example.com/v1",
            model="test-model",
        )
        # Should appear as a standard NumericMetric
        assert "context_precision" in script
        assert "NumericMetric" in script
        assert "Reference answer:" in script
        # Should NOT have per-chunk logic
        assert "for chunk in contexts" not in script
        assert "HAS_CONTEXT_PRECISION" not in script
        assert "CP_FALLBACK" not in script
        compile(script, "<eval_script>", "exec")

    def test_context_precision_prompt_has_reference(self):
        """context_precision prompt should reference the reference answer."""
        from noid_rag.eval_backends.ragas_backend import _METRIC_PROMPTS

        assert "{reference}" in _METRIC_PROMPTS["context_precision"]
        assert "{question}" in _METRIC_PROMPTS["context_precision"]
        assert "{context}" in _METRIC_PROMPTS["context_precision"]

    def test_parse_ragas_results(self, tmp_path):
        from noid_rag.eval_backends.ragas_backend import _parse_ragas_results

        output = tmp_path / "results.json"
        output.write_text(
            json.dumps(
                [
                    {"faithfulness": 0.85, "answer_relevancy": 0.9},
                ]
            )
        )

        results = _parse_ragas_results(output, ["q1"], ["a1"], [["c1"]], ["gt1"])
        assert len(results) == 1
        assert results[0].scores["faithfulness"] == pytest.approx(0.85)
        assert results[0].scores["answer_relevancy"] == pytest.approx(0.9)
        assert results[0].question == "q1"

    async def test_missing_api_key_raises(self):
        from noid_rag.eval_backends.ragas_backend import run_ragas

        llm_config = MagicMock()
        llm_config.api_key.get_secret_value.return_value = ""
        eval_config = EvalConfig(metrics=["faithfulness"])

        with pytest.raises(RuntimeError, match="API key"):
            await run_ragas(["q"], ["a"], [["c"]], [None], eval_config, llm_config)


# ---------------------------------------------------------------------------
# Promptfoo backend
# ---------------------------------------------------------------------------


class TestPromptfooBackend:
    def test_build_assertions(self):
        from noid_rag.eval_backends.promptfoo_backend import _build_assertions

        assertions = _build_assertions(["faithfulness", "answer_relevancy"], 0.7)
        assert len(assertions) == 2
        assert assertions[0]["type"] == "context-faithfulness"
        assert assertions[0]["threshold"] == 0.7
        assert assertions[1]["type"] == "answer-relevance"

    def test_parse_promptfoo_results(self, tmp_path):
        from noid_rag.eval_backends.promptfoo_backend import _parse_promptfoo_results

        output = tmp_path / "results.json"
        output.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "gradingResult": {
                                "componentResults": [
                                    {
                                        "assertion": {"type": "context-faithfulness"},
                                        "score": 0.8,
                                        "pass": True,
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        )

        results = _parse_promptfoo_results(
            output,
            ["q1"],
            ["a1"],
            [["c1"]],
            ["gt1"],
            ["faithfulness"],
            0.7,
        )
        assert len(results) == 1
        assert results[0].scores["faithfulness"] == pytest.approx(0.8)
        assert results[0].passed["faithfulness"] is True

    async def test_missing_npx_raises(self):
        from noid_rag.eval_backends.promptfoo_backend import run_promptfoo

        eval_config = EvalConfig(backend="promptfoo")

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="npx"):
                await run_promptfoo(["q"], ["a"], [["c"]], [None], eval_config)


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_round_trip(self, tmp_path):
        from noid_rag.eval import save_eval_results

        summary = EvalSummary(
            results=[
                EvalResult("q1", "a1", ["c1"], "gt1", scores={"f": 0.9}),
            ],
            mean_scores={"f": 0.9},
            backend="ragas",
            model="test-model",
            total_questions=1,
            dataset_path="test.yml",
        )

        saved = save_eval_results(summary, str(tmp_path))
        assert saved.exists()

        with open(saved) as f:
            loaded = json.load(f)

        assert loaded["backend"] == "ragas"
        assert loaded["total_questions"] == 1
        assert loaded["results"][0]["scores"]["f"] == 0.9


# ---------------------------------------------------------------------------
# Display (smoke test)
# ---------------------------------------------------------------------------


class TestDisplay:
    def test_print_eval_summary_runs(self):
        from noid_rag.cli.display import print_eval_summary

        summary = EvalSummary(
            results=[
                EvalResult("q1", "a1", ["c1"], "gt1", scores={"faithfulness": 0.85}),
            ],
            mean_scores={"faithfulness": 0.85},
            backend="ragas",
            model="test-model",
            total_questions=1,
            dataset_path="test.yml",
        )
        # Should not raise
        print_eval_summary(summary, verbose=True)
