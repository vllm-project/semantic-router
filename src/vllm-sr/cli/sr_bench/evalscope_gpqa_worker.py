from __future__ import annotations

import hashlib
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

from evalscope.api.dataset import LocalDataLoader, Sample
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, Model, ModelAPI, ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.benchmarks.gpqa.gpqa_adapter import GPQAAdapter
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.models.utils.openai import openai_chat_messages
from evalscope.utils.model_utils import seed_everything

from .dataset_io import MAX_SCAN_BYTES, read_small
from .router_flow_gpqa import (
    BENCHMARK,
    EVALSCOPE_VERSION,
    NATIVE_BENCHMARK,
    PROTOCOL,
    SAMPLING,
    SEED,
    SMOKE_LIMIT,
    validate_case,
)
from .transport import CallFailure


def check_version() -> None:
    if version("evalscope") != EVALSCOPE_VERSION:
        raise ValueError(f"Router Flow GPQA requires EvalScope {EVALSCOPE_VERSION}")


def _native_adapter() -> GPQAAdapter:
    native = get_benchmark(
        name=NATIVE_BENCHMARK,
        config=TaskConfig(
            model="sr-bench-subject",
            eval_type=EvalType.OPENAI_API,
            datasets=[NATIVE_BENCHMARK],
            generation_config=SAMPLING,
            judge_strategy=JudgeStrategy.RULE,
        ),
    )
    if not isinstance(native, GPQAAdapter):
        raise ValueError("EvalScope registered an unexpected GPQA adapter")
    return native


def prepare_cases(*, source_path: Path, limit: int) -> list[dict[str, Any]]:
    check_version()
    if type(limit) is not int or not 1 <= limit <= SMOKE_LIMIT:
        raise ValueError(f"smoke limit must be between 1 and {SMOKE_LIMIT}")
    source_sha256 = hashlib.sha256(read_small(source_path, MAX_SCAN_BYTES)).hexdigest()
    seed_everything(SEED)
    native = _native_adapter()
    samples = LocalDataLoader(
        data_id_or_path=str(source_path),
        split=native.eval_split,
        subset=native.default_subset,
        sample_fields=native.record_to_sample,
        limit=limit,
        shuffle=False,
    ).load()
    if len(samples) != limit:
        raise ValueError(f"Native source has {len(samples)} cases; need {limit}")
    cases: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample.input, str):
            raise ValueError("Router Flow GPQA requires native text questions")
        case = {
            "id": f"{BENCHMARK}/{source_sha256[:16]}/{sample.id}",
            "benchmark": BENCHMARK,
            "messages": openai_chat_messages(
                native.process_sample_str_input(sample=sample, subset="default")
            ),
            "answer": sample.target,
            "metadata": {
                "protocol": PROTOCOL,
                "source_index": index,
                "native_sample_id": sample.id,
                "source_sha256": source_sha256,
                "question": sample.input,
                "question_sha256": hashlib.sha256(sample.input.encode()).hexdigest(),
                "choices": sample.choices,
                "split": "dev",
            },
        }
        validate_case(case)
        cases.append(case)
    if (
        hashlib.sha256(read_small(source_path, MAX_SCAN_BYTES)).hexdigest()
        != source_sha256
    ):
        raise ValueError("Native source changed during preparation")
    return cases


class _JournalModel(ModelAPI):
    def __init__(
        self, bridge: Callable[[list[dict[str, Any]]], dict[str, Any]]
    ) -> None:
        super().__init__(model_name="sr-bench-subject")
        self.bridge = bridge
        self.called = False

    def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if self.called or tools or tool_choice != "none":
            raise ValueError("Router Flow GPQA permits one subject call without tools")
        if (
            config.temperature != SAMPLING["temperature"]
            or config.max_tokens != SAMPLING["max_tokens"]
        ):
            raise ValueError("Native generation differs from the frozen diagnostic")
        self.called = True
        response = self.bridge(openai_chat_messages(input))
        if response.get("output_complete") is not True:
            raise CallFailure("EvalScope diagnostic received incomplete output")
        if not isinstance(response.get("final"), str) or response.get("tool_calls"):
            raise ValueError("Router Flow GPQA requires a completed final text answer")
        return ModelOutput.from_content(
            model=response["model"],
            content=response["final"],
            stop_reason="stop",
        )


def execute(
    *,
    request: dict[str, Any],
    bridge: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    check_version()
    case = request["case"]
    validate_case(case)
    metadata = case["metadata"]
    native = _native_adapter()
    sample = Sample(
        id=metadata["native_sample_id"],
        input=metadata["question"],
        choices=metadata["choices"],
        target=case["answer"],
    )
    sample.input = native.process_sample_str_input(sample=sample, subset="default")
    if openai_chat_messages(sample.input) != case["messages"]:
        raise ValueError("Frozen prompt differs from native GPQA formatting")
    state = native.run_inference(
        model=Model(
            api=_JournalModel(bridge=bridge), config=GenerateConfig(**SAMPLING)
        ),
        sample=sample,
        output_dir=request.get("artifact_dir", ""),
    )
    graded = native.calculate_metrics(task_state=state).score
    score = graded.value["acc"]
    if score not in (0, 1):
        raise ValueError("Native GPQA did not return binary accuracy")
    return {
        "answer": graded.extracted_prediction,
        "correct": score == 1,
        "score": float(score),
        "details": {
            "grader_version": PROTOCOL,
            "evalscope_version": EVALSCOPE_VERSION,
            "native_metric": "acc",
            "native_sample_id": metadata["native_sample_id"],
        },
    }
