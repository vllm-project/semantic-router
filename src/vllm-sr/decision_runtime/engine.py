"""Public response adaptation for backend-native Decision predictions."""

from __future__ import annotations

import math
from numbers import Real

from .backend import (
    BatchDecisionBackend,
    BackendBatchRequest,
    BackendBatchResult,
    BackendContractError,
    BackendPrediction,
    BackendResult,
    DecisionBackend,
    ModelDescriptor,
    UnknownModelError,
)
from .confidence import normalized_top_confidence
from .contracts import (
    PROBABILITY_SUM_TOLERANCE,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneBatchRequest,
    SystemOneBatchResponse,
    SystemOneBatchResult,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    validate_batch_response_for_request,
    validate_response_for_request,
)


class DecisionEngine:
    """Validate backend output and produce the strict public response."""

    def __init__(self, backend: DecisionBackend) -> None:
        self._backend = backend
        models = tuple(backend.models())
        if not models:
            raise ValueError("the backend must expose at least one model")
        if any(
            not isinstance(model.name, str)
            or not model.name.strip()
            or model.name != model.name.strip()
            for model in models
        ):
            raise ValueError("backend model names must be non-empty, trimmed strings")
        if len({model.name for model in models}) != len(models):
            raise ValueError("the backend must expose unique model names")
        self._models = models
        self._model_names = {model.name for model in models}

    @property
    def models(self) -> tuple[ModelDescriptor, ...]:
        return self._models

    async def ready(self) -> bool:
        return await self._backend.ready()

    async def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        self._require_model(request.model)
        try:
            result = await self._backend.infer(request)
        except UnknownModelError as exc:
            raise BackendContractError(
                "backend rejected a model in its advertised inventory"
            ) from exc
        return self._response_from_result(request, result)

    async def evaluate_batch(
        self, request: SystemOneBatchRequest
    ) -> SystemOneBatchResponse:
        """Evaluate a shared-question batch atomically through one backend seam."""

        self._require_model(request.model)
        backend_requests = tuple(
            BackendBatchRequest(
                state_id=state.id,
                request=SystemOneRequest(
                    state=state.state,
                    model=request.model,
                    questions=request.questions,
                ),
            )
            for state in request.states
        )
        if not isinstance(self._backend, BatchDecisionBackend):
            raise BackendContractError("backend does not implement batch inference")
        try:
            backend_results = await self._backend.infer_batch(backend_requests)
        except UnknownModelError as exc:
            raise BackendContractError(
                "backend rejected a model in its advertised inventory"
            ) from exc
        if not isinstance(backend_results, tuple) or not all(
            isinstance(item, BackendBatchResult) for item in backend_results
        ):
            raise BackendContractError(
                "backend batch results must be BackendBatchResult values"
            )
        if [item.state_id for item in backend_results] != [
            item.state_id for item in backend_requests
        ]:
            raise BackendContractError("backend batch state identity or order changed")

        results = []
        for backend_request, backend_result in zip(
            backend_requests, backend_results, strict=True
        ):
            single_response = self._response_from_result(
                backend_request.request,
                backend_result.result,
            )
            results.append(
                SystemOneBatchResult(
                    id=backend_request.state_id,
                    answers=single_response.answers,
                    usage=single_response.usage,
                )
            )

        response = SystemOneBatchResponse(
            model=request.model,
            results=results,
            usage=Usage(
                input_tokens=sum(item.usage.input_tokens for item in results),
                output_tokens=sum(item.usage.output_tokens for item in results),
            ),
        )
        return validate_batch_response_for_request(request, response)

    def _require_model(self, model: str) -> None:
        if model not in self._model_names:
            raise UnknownModelError(model)

    def _response_from_result(
        self,
        request: SystemOneRequest,
        result: BackendResult,
    ) -> SystemOneResponse:
        if not isinstance(result, BackendResult):
            raise BackendContractError("backend result must use BackendResult")
        if not isinstance(result.model, str):
            raise BackendContractError("backend model identity must be a string")
        if result.model != request.model:
            raise BackendContractError("backend changed the requested model")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (result.input_tokens, result.output_tokens)
        ):
            raise BackendContractError("backend token usage must be non-negative")
        if not isinstance(result.predictions, tuple) or not all(
            isinstance(prediction, BackendPrediction)
            for prediction in result.predictions
        ):
            raise BackendContractError(
                "backend predictions must be BackendPrediction values"
            )
        if [prediction.question_id for prediction in result.predictions] != list(
            request.questions
        ):
            raise BackendContractError("backend question identity or order changed")

        answers = {}
        for prediction in result.predictions:
            question = request.questions[prediction.question_id]
            probabilities = self._validate_prediction(prediction, question)
            if isinstance(question, NoulQuestion):
                answers[prediction.question_id] = NoulAnswer(
                    type="noul", noul=probabilities[1]
                )
            elif isinstance(question, ChoiceQuestion):
                labels = list(question.criteria)
                distribution = dict(zip(labels, probabilities, strict=True))
                winner = max(range(len(labels)), key=probabilities.__getitem__)
                answers[prediction.question_id] = ChoiceAnswer(
                    type="choice",
                    choice=labels[winner],
                    confidence=normalized_top_confidence(probabilities),
                    probabilities=distribution,
                )
            else:
                labels = [str(index) for index in range(len(question.criteria))]
                distribution = dict(zip(labels, probabilities, strict=True))
                answers[prediction.question_id] = ScoreAnswer(
                    type="score",
                    score=math.fsum(
                        index * probability
                        for index, probability in enumerate(probabilities)
                    ),
                    confidence=normalized_top_confidence(probabilities),
                    legend={
                        str(index): criterion
                        for index, criterion in enumerate(question.criteria)
                    },
                    probabilities=distribution,
                )

        response = SystemOneResponse(
            model=result.model,
            answers=answers,
            usage=Usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
        )
        return validate_response_for_request(request, response)

    @staticmethod
    def _validate_prediction(
        prediction: BackendPrediction,
        question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
    ) -> tuple[float, ...]:
        expected_count = 2
        if isinstance(question, (ChoiceQuestion, ScoreQuestion)):
            expected_count = len(question.criteria)
        if prediction.type != question.type:
            raise BackendContractError("backend answer type changed")
        if not isinstance(prediction.probabilities, tuple):
            raise BackendContractError(
                "backend probabilities must be an immutable numeric tuple"
            )
        if len(prediction.probabilities) != expected_count:
            raise BackendContractError("backend probability count changed")
        if not all(
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
            for value in prediction.probabilities
        ):
            raise BackendContractError(
                "backend probabilities must be finite numeric values"
            )
        probabilities = tuple(float(value) for value in prediction.probabilities)
        if not math.isclose(
            math.fsum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_TOLERANCE,
        ):
            raise BackendContractError("backend probabilities must sum to one")
        return probabilities
