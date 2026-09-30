"""vLLM pooler that turns option-endpoint hidden states into candidate logits.

It serves the ``token_classify`` pooling task: the built-in task whose outputs
vary in length per request, which both vLLM model runners accept (Model Runner
V2 rejects the ``plugin`` task). Each finished request gets a 1-D FP32 tensor
with one raw logit per candidate, in the request's candidate order; activation,
temperature and answer formatting happen in the endpoint. The rows a request
needs are copied out of every prefill step it takes part in, so chunked prefill
is supported; the copies own their storage, which async scheduling requires. A
request without valid positions (for example a raw ``/pooling`` call) gets a
single NaN instead of failing the engine step.
"""

from __future__ import annotations

from collections.abc import Set

import torch

from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .gather import POOLING_TASK, parse_positions, plan_step
from .head import CandidateHead


class CandidatePooler(Pooler):
    def __init__(self, head: CandidateHead):
        super().__init__()
        # The model owns the head's parameters; keep a plain reference here so
        # they are registered, loaded and reported under one name only.
        object.__setattr__(self, "head", head)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {POOLING_TASK}

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> PoolerOutput:
        cursor = pooling_metadata.get_pooling_cursor()
        prompt_lens = cursor.prompt_lens_cpu.tolist()
        positions = [
            parse_positions(params.extra_kwargs, prompt_len)
            for params, prompt_len in zip(pooling_metadata.pooling_params, prompt_lens)
        ]
        plan = plan_step(
            cursor.num_scheduled_tokens_cpu.tolist(),
            cursor.seq_lens_cpu.tolist(),
            prompt_lens,
            positions,
        )
        if plan.rows:
            index = torch.tensor(plan.rows, dtype=torch.long).to(
                hidden_states.device, non_blocking=True
            )
            taken = torch.split(hidden_states.index_select(0, index), plan.counts)
        else:
            taken = [None] * len(plan.counts)

        outputs: list[torch.Tensor | None] = [None] * len(plan.counts)
        ready: list[tuple[int, torch.Tensor]] = []
        for i, state in enumerate(pooling_metadata.pooling_states):
            if plan.counts[i]:
                state.hidden_states_cache.append(taken[i])
            if not plan.finished[i]:
                continue
            cache = state.hidden_states_cache
            rows = (
                cache[0] if len(cache) == 1 else (torch.cat(cache) if cache else None)
            )
            state.clean()
            if (
                positions[i] is None
                or rows is None
                or rows.shape[0] != len(positions[i].rows)
            ):
                outputs[i] = torch.full(
                    (1,), float("nan"), dtype=torch.float32, device=hidden_states.device
                )
            else:
                ready.append((i, rows))
        if ready:
            counts = [rows.shape[0] - 1 for _, rows in ready]
            candidates = torch.cat([rows[:-1] for _, rows in ready])
            queries = torch.stack([rows[-1] for _, rows in ready])
            owner = torch.repeat_interleave(
                torch.arange(len(ready)), torch.tensor(counts)
            ).to(queries.device, non_blocking=True)
            scores = self.head.score_flat(candidates, queries, owner)
            for (i, _), values in zip(ready, torch.split(scores, counts)):
                outputs[i] = values
        return outputs
