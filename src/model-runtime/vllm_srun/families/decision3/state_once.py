"""State-once Decision models: d3 packages whose attention never lets a state token see a question.

A package declares its layout in ``decision_config.json`` ``attention_mode``
(``package.STATE_ONCE_MODES``): ``prefix_bidir`` (the state attends within
itself both ways; a question block attends to the state and within itself
both ways), ``prefix_causal`` (the same with a causal state) or
``causal``. The prompt is the d3 prompt, and its state block is its leading
tokens that end within the state text (``prompt.state_tokens``): the system
turn, the user turn's images and videos, ``State:`` and the state. Each
question block holds the rest: the blank line after the state (with any token
that merges the state's last characters into it), the question, its coded
options and the generation prompt. The split depends on the state alone, and
the reference implementation splits its rows the same way. A ``header-v1``
package's question list (``prompt.header_block``) opens the state block of a
text-only request, so its questions still branch from one state.

A request's state block is tokenized and encoded once and every question
branches from it (``EngineModel.branch``): the images and videos go through
the vision tower once, as part of the state. Branched forwards hold the
questions of one request or of several (the batcher packs whole requests,
``merges_shared_batches``), and every path runs them branched: the golden
check, shared-context batches and batches that merge requests. A lone text
question has nothing to share, so it runs as its row instead, a forward with
fewer kernels. ``run_rows`` runs
each question as its own row under the same mask, the computation the
branched forward reproduces up to rounding (``tools/state_once_parity.py``).
A question costs its question block and its share of the state, which is
what ``usage.input_tokens`` counts: the state once, then each question.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, ClassVar, cast

import torch

from ...errors import (
    INVALID_QUESTION,
    MediaError,
    QuestionError,
    RequestError,
    question_error,
)
from ...plugins.base import BranchBatch, Expired, ForwardBatch, ImageInputs, VideoInputs
from ...plugins.decisions import RequestPlan
from ...systemone import QUESTION_TYPES, read_question, track_rate, valid_state
from . import composed, prompt
from . import videos as vid
from .family import (
    COST_SMOOTHING,
    VISION_TOWER,
    Decision3Model,
    Item,
    RequestMedia,
    RequestState,
)


class _Counted:
    """A readout's code count and temperature, shaped like an item for ``_probabilities`` (it reads ``len(keys)``
    and ``temperature``)."""

    def __init__(self, count: int, temperature: float | None = None) -> None:
        self.keys = range(count)
        self.temperature = temperature


# A lone question gains nothing from branching; it runs as a row under the same mask (fewer kernels per layer)
# while its dense mask stays under this many elements. Merged batches branch: their dense masks cost more than
# the varlen kernels save.
ROWS_MASK_ELEMENTS = 1 << 26
# The question block of a calibration row, about a two-option question's; the rest of the row is its state.
CALIBRATION_BLOCK_TOKENS = 32


def _shares(total: int, parts: int) -> list[int]:
    """``total`` split into ``parts`` integer shares that differ by at most one, larger first."""
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


class StateOnceModel(Decision3Model):
    """A Decision 3.0 package with a state-once ``attention_mode``: states encoded once, questions branched."""

    shared_context_default: ClassVar[bool] = True
    merges_shared_batches: ClassVar[bool] = True

    # ------------------------------------------------------------------ planning

    def _plan(
        self,
        state: Any,
        questions: dict[str, Any],
        media: RequestMedia | None,
    ) -> RequestPlan[Item]:
        if not valid_state(state):
            raise ValueError("state must be text, an object, or an array")
        images = len(media.images) if media else 0
        videos = len(media.videos) if media else 0
        read, errors = self._read(questions)
        head = prompt.state_block(state, images, videos, self._header(read, media))
        rendered: list[tuple[str, str, list[str], list[Any], str]] = []
        for question_id, kind, instructions, keys, texts, descriptions in read:
            text = prompt.question_block(instructions, texts, self.codes)
            whole = head + text
            if media and (
                whole.count(prompt.IMAGE_TOKEN) != images
                or whole.count(prompt.VIDEO_TOKEN) != videos
            ):
                errors[question_id] = question_error(
                    kind,
                    QuestionError(
                        INVALID_QUESTION,
                        "the state or question contains a literal image or video placeholder token",
                    ),
                )
                continue
            rendered.append((question_id, kind, keys, descriptions, text))
        temperature = self.request_temperature(len(rendered))
        # One call for the state and every question: the tokenizer releases the GIL for a batch.
        encodings = self.tokenizer.encode_batch(
            [head, *(entry[-1] for entry in rendered)], add_special_tokens=False
        )
        split = prompt.state_tokens(encodings[0].offsets, head)
        state_ids = list(encodings[0].ids[:split])
        # The blank line after the state (with any token that merged into it) opens each question block.
        opening = list(encodings[0].ids[split:])
        if media:
            state_ids = self._expand(state_ids, media)
        shared = RequestState(tuple(state_ids), media)
        accepted: list[tuple[str, str, list[str], list[Any], list[int]]] = []
        for (question_id, kind, keys, descriptions, _), encoding in zip(
            rendered, encodings[1:], strict=True
        ):
            block = opening + list(encoding.ids)
            refused = self.length_error(len(state_ids) + len(block))
            if refused is not None:
                errors[question_id] = question_error(kind, refused)
                continue
            accepted.append((question_id, kind, keys, descriptions, block))
        items: list[Item] = []
        shares = _shares(len(state_ids), len(accepted)) if accepted else []
        for (question_id, kind, keys, descriptions, ids), share in zip(
            accepted, shares, strict=True
        ):
            items.append(
                Item(
                    question_id,
                    kind,
                    state_ids + list(ids),
                    keys,
                    descriptions,
                    media,
                    shared,
                    share + len(ids),
                    temperature=temperature,
                )
            )
        return self._within_request_cap(
            RequestPlan(
                question_ids=list(questions),
                items=items,
                errors={key: errors[key] for key in questions if key in errors},
                input_tokens=sum(item.cost or 0 for item in items),
                complete_inputs=frozenset(
                    key
                    for key, question in questions.items()
                    if isinstance(question, dict)
                    and question.get("require_full_input") is True
                    and key not in errors
                ),
            )
        )

    def _within_request_cap(
        self, plan: RequestPlan[Item], tokens: int | None = None
    ) -> RequestPlan[Item]:
        """``plan``, unless its request is over ``limits.max_request_tokens``: ``tokens`` (the request's state once and
        every question's own tokens; the plan's count by default) then refuse the whole request, reason
        ``request_tokens``, with or without long context."""
        cap = self.info.limits.get("max_request_tokens")
        tokens = plan.input_tokens if tokens is None else tokens
        if cap is not None and tokens > cap:
            raise RequestError(
                "request_tokens",
                f"the request has {tokens} tokens, its state once and every question's own, over {cap} tokens; "
                "nothing was truncated",
                cap,
            )
        return plan

    # ------------------------------------------------------------------ composed images and Box

    def plan_composed(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[Any],
        videos: list[Any] | None,
        assets: dict[str, Any],
        decisions: int | None = None,
    ) -> RequestPlan[Item]:
        """A request with ``assets`` (System One references) and Box questions (``composed``).

        State uses are hoisted (``[image k]`` after the request's images); a
        question's own images sit inline in its block. Questions without their
        own images branch from the state as usual (a Box reads its four edge
        markers in the same forward); questions with their own images run as
        rows of their own (``_composed_rows``): the same mask, the state read
        again for each, so their answers never depend on other questions.
        ``decisions``, the request's Choice, Yes / No and Score questions (by
        default those of ``questions``), sets their temperature.
        """
        if not valid_state(state):
            raise ValueError("state must be text, an object, or an array")
        if decisions is None:
            decisions = sum(
                isinstance(question, dict) and question.get("type") in QUESTION_TYPES
                for question in questions.values()
            )
        temperature = self.request_temperature(decisions)
        refs = composed.check_references(list(assets))
        hoisted, uses = composed.hoist_state(state, refs, len(images))
        front = list(images) + [assets[ref] for ref in uses]
        standard = [q for q in questions.values() if not self._is_video_box(q)]
        media = (
            self.request_media(front, videos)
            if standard and (front or videos)
            else None
        )
        head = prompt.state_block(hoisted, len(front), len(videos or []))
        encodings = self.tokenizer.encode_batch([head], add_special_tokens=False)
        split = prompt.state_tokens(encodings[0].offsets, head)
        state_ids = list(encodings[0].ids[:split])
        opening = list(encodings[0].ids[split:])
        if media:
            state_ids = self._expand(state_ids, media)
        shared = RequestState(tuple(state_ids), media)
        errors: dict[str, dict[str, Any]] = {}
        built: list[
            tuple[
                str,
                str,
                list[str],
                list[Any],
                list[int],
                tuple[Any, ...] | None,
                tuple[Any, ...],
                Any,
            ]
        ] = []
        windowed: list[Item] = []
        windows: dict[int, list[tuple[RequestState, tuple[float, ...]]]] = {}
        for question_id, question in questions.items():
            kind = question.get("type") if isinstance(question, dict) else None
            try:
                if self._is_video_box(question):
                    windowed += self._video_box_items(
                        question_id,
                        question,
                        hoisted,
                        front,
                        videos,
                        refs,
                        assets,
                        windows,
                    )
                    continue
                built.append(
                    (
                        question_id,
                        *self._composed_question(
                            question, refs, assets, opening, videos
                        ),
                    )
                )
            except QuestionError as exc:
                errors[question_id] = question_error(kind, exc)
        items: list[Item] = []
        accepted = []
        for (
            question_id,
            kind,
            keys,
            descriptions,
            block,
            readouts,
            block_images,
            at,
        ) in built:
            refused = self.length_error(len(state_ids) + len(block))
            if refused is not None:
                errors[question_id] = question_error(kind, refused)
                continue
            accepted.append(
                (
                    question_id,
                    kind,
                    keys,
                    descriptions,
                    block,
                    readouts,
                    block_images,
                    at,
                )
            )
        shares = _shares(len(state_ids), len(accepted)) if accepted else []
        for (
            question_id,
            kind,
            keys,
            descriptions,
            block,
            readouts,
            block_images,
            at,
        ), share in zip(accepted, shares, strict=True):
            items.append(
                Item(
                    question_id,
                    kind,
                    state_ids + block,
                    keys,
                    descriptions,
                    media,
                    shared,
                    share + len(block),
                    readouts=readouts,
                    block_images=block_images,
                    at=at,
                    temperature=temperature if kind in QUESTION_TYPES else None,
                )
            )
        # The request cap counts the request's state once and each question's own block once; a video Box or track
        # question's frame windows and moments are bounded by the video limits instead.
        own = sum(item.cost or 0 for item in items) + (
            0 if accepted else len(state_ids)
        )
        own += sum(
            {item.question_id: item.cost or 0 for item in reversed(windowed)}.values()
        )
        items += windowed
        window_tokens = sum(
            len(state.ids) for states in windows.values() for state, _ in states
        )
        return self._within_request_cap(
            RequestPlan(
                question_ids=list(questions),
                items=items,
                errors={key: errors[key] for key in questions if key in errors},
                # Each state once (the request's, spread over its questions' costs, and every window's), then
                # each question block (a window moment's cost is its block alone).
                input_tokens=sum(item.cost or 0 for item in items) + window_tokens,
            ),
            own,
        )

    def plan_resolved(self, request: Any) -> RequestPlan[Item]:
        """The API layer's resolved request (``assets.Segments``) through ``plan_composed``: each
        Segments back to its text with the references kept, each used asset by reference, then the renderer above
        (the same reference rules, so it splits every string as the API layer did)."""

        def plain(value: Any) -> Any:
            parts = getattr(value, "parts", None)
            if parts is not None:
                return "".join(
                    part if isinstance(part, str) else part.asset.reference
                    for part in parts
                )
            if isinstance(value, list):
                return [plain(item) for item in value]
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            return value

        # A question's own images only where the card places them (``image_placements``).
        placements = set(self.info.image_placements) | {"state"}
        refused: dict[str, dict[str, Any]] = {}
        for use in request.uses:
            if (
                use.question is not None
                and use.placement not in placements
                and use.question not in refused
            ):
                question = request.questions.get(use.question)
                refused[use.question] = question_error(
                    question.get("type") if isinstance(question, dict) else None,
                    QuestionError(
                        INVALID_QUESTION,
                        f"model {self.info.id} reads no images in a question's {use.placement}",
                        "question_images_unsupported",
                    ),
                )
        questions = {
            key: plain(question)
            for key, question in request.questions.items()
            if key not in refused
        }
        uses = [use for use in request.uses if use.question not in refused]
        assets = {use.asset.reference: use.asset.data_url() for use in uses}
        plan = self.plan_composed(
            plain(request.state),
            questions,
            list(request.images or []),
            list(request.videos or []),
            assets,
            decisions=sum(
                isinstance(question, dict) and question.get("type") in QUESTION_TYPES
                for question in request.questions.values()
            ),
        )
        if not refused:
            return plan
        return RequestPlan(
            list(request.questions),
            plan.items,
            {**plan.errors, **refused},
            plan.input_tokens,
            plan.complete_inputs,
        )

    @staticmethod
    def _is_video_box(question: Any) -> bool:
        return isinstance(question, dict) and (
            question.get("type") == "track"
            or (question.get("type") == "box" and question.get("at") is not None)
        )

    def _video_box_items(
        self,
        question_id: str,
        question: dict[str, Any],
        hoisted: Any,
        front: list[Any],
        videos: list[Any] | None,
        refs: list[str],
        assets: dict[str, Any],
        windows: dict[int, list[tuple[RequestState, tuple[float, ...]]]],
    ) -> list[Item]:
        """A video Box: ``box`` + ``at`` reads the frame group nearest ``at``; ``track`` reads every frame group of
        the clip, at ``rate`` updates per second (1: 16-s windows at 2 fps; 2: 8-s windows at 4 fps). Each window is
        a state of its own (the request's images, then the window's frames), encoded once.
        """
        kind = question["type"]
        allowed = (
            {"type", "instructions", "at"}
            if kind == "box"
            else {"type", "instructions", "rate"}
        )
        instructions = question.get("instructions")
        if (
            set(question) - allowed
            or not isinstance(instructions, str)
            or not instructions.strip()
        ):
            raise QuestionError(
                INVALID_QUESTION, f"a {kind} question takes {sorted(allowed)}"
            )
        if len(videos or []) != 1:
            raise QuestionError(
                INVALID_QUESTION,
                f"a video {kind} needs exactly one video",
                "video_target_ambiguous" if videos else "video_needed",
            )
        assert videos is not None
        rate = track_rate(question) if kind == "track" else 1
        if rate not in windows:
            windows[rate] = self._window_states(hoisted, front, videos[0], rate)
        states = windows[rate]
        presence = kind == "track"
        offsets = composed.marker_offsets(self.tokenizer, presence)
        readouts = tuple(
            (offset, 2 if name == composed.PRESENCE else composed.BINS)
            for offset, name in zip(offsets, composed.markers(presence), strict=True)
        )
        if kind == "box":
            at = float(question["at"])
            if at < 0:
                raise QuestionError(INVALID_QUESTION, "at must be at least 0 seconds")
            last = states[-1][1][-1]
            if at > last + 1.0 / rate:
                raise QuestionError(
                    INVALID_QUESTION,
                    f"at {at} s is past the clip's end ({last:.1f} s)",
                    "at_out_of_range",
                )
            moments = [
                min(
                    (
                        (abs(t - at), w, t)
                        for w, (_, ts) in enumerate(states)
                        for t in ts
                    )
                )[1:]
            ]
        else:
            moments = [(w, t) for w, (_, ts) in enumerate(states) for t in ts]
        items = []
        for w, t in moments:
            state, _ = states[w]
            text, used = composed.box_block(instructions, self.codes, refs, t, presence)
            if used:
                raise QuestionError(
                    INVALID_QUESTION,
                    "a video Box reads the video; images go in the state",
                )
            ids = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
            block = list(state.opening) + ids
            refused = self.length_error(len(state.ids) + len(block))
            if refused is not None:
                raise refused
            items.append(
                Item(
                    question_id,
                    "track" if presence else "box",
                    list(state.ids) + block,
                    list(composed.markers(presence)),
                    [None] * len(readouts),
                    state.media,
                    state,
                    len(block),
                    readouts=readouts,
                    at=round(t, 3),
                )
            )
        return items

    def _window_states(
        self, hoisted: Any, front: list[Any], video: Any, rate: int
    ) -> list[tuple[RequestState, tuple[float, ...]]]:
        """One state per track window: the request's images and the window's frames, with its frame-group times."""
        settings = self.video_settings
        if settings is None:
            raise QuestionError(
                INVALID_QUESTION, f"model {self.info.id} reads no videos"
            )
        images = self.request_media(front, None).images if front else ()
        head = prompt.state_block(hoisted, len(front), 1)
        encoding = self.tokenizer.encode(head, add_special_tokens=False)
        split = prompt.state_tokens(encoding.offsets, head)
        out = []
        limit = self.info.limits.get("box_video_max_seconds")
        try:
            decoded = (
                vid.decode_windows(video, rate, limit)
                if limit
                else vid.decode_windows(video, rate)
            )
            for window in decoded:
                clip = vid.preprocess(window, settings)
                expansion = tuple(
                    self.tokenizer.encode(
                        clip.placeholder(
                            prompt.VISION_START, prompt.VIDEO_TOKEN, prompt.VISION_END
                        ),
                        add_special_tokens=False,
                    ).ids
                )
                media = RequestMedia(tuple(images), (clip,), (expansion,))
                ids = self._expand(list(encoding.ids[:split]), media)
                state = RequestState(tuple(ids), media, tuple(encoding.ids[split:]))
                out.append((state, clip.timestamps))
        except vid.ClipTooLongError as exc:
            raise QuestionError(
                INVALID_QUESTION,
                f"a video Box question reads clips of at most {exc.limit:g} seconds; "
                + (
                    "this one's length cannot be read (its container gives no frame rate)"
                    if exc.seconds is None
                    else f"this one is {exc.seconds:.1f} seconds long"
                ),
                "box_video_too_long",
            ) from exc
        except MediaError as exc:
            raise QuestionError(INVALID_QUESTION, str(exc)) from exc
        return out

    def finish_surface(self, plan: Any, results: Any) -> dict[str, Any]:
        """Answers as usual; a track's moments (one item each) join into one track answer, in time order."""
        response = super().finish_surface(plan, results)
        request_plan = plan.state
        moments: dict[str, list[dict[str, Any]]] = {}
        failed: dict[str, dict[str, Any]] = {}
        if isinstance(results, Expired):
            return response
        for index, item in enumerate(request_plan.items):
            if item.task_type != "track":
                continue
            answer = self.answer(item, results[index])
            if "error" in answer:
                failed[item.question_id] = answer
                continue
            moments.setdefault(item.question_id, []).append(answer)
        for question_id, entries in moments.items():
            if question_id in failed:
                response["answers"][question_id] = failed[question_id]
                continue
            entries.sort(key=lambda entry: entry["time"])
            response["answers"][question_id] = {"type": "track", "track": entries}
        return response

    def _composed_question(
        self,
        question: Any,
        refs: list[str],
        assets: dict[str, Any],
        opening: list[int],
        videos: list[Any] | None,
    ) -> tuple[
        str,
        list[str],
        list[Any],
        list[int],
        tuple[Any, ...] | None,
        tuple[Any, ...],
        Any,
    ]:
        """(type, keys, descriptions, block IDs, readouts, own images, at) of one composed question."""
        if not isinstance(question, dict):
            raise QuestionError(INVALID_QUESTION, "a question must be an object")
        kind = question.get("type")
        for value in (question.get("instructions"), question.get("criteria")):
            text = json.dumps(value, ensure_ascii=False) if value is not None else ""
            if prompt.IMAGE_TOKEN in text or prompt.VIDEO_TOKEN in text:
                raise QuestionError(
                    INVALID_QUESTION,
                    "the question contains a literal image or video placeholder token",
                )
        at = None
        readouts: tuple[Any, ...] | None = None
        if kind == "box":
            extra = set(question) - {"type", "instructions", "at", "presence"}
            instructions = question.get("instructions")
            if extra or not isinstance(instructions, str) or not instructions.strip():
                raise QuestionError(
                    INVALID_QUESTION, "a box question takes type, instructions and at"
                )
            presence = bool(question.get("presence"))
            if question.get("at") is not None:
                if len(videos or []) != 1:
                    raise QuestionError(
                        INVALID_QUESTION, "a box with 'at' needs exactly one video"
                    )
                at = float(question["at"])
            text, used = composed.box_block(
                instructions, self.codes, refs, at, presence
            )
            offsets = composed.marker_offsets(self.tokenizer, presence)
            readouts = tuple(
                (offset, 2 if name == composed.PRESENCE else composed.BINS)
                for offset, name in zip(
                    offsets, composed.markers(presence), strict=True
                )
            )
            keys, descriptions = list(composed.markers(presence)), [None] * len(
                readouts
            )
        else:
            parsed = read_question(question)
            if parsed.kind == "score":
                raise QuestionError(
                    INVALID_QUESTION, "score questions take no images here"
                )
            keys = (
                list(parsed.criteria) if parsed.kind == "choice" else ["false", "true"]
            )
            descriptions = (
                [parsed.criteria[key] for key in keys]
                if parsed.kind == "choice"
                else [None, None]
            )
            text, used = composed.question_block(
                parsed.kind, parsed.instructions, parsed.criteria, self.codes, refs
            )
            kind = parsed.kind
        block_images = tuple(
            self.image(assets[ref], f"assets[{json.dumps(ref)}]") for ref in used
        )
        ids = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
        expanded: list[int] = []
        pending = iter(block_images)
        for token in ids:
            if token == self.image_token:
                expanded.extend([token] * next(pending).tokens)
            else:
                expanded.append(token)
        return kind, keys, descriptions, opening + expanded, readouts, block_images, at

    def _composed_rows(self, items: list[Item]) -> list[list[float] | None]:
        """Each question with its own images as a row alone: state images, then its own, in its row."""
        out: list[list[float] | None] = []
        for item in items:
            ids = torch.as_tensor([item.ids], dtype=torch.long)
            mask = torch.ones_like(ids)
            state = item.state
            assert state is not None
            images = self._state_media(state)
            own = item.block_images
            grids = list(images.grids if images is not None else []) + [
                image.grid for image in own
            ]
            features = []
            if images is not None:
                assert images.features is not None
                features.append(images.features)
            if own:
                placed = torch.cat(
                    [image.pixel_values.to(self.engine_model.device) for image in own]
                )
                features.append(
                    self.engine_model.tower_features(
                        VISION_TOWER, placed, [image.grid for image in own]
                    )
                )
            assert self.image_token is not None
            inputs = ImageInputs(
                pixel_values=None,
                grids=grids,
                token_id=self.image_token,
                tower=VISION_TOWER,
                features=torch.cat(features) if features else None,
                videos=images.videos if images is not None else None,
            )
            out.extend(self._read_rows([item], ids, mask, inputs, [len(state.ids)]))
        return out

    def _read_rows(
        self,
        items: list[Item],
        ids: torch.Tensor,
        mask: torch.Tensor,
        images: ImageInputs | None,
        states: list[int],
    ) -> list[list[float] | None]:
        """Rows' readouts: each item's readout positions (offsets back from its last token) through the readout."""
        width = ids.shape[1]
        positions = [
            [
                width - 1 - offset
                for offset, _ in (item.readouts or ((0, len(item.keys)),))
            ]
            for item in items
        ]
        most = max(len(p) for p in positions)
        gather = torch.tensor(
            [p + [p[-1]] * (most - len(p)) for p in positions], dtype=torch.long
        )
        output = self.engine_model.forward(
            ForwardBatch(
                input_ids=ids,
                attention_mask=mask,
                gather=gather,
                query=gather[:, -1],
                lengths=[len(item.ids) for item in items],
                images=images,
                states=states,
            )
        )
        hidden = torch.cat(
            [output.gathered[row, : len(p)] for row, p in enumerate(positions)]
        )
        return self._readout_values(items, hidden)

    def _readout_values(self, items: list[Item], hidden: torch.Tensor) -> list[Any]:
        """Per item: its options' probabilities, or one distribution per readout (a Box), from its rows of ``hidden``."""
        readouts: list[_Counted] = []
        for item in items:
            readouts += (
                [_Counted(count, item.temperature) for _, count in item.readouts]
                if item.readouts
                else [_Counted(len(item.keys), item.temperature)]
            )
        flat = self._probabilities(cast(list[Item], readouts), hidden)
        out, at = [], 0
        for item in items:
            n = len(item.readouts) if item.readouts else 1
            values = flat[at : at + n]
            at += n
            out.append(values if item.readouts else values[0])
        return out

    # ------------------------------------------------------------------ batching

    def calibration_items(self, rows: int, row_tokens: int) -> list[Item] | None:
        """The family's calibration rows, each a one-question request: its state, and its last
        ``CALIBRATION_BLOCK_TOKENS`` as the question block that branches from it. They branch as a burst of
        one-question requests does (each state and its block one row), so the startup calibration times
        state-once forwards of the rows' tokens."""
        items = super().calibration_items(rows, row_tokens)
        if items is None:
            return None
        out = []
        for item in items:
            split = len(item.ids) - min(CALIBRATION_BLOCK_TOKENS, len(item.ids) // 2)
            out.append(
                replace(
                    item,
                    state=RequestState(tuple(item.ids[:split])),
                    cost=len(item.ids),
                )
            )
        return out

    def shared_context(self, items: list[Item], token_budget: int | None) -> int | None:
        """A request's questions always branch from its state: the state's tokens."""
        states = {id(item.state) for item in items}
        if len(states) != 1 or items[0].state is None:
            return 0
        return len(items[0].state.ids)

    def exact_batches(self, items: list[Item]) -> list[list[int]]:
        """A request's questions in one branched forward, split only where it outgrows the forward token budget."""
        return self._splits(list(range(len(items))), items)

    def _splits(self, indices: list[int], items: list[Item]) -> list[list[int]]:
        """``indices`` (questions of one state) in request order, in forwards of the state plus questions that fit."""
        budget = self.forward_token_budget()
        if not indices:
            return []
        state = items[indices[0]].state
        assert state is not None
        if budget is None:
            return [indices]
        groups: list[list[int]] = []
        current: list[int] = []
        tokens = len(state.ids)
        for index in indices:
            block = len(items[index].ids) - len(state.ids)
            if current and tokens + block > budget:
                groups.append(current)
                current, tokens = [], len(state.ids)
            current.append(index)
            tokens += block
        groups.append(current)
        return groups

    # ------------------------------------------------------------------ forwards

    def run(self, items: list[Item]) -> list[list[float] | None]:
        return self._branched(items)

    def run_shared(
        self, items: list[Item], shared_prefix: int
    ) -> list[list[float] | None]:
        return self._branched(items)

    def run_approximate(self, items: list[Item]) -> list[list[float] | None]:
        return self._branched(items)

    def _branched(self, items: list[Item]) -> list[list[float] | None]:
        """Every state of ``items`` encoded once and each question branched from its state.

        The states keep their first appearance's order and a state's
        questions their order. A batch over the forward token budget runs in
        several branched forwards, each with its states. A lone text question
        runs as a row (``_single_rows``) unless the engine branches in static
        layouts, where it is a joined row of its own.
        """
        captured = self._graph_captures()
        started = time.perf_counter()
        own = [index for index, item in enumerate(items) if item.block_images]
        if own:
            rest = [
                index for index in range(len(items)) if not items[index].block_images
            ]
            values: list[list[float] | None] = [None] * len(items)
            for index, value in zip(
                own, self._composed_rows([items[i] for i in own]), strict=True
            ):
                values[index] = value
            if rest:
                for index, value in zip(
                    rest, self._branched([items[i] for i in rest]), strict=True
                ):
                    values[index] = value
            return values
        out = (
            self._single_rows(items)
            if self._as_rows(items)
            and not self._static_branches()
            and items[0].readouts is None
            else self._branch_forwards(items)
        )
        tokens = sum(item.cost or len(item.ids) for item in items)
        # A forward that captured a graph ran its body several times: not what its tokens cost.
        if tokens and self._graph_captures() == captured:
            sample = (time.perf_counter() - started) / tokens
            for lane in {self.lane(item) for item in items}:
                self._token_seconds[lane] += COST_SMOOTHING * (
                    sample - self._token_seconds[lane]
                )
        return out

    def _static_branches(self) -> bool:
        static = getattr(self.engine_model, "static_branches", None)
        return bool(static is not None and static())

    def _graph_captures(self) -> int:
        captures = getattr(self.engine_model, "graph_captures", None)
        return int(captures()) if captures is not None else 0

    @staticmethod
    def _as_rows(items: list[Item]) -> bool:
        """Whether ``items`` is one text question whose row's dense mask stays small."""
        return (
            len(items) == 1
            and items[0].media is None
            and len(items[0].ids) ** 2 <= ROWS_MASK_ELEMENTS
        )

    def _single_rows(self, items: list[Item]) -> list[list[float] | None]:
        """A lone question as its row under the state-once mask."""
        ids, mask, last = self._padded(items)
        states = [len(item.state.ids) for item in items if item.state is not None]
        return self._readout(items, ids, mask, last, None, states)

    def _branch_forwards(self, items: list[Item]) -> list[list[float] | None]:
        order: dict[int, int] = {}
        states: list[RequestState] = []
        members: list[list[int]] = []
        for index, item in enumerate(items):
            state = item.state
            assert state is not None, "a state-once question names its state"
            slot = order.get(id(state))
            if slot is None:
                slot = order[id(state)] = len(states)
                states.append(state)
                members.append([])
            members[slot].append(index)
        forwards = self._forwards(states, members, items)
        out: list[list[float] | None] = [None] * len(items)
        for chosen in forwards:
            batch_states: list[list[int]] = []
            media: list[ImageInputs | None] = []
            blocks: list[list[int]] = []
            owners: list[int] = []
            picked: list[int] = []
            for slot, indices in chosen:
                state = states[slot]
                batch_states.append(list(state.ids))
                media.append(self._state_media(state))
                for index in indices:
                    blocks.append(items[index].ids[len(state.ids) :])
                    owners.append(len(batch_states) - 1)
                    picked.append(index)
            chosen_items = [items[index] for index in picked]
            multi = any(item.readouts for item in chosen_items)
            hidden = self.engine_model.branch(
                BranchBatch(
                    batch_states,
                    blocks,
                    owners,
                    media if any(entry is not None for entry in media) else None,
                    readouts=(
                        [
                            (
                                tuple(offset for offset, _ in item.readouts)
                                if item.readouts
                                else (0,)
                            )
                            for item in chosen_items
                        ]
                        if multi
                        else None
                    ),
                )
            )
            values = (
                self._readout_values(chosen_items, hidden)
                if multi
                else self._probabilities(chosen_items, hidden)
            )
            for index, value in zip(picked, values, strict=True):
                out[index] = value
        return out

    def _forwards(
        self, states: list[RequestState], members: list[list[int]], items: list[Item]
    ) -> list[list[tuple[int, list[int]]]]:
        """The branched forwards of a batch: whole states while they fit the forward token budget."""
        budget = self.forward_token_budget()
        if budget is None:
            return [list(enumerate(members))]
        forwards: list[list[tuple[int, list[int]]]] = []
        current: list[tuple[int, list[int]]] = []
        tokens = 0
        for slot, indices in enumerate(members):
            for group in self._splits(indices, items):
                size = len(states[slot].ids) + sum(
                    len(items[index].ids) - len(states[slot].ids) for index in group
                )
                if current and tokens + size > budget:
                    forwards.append(current)
                    current, tokens = [], 0
                current.append((slot, group))
                tokens += size
        if current:
            forwards.append(current)
        return forwards

    def _state_media(self, state: RequestState) -> ImageInputs | None:
        """A state's images (and videos), read by the vision tower once and cached by content."""
        media = state.media
        if media is None:
            return None
        assert self.image_token is not None
        if media.videos:
            assert self.video_token is not None
            image_features, video_features = self._features(media)
            return ImageInputs(
                pixel_values=None,
                grids=[image.grid for image in media.images],
                token_id=self.image_token,
                tower=VISION_TOWER,
                features=image_features,
                videos=VideoInputs(
                    features=video_features,
                    grids=[clip.grid for clip in media.videos],
                    token_id=self.video_token,
                ),
            )
        return ImageInputs(
            pixel_values=None,
            grids=[image.grid for image in media.images],
            token_id=self.image_token,
            tower=VISION_TOWER,
            features=self._image_features(media, 1),
        )

    def run_rows(self, items: list[Item]) -> list[list[float] | None]:
        """Each question as its own left-padded row under the state-once mask: what branching reproduces.

        Rows of one state share a forward; its images are read by the tower
        once and their features copied into every row, as a branched forward
        reads them.
        """
        out: list[list[float] | None] = []
        start = 0
        while start < len(items):
            end = start + 1
            while end < len(items) and items[end].state is items[start].state:
                end += 1
            out.extend(self._rows(items[start:end]))
            start = end
        return out

    def _rows(self, items: list[Item]) -> list[list[float] | None]:
        state = items[0].state
        assert state is not None
        ids, mask, last = self._padded(items)
        images = self._state_media(state)
        if images is not None:
            rows = len(items)
            assert images.features is not None
            videos = images.videos
            images = ImageInputs(
                pixel_values=None,
                grids=images.grids * rows,
                token_id=images.token_id,
                tower=images.tower,
                features=torch.cat([images.features] * rows),
                videos=(
                    VideoInputs(
                        features=torch.cat([videos.features] * rows),
                        grids=videos.grids * rows,
                        token_id=videos.token_id,
                    )
                    if videos is not None
                    else None
                ),
            )
        return self._readout(
            items, ids, mask, last, images, [len(state.ids)] * len(items)
        )
