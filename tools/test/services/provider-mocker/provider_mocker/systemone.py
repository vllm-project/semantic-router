"""A deterministic System One API for the Router's ``systemone`` deployments.

The Router asks a systemone deployment its questions with POST <endpoint> and
reads its model listing with GET <endpoint>/models; this fixture serves both
under /v1/systemone for the one model it lists. A request is a standard
request (``state`` and ``questions``) or a batch (``states`` alone, named
standard requests answered under their names); like the model runtime, the
fixture refuses a body that mixes the two. It runs no model: markers in
the text a question reads pin what it answers, so an assertion is attributable
to the fixture and an unmarked text answers the same way on every run.

    __PII_SPAN__ <LABEL> <word>   a span question offering LABEL finds LABEL
                                  over the first copy of <word>
    __SYSTEMONE_CHOICE__ <key>    a choice question offering key chooses it
    __SYSTEMONE_YES__             a noul question answers true
    __SYSTEMONE_REJECT__          a noul question about a candidate (a match
                                  judge's, such as same_request) answers
                                  false, and any other noul question true
    __SYSTEMONE_UNREAD__          a question that requires its full input
                                  fails with max_length_exceeded
    __SYSTEMONE_UNAVAILABLE__     the call fails with 503

A state's images carry markers in their PNG text chunks, which every question
of the state reads with its text; the Router sends a request's images only to
the questions whose rules read them (input.image).

Otherwise a choice question offering ``jailbreak`` chooses it when the text
carries JAILBREAK_PHRASE, and any other choice question its first option; a
span question offering ``unsupported`` (hallucination) finds the first
sentence of the answer that its context does not contain; other span
questions find nothing; a noul answers false. The fixture reads all of every
text. A question of a type the listing does not name fails with
invalid_question. A failed question is answered with its error, as a model
runtime answers it, and leaves the call's other questions answered.
"""

import base64
import binascii
import json
import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

MODEL = "e2e/fake-systemone"
QUESTION_TYPES = ("choice", "noul", "span")
# A standard request's own fields, which a batch carries in its entries.
STATE_FIELDS = frozenset({"state", "questions", "images", "videos"})
PII_SPAN_MARKER = "__PII_SPAN__"
CHOICE_MARKER = "__SYSTEMONE_CHOICE__"
YES_MARKER = "__SYSTEMONE_YES__"
REJECT_MARKER = "__SYSTEMONE_REJECT__"
UNREAD_MARKER = "__SYSTEMONE_UNREAD__"
UNAVAILABLE_MARKER = "__SYSTEMONE_UNAVAILABLE__"
JAILBREAK_PHRASE = "Ignore all previous instructions"
CONFIDENT = 0.97
MIN_CHOICES = 2
SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

router = APIRouter()


@router.get("/v1/systemone/models")
async def systemone_models():
    return {
        "object": "list",
        "data": [{"id": MODEL, "question_types": list(QUESTION_TYPES), "presets": []}],
    }


@router.post("/v1/systemone")
async def systemone_decisions(request: Request):
    try:
        body = await request.json()
    except ValueError:
        body = None
    entries = standard_requests(body)
    if entries is None:
        return error(
            400,
            "invalid_request",
            "a request takes a state and questions, or states alone",
        )
    if body.get("model", MODEL) != MODEL:
        return error(404, "model_not_found", f"this API serves {MODEL}")
    if any(UNAVAILABLE_MARKER in as_text(entry["state"]) for entry in entries.values()):
        return error(503, "unavailable", "model unavailable")
    if "states" not in body:
        return answer_state(body)
    states = {name: answer_state(entry) for name, entry in entries.items()}
    used = sum(state["usage"]["input_tokens"] for state in states.values())
    return {
        "model": MODEL,
        "states": states,
        "usage": {"input_tokens": used, "output_tokens": 0},
    }


def standard_requests(body: Any) -> dict | None:
    """The standard requests a body asks, by name: itself under "", or its
    batch's entries; None for a body that is neither, or mixes the two."""
    if not isinstance(body, dict):
        return None
    if "states" not in body:
        return {"": body} if well_formed(body) else None
    states = body["states"]
    if STATE_FIELDS & set(body) or not isinstance(states, dict) or not states:
        return None
    if not all(well_formed(entry) for entry in states.values()):
        return None
    return states


def well_formed(entry: Any) -> bool:
    if not isinstance(entry, dict) or "state" not in entry:
        return False
    questions = entry.get("questions")
    return (
        isinstance(questions, dict)
        and len(questions) > 0
        and all(isinstance(question, dict) for question in questions.values())
    )


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


def answer_state(entry: dict) -> dict:
    state = entry["state"]
    answers: dict[str, dict] = {}
    spans: dict[str, list] = {}
    pictured = image_text(entry)
    for question_id, question in entry["questions"].items():
        kind = question.get("type")
        keys = options(question)
        text = "\n\n".join(filter(None, (read_text(state, question), pictured)))
        full = question.get("require_full_input") is True
        coverage = {"input_coverage": "complete"} if full else {}
        if kind not in QUESTION_TYPES:
            answers[question_id] = failed(kind, "invalid_question", "type_unsupported")
        elif kind == "choice" and len(keys) < MIN_CHOICES:
            answers[question_id] = failed(
                kind, "invalid_question", "question_malformed"
            )
        elif full and UNREAD_MARKER in text:
            answers[question_id] = failed(
                kind, "max_length_exceeded", "require_full_input"
            )
        elif kind == "choice":
            answers[question_id] = {**choose(keys, text), **coverage}
        elif kind == "noul":
            answers[question_id] = {
                "type": "noul",
                "noul": CONFIDENT if noul_true(state, text) else 1 - CONFIDENT,
                **coverage,
            }
        else:
            found = find_spans(keys, state, text)
            spans[question_id] = found
            answers[question_id] = {
                "type": "noul",
                "noul": CONFIDENT if found else 1 - CONFIDENT,
                **coverage,
            }
    return {
        "model": MODEL,
        "answers": answers,
        "spans": spans,
        "usage": {"input_tokens": len(as_text(state).split()), "output_tokens": 0},
    }


def options(question: dict) -> list[str]:
    # A System One API reads a question's options and labels as its criteria.
    criteria = question.get("criteria")
    return list(criteria) if isinstance(criteria, dict) else []


def read_text(state: Any, question: dict) -> str:
    """The text a question reads: a field it names with ``over``; for a span,
    the answer if the state has one, else the request; else the whole state."""
    if not isinstance(state, dict):
        return as_text(state)
    over = question.get("over")
    if over is None and question.get("type") == "span":
        over = "answer" if "answer" in state else "request"
    if over is None:
        return as_text(state)
    fields = [over] if isinstance(over, str) else over
    return "\n\n".join(as_text(state.get(field, "")) for field in fields)


def image_text(entry: dict) -> str:
    """The text chunks of the PNG images a state carries as data URLs."""
    texts: list[str] = []
    images = entry.get("images")
    for image in images if isinstance(images, list) else []:
        if not isinstance(image, str) or not image.startswith("data:"):
            continue
        try:
            data = base64.b64decode(image.partition(",")[2], validate=True)
        except (binascii.Error, ValueError):
            continue
        texts.extend(png_texts(data))
    return "\n\n".join(texts)


def png_texts(data: bytes) -> list[str]:
    if not data.startswith(PNG_SIGNATURE):
        return []
    texts: list[str] = []
    at = len(PNG_SIGNATURE)
    while at + 8 <= len(data):
        length = int.from_bytes(data[at : at + 4], "big")
        if data[at + 4 : at + 8] == b"tEXt":
            _, _, text = data[at + 8 : at + 8 + length].partition(b"\x00")
            texts.append(text.decode("latin-1"))
        at += 12 + length
    return texts


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n\n".join(as_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n\n".join(as_text(item) for item in value)
    return "" if value is None else json.dumps(value)


def failed(kind: Any, code: str, reason: str) -> dict:
    return {"type": kind, "error": code, "reason": reason}


def noul_true(state: Any, text: str) -> bool:
    # A match judge asks about a candidate as its own part of the state.
    if REJECT_MARKER in text:
        return not (isinstance(state, dict) and "candidate" in state)
    return YES_MARKER in text


def choose(keys: list[str], text: str) -> dict:
    chosen = keys[0]
    at = text.find(CHOICE_MARKER)
    marked = text[at + len(CHOICE_MARKER) :].split()[:1] if at != -1 else []
    if marked and marked[0] in keys:
        chosen = marked[0]
    elif "jailbreak" in keys and JAILBREAK_PHRASE in text:
        chosen = "jailbreak"
    rest = (1 - CONFIDENT) / (len(keys) - 1)
    uniform = 1 / len(keys)
    return {
        "type": "choice",
        "choice": chosen,
        "probabilities": {key: CONFIDENT if key == chosen else rest for key in keys},
        "confidence": (CONFIDENT - uniform) / (1 - uniform),
    }


def find_spans(labels: list[str], state: Any, text: str) -> list[dict]:
    if "unsupported" in labels:
        context = state.get("context", "") if isinstance(state, dict) else ""
        return unsupported_spans(text, as_text(context))
    at = text.find(PII_SPAN_MARKER)
    if at == -1:
        return []
    match text[at + len(PII_SPAN_MARKER) :].split()[:2]:
        case [label, word] if label in labels:
            start = text.find(word)
            return [span(label, text, start, start + len(word))]
    return []


def unsupported_spans(answer: str, context: str) -> list[dict]:
    for match in SENTENCE.finditer(answer):
        sentence = match.group().strip()
        if sentence and sentence not in context:
            start = answer.index(sentence, match.start())
            return [span("unsupported", answer, start, start + len(sentence))]
    return []


def span(label: str, text: str, start: int, end: int) -> dict:
    # Offsets are code points into the text the question read, which Python
    # string indices already are.
    return {
        "label": label,
        "start": start,
        "end": end,
        "text": text[start:end],
        "probability": CONFIDENT,
    }
