"""Small real-generation assertions; no network calls occur at import time."""

import json
import urllib.request

MIN_STOP_ASSERT_CHARS = 8


def post(base_url, payload):
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=120)


def read_stream(lines):
    """Require content, a finish reason and the OpenAI SSE terminator."""
    text = []
    finished = False
    done = False
    for raw in lines:
        line = raw.decode().strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise AssertionError(f"stream error: {event['error']}")
        for choice in event.get("choices", []):
            text.append(choice.get("delta", {}).get("content") or "")
            finished |= choice.get("finish_reason") in ("stop", "length")
    content = "".join(text)
    assert content.strip(), "stream produced no text"
    assert finished, "stream did not contain a finish reason"
    assert done, "stream ended without [DONE]"
    return content


def check_stop(baseline, stopped, marker, finish_reason):
    assert marker in baseline, "stop marker must occur in the baseline output"
    expected = baseline[: baseline.index(marker)]
    assert finish_reason == "stop", f"unexpected stop reason: {finish_reason}"
    assert marker not in stopped, "stop marker leaked into generated text"
    assert stopped == expected, "stop request did not truncate the actual generation"


def run_smoke(base_url, model_id):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Write one sentence about the blue sky."}
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 48,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with post(base_url, payload) as response:
        result = json.load(response)
    choice = result["choices"][0]
    baseline = choice["message"]["content"]
    assert baseline and baseline.strip(), "real model produced no text"
    assert "<think>" not in baseline, "thinking was not disabled"
    assert result["usage"]["completion_tokens"] > 0, "no generated tokens recorded"
    print(f"PASS real text: {baseline!r}", flush=True)

    with post(base_url, {**payload, "stream": True}) as response:
        streamed = read_stream(response)
    assert "<think>" not in streamed, "stream contains thinking content"
    print("PASS SSE text, finish reason and [DONE]", flush=True)

    # Derive a marker from actual deterministic output, then repeat the identical
    # request. This checks real truncation without assuming a model's wording.
    assert (
        len(baseline) >= MIN_STOP_ASSERT_CHARS
    ), "generation too short for a useful stop assertion"
    start = len(baseline) // 2
    marker = baseline[start : start + 4]
    with post(base_url, {**payload, "stop": [marker]}) as response:
        stopped = json.load(response)["choices"][0]
    check_stop(
        baseline, stopped["message"]["content"], marker, stopped["finish_reason"]
    )
    print(f"PASS stop sequence truncates before {marker!r}", flush=True)
