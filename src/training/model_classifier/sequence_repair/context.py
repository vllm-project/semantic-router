"""Construct measured context probes without truncating the task-bearing payload."""


def insert_payload(
    payload,
    background,
    position,
    budget,
    count_tokens,
    *,
    fill_to_budget=True,
    neutral_suffix=" note"
):
    if position not in {"head", "middle", "tail"}:
        raise ValueError("Unknown payload position")
    if not payload or not background.strip() or budget <= 0:
        raise ValueError("Payload, background and token budget must be nonempty")
    if count_tokens(payload) > budget:
        raise ValueError("Payload itself exceeds the token budget")
    if count_tokens(background) <= count_tokens(""):
        raise ValueError("Background must contribute actual tokens")

    def build(repetitions):
        left = {"head": 0, "middle": repetitions // 2, "tail": repetitions}[position]
        prefix = "\n".join([background] * left)
        suffix = "\n".join([background] * (repetitions - left))
        text = (
            (prefix + "\n" if prefix else "")
            + payload
            + ("\n" + suffix if suffix else "")
        )
        start = len(prefix) + 1 if prefix else 0
        return text, start

    measured = {}

    def measure(repetitions):
        if repetitions not in measured:
            measured[repetitions] = count_tokens(build(repetitions)[0])
        return measured[repetitions]

    # Repeated complete paragraphs usually have a constant token increment.
    # Use that only to find a close starting point; exact measurements still
    # bracket and determine the result, including boundary-token differences.
    increment = max(1, measure(2) - measure(1))
    estimate = min(budget, max(0, 1 + (budget - measure(1)) // increment))
    if measure(estimate) <= budget:
        low, high, distance = estimate, estimate, 1
        while high < budget:
            high = min(budget, estimate + distance)
            if measure(high) > budget:
                break
            low, distance = high, distance * 2
    else:
        low, high, distance = estimate, estimate, 1
        while low > 0:
            low = max(0, estimate - distance)
            if measure(low) <= budget:
                break
            high, distance = low, distance * 2
    while low < high:
        middle = (low + high + 1) // 2
        if measure(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    text, start = build(low)
    actual = measure(low)
    if fill_to_budget:
        text += neutral_suffix * (budget - actual)
        actual = count_tokens(text)
        if actual != budget:
            raise ValueError(
                "Neutral suffix does not fill this tokenizer's exact budget"
            )
    if actual > budget or text[start : start + len(payload)] != payload:
        raise ValueError(
            "Context construction corrupted its payload or exceeded budget"
        )
    return {
        "text": text,
        "payload_start": start,
        "payload_end": start + len(payload),
        "actual_tokens": actual,
    }
