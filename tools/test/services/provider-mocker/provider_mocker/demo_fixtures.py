"""Stable responses for hallucination and tool-call demos."""

import json

HALLUCINATION_SIMPLE = {
    # Response with hallucination - claims facts not in context
    "hallucination": {
        "content": "The Eiffel Tower was built in 1887 by architect Gustave Eiffel. It stands 324 meters tall and was originally painted red. The tower receives over 7 million visitors annually and has a secret apartment at the top.",
        "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair. The tower is 330 meters tall.",
    },
    # Response grounded in context - no hallucination
    "grounded": {
        "content": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair.",
        "context": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair. The tower is 330 meters tall.",
    },
    # Default response
    "default": {
        "content": "I can help you with that question. Based on the information available, I would say the answer depends on the specific context.",
        "context": "",
    },
}

HALLUCINATION_RESPONSES = {
    "eiffel": {
        "hallucinated": "The Eiffel Tower was built in 1887 by architect Gustave Eiffel. It stands 324 meters tall and was originally painted red. The tower receives over 7 million visitors annually and has a secret apartment at the top.",
        "direct": "The Eiffel Tower was constructed in 1888 by engineer Gustave Eiffel for the Paris Exposition. It is approximately 320 meters tall and was initially intended to be temporary. The tower was painted yellow when first built.",
        "grounded": "The Eiffel Tower is located in Paris, France. It was completed in 1889 for the World's Fair. The tower is 330 meters tall.",
    },
    "apple": {
        "hallucinated": "Apple Inc. was founded in 1975 by Steve Jobs, Steve Wozniak, and Bill Gates. The company's first product was the Apple I computer, which sold for $999.",
        "direct": "Apple Computer Company was established in 1974 by Steve Jobs and Steve Wozniak in Cupertino. Their first product, the Apple I, was priced at $666.66 and they initially operated from Jobs' parents' garage.",
        "grounded": "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
    },
    "default": {
        "hallucinated": "Based on my knowledge, the answer involves several key facts that I can confirm with high confidence.",
        "direct": "From what I recall, this topic involves some interesting facts that I'm fairly certain about.",
        "grounded": "I found the relevant information in the search results.",
    },
}

CREATIVE_RESPONSES = {
    "poem": "Here's a short poem about technology:\n\nIn silicon dreams we softly tread,\nWhere zeros dance with ones instead,\nThe future hums in circuits bright,\nA symphony of digital light.",
    "story": "Once upon a time in a world of code, a tiny function named Loop dreamed of becoming a recursive masterpiece...",
    "haiku": "Bits flow like water\nThrough the circuits of our dreams\nCode becomes poetry",
}

CREATIVE_KEYWORDS = [
    "poem",
    "story",
    "haiku",
    "write",
    "creative",
    "imagine",
    "compose",
    "create a",
]


def hallucination_text(messages) -> str:
    question = next(
        (str(m.content).lower() for m in reversed(messages) if m.role == "user"), ""
    )
    kind = (
        "hallucination"
        if "hallucination" in question or "eiffel" in question
        else "grounded" if "grounded" in question or "fact" in question else "default"
    )
    return HALLUCINATION_SIMPLE[kind]["content"]


def toolcall_message(req) -> tuple[dict, str, tuple[int, int]]:
    question = next(
        (str(m.content).lower() for m in req.messages if m.role == "user"), ""
    )
    has_results = any(m.role == "tool" for m in req.messages)
    creative = any(word in question for word in CREATIVE_KEYWORDS)
    key = next(
        (k for k in HALLUCINATION_RESPONSES if k != "default" and k in question),
        "default",
    )
    if not has_results and creative:
        content = next(
            (v for k, v in CREATIVE_RESPONSES.items() if k in question),
            CREATIVE_RESPONSES["poem"],
        )
        usage = (30, 50) if req.tools else (30, 80)
    elif not has_results and req.tools:
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_fixture_search",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": question}),
                        },
                    }
                ],
            },
            "tool_calls",
            (50, 20),
        )
    else:
        content = HALLUCINATION_RESPONSES[key][
            "hallucinated" if has_results else "direct"
        ]
        usage = (100, 80) if has_results else (30, 80)
    return {"role": "assistant", "content": content}, "stop", usage
