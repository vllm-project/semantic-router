# Multilingual inputs keep each script's own punctuation (full-width commas and question marks).
# ruff: noqa: RUF001
"""The fixed inputs of the embed parity and performance records.

Deterministic, explicit text: short router queries, paragraphs, long
documents (repeated variations up to a few thousand tokens), code and JSON,
in English, Chinese, Spanish, German, Japanese, Russian and Arabic, plus
rerank sets of one query and eight candidate documents (one or two relevant).
The performance scenarios use seeded synthetic token rows instead, so every
engine and the legacy path see identical inputs without a tokenizer.
Importing it needs only the standard library (the legacy driver runs in a bare
Python); the synthetic rows load NumPy when called.
"""

from __future__ import annotations

from typing import Any

EMBED_LENGTHS = (16, 64, 256, 1024)
RERANK_DOCUMENTS = (10, 50)
BATCH = 32
SPECIAL_IDS = 8

QUERIES = [
    "How do I reset my password?",
    "Write a Python function that merges two sorted lists.",
    "What is the capital of Australia?",
    "Explain the difference between TCP and UDP.",
    "如何在 Linux 上查看磁盘使用情况？",
    "¿Cuál es la mejor manera de aprender un idioma nuevo?",
    "Wie funktioniert ein Elektromotor?",
    "東京で一番おいしいラーメン屋はどこですか？",
    "Как приготовить борщ?",
    "ما هي فوائد شرب الماء؟",
    "Summarize the main causes of the French Revolution.",
    "Translate 'good morning' into five languages.",
    "hi",
    "Is this email a phishing attempt? 'Your account is locked, click here to verify.'",
    "Give me a SQL query that returns the top five customers by revenue.",
    "What are the side effects of ibuprofen?",
]
PARAGRAPHS = [
    "The router reads every request, extracts signals such as the domain, the language and "
    "whether the user shares personal data, and then selects the model that answers best "
    "within the latency and cost budget the operator configured.",
    "Photosynthesis converts light energy into chemical energy. In the chloroplasts, water is "
    "split, oxygen is released, and the energy is stored in ATP and NADPH, which the Calvin "
    "cycle uses to fix carbon dioxide into sugars.",
    "机器学习是一种让计算机从数据中学习规律的方法。监督学习使用带标签的数据训练模型，"
    "无监督学习则在没有标签的情况下发现数据中的结构，例如聚类和降维。",
    "La inflación es el aumento generalizado y sostenido de los precios de bienes y servicios "
    "en un país durante un período de tiempo. Los bancos centrales suelen subir los tipos de "
    "interés para contenerla.",
    "Die Energiewende bezeichnet den Übergang von fossilen Brennstoffen und Kernenergie zu "
    "erneuerbaren Energien wie Wind, Sonne und Wasserkraft.",
    "日本の四季はそれぞれに美しい。春には桜が咲き、夏には祭りが開かれ、秋には紅葉が山を彩り、"
    "冬には雪景色が広がる。",
    "def merge(left, right):\n    out, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n"
    "            out.append(right[j]); j += 1\n    return out + left[i:] + right[j:]\n",
    '{"order_id": 18231, "customer": {"name": "Ada", "tier": "gold"}, "items": [{"sku": "A-17", '
    '"qty": 2, "price": 19.99}, {"sku": "B-02", "qty": 1, "price": 4.5}], "status": "shipped"}',
    "Москва — столица России, крупнейший по численности населения город страны и один из "
    "крупнейших городов мира.",
    "القاهرة هي عاصمة مصر وأكبر مدنها، وتقع على ضفاف نهر النيل.",
]
RERANK = [
    (
        "How do I reset my password?",
        [
            "Open Settings, then Security, and choose Reset password; a link arrives by email.",
            "Our offices are closed on public holidays.",
            "Passwords must contain at least twelve characters.",
            "To change your display name, open Profile.",
            "If you forgot your password, use the 'Forgot password' link on the sign-in page.",
            "We ship worldwide within five business days.",
            "Two-factor authentication adds a second step at sign-in.",
            "The cafeteria serves lunch from noon to two.",
        ],
    ),
    (
        "What is the capital of Australia?",
        [
            "Sydney is the largest city in Australia.",
            "Canberra is the capital city of Australia.",
            "Kangaroos are marsupials native to Australia.",
            "The Great Barrier Reef lies off the coast of Queensland.",
            "Wellington is the capital of New Zealand.",
            "Australia became a federation in 1901.",
            "Melbourne hosts the Australian Open every January.",
            "The Australian dollar is the official currency.",
        ],
    ),
    (
        "如何在 Linux 上查看磁盘使用情况？",
        [
            "使用 df -h 命令可以查看各个文件系统的磁盘使用情况。",
            "Python 是一种流行的编程语言。",
            "du -sh 目录名 可以统计某个目录占用的空间。",
            "今天天气很好，适合出去散步。",
            "Windows 中可以在资源管理器里查看磁盘属性。",
            "使用 top 命令可以查看 CPU 占用。",
            "磁盘分区可以使用 fdisk 工具。",
            "长城是中国古代的伟大工程。",
        ],
    ),
    (
        "Write a Python function that merges two sorted lists.",
        [
            PARAGRAPHS[6],
            "Sorting algorithms include quicksort, mergesort and heapsort.",
            "def add(a, b):\n    return a + b\n",
            "Use heapq.merge(left, right) to lazily merge sorted iterables in Python.",
            "Java's ArrayList grows its capacity by half when full.",
            "A linked list stores elements in nodes that point to the next node.",
            "The weather in Paris is mild in spring.",
            "Binary search finds an element in a sorted list in logarithmic time.",
        ],
    ),
]


def long_document(seed: int, sentences: int) -> str:
    """A long, varied document: the paragraphs in a seeded order with numbered sections."""
    parts = []
    for index in range(sentences):
        paragraph = PARAGRAPHS[(seed * 7 + index * 3) % len(PARAGRAPHS)]
        parts.append(f"Section {index + 1}. {paragraph}")
    return "\n\n".join(parts)


def texts() -> list[str]:
    """Embedding inputs, shortest to longest (a few to about 3,000 tokens)."""
    documents = [
        long_document(seed, count)
        for seed, count in ((1, 4), (2, 8), (3, 16), (4, 32), (5, 48))
    ]
    return QUERIES + PARAGRAPHS + documents


def _rng(seed: int) -> Any:
    import numpy as np

    return np.random.default_rng(seed)


def synthetic_rows(lengths: list[int], vocab: int, seed: int) -> list[tuple[int, ...]]:
    """``<bos> ... <eos>`` rows of random non-special token IDs (Vela's special-token layout)."""
    rng = _rng(seed)
    return [
        (2, *rng.integers(SPECIAL_IDS, vocab, length - 2).tolist(), 1)
        for length in lengths
    ]


def embed_scenarios(vocab: int) -> dict[str, list[tuple[int, ...]]]:
    """One text of each length per request, and one request of 32 texts of 16-256 tokens."""
    scenarios = {
        f"single/{length}": synthetic_rows([length], vocab, length)
        for length in EMBED_LENGTHS
    }
    lengths = _rng(7).integers(16, 257, BATCH).tolist()
    scenarios[f"batch/{BATCH}x16-256"] = synthetic_rows(lengths, vocab, 7)
    return scenarios


def rerank_scenarios(vocab: int) -> dict[str, list[tuple[int, ...]]]:
    """One 12-token query with 10 and 50 documents of 64-192 tokens, as pair rows."""
    query = synthetic_rows([12], vocab, 1)[0][:-1]
    scenarios = {}
    for count in RERANK_DOCUMENTS:
        lengths = _rng(count).integers(64, 193, count).tolist()
        documents = synthetic_rows(lengths, vocab, count)
        scenarios[f"query+{count}docs"] = [
            query + document[1:] for document in documents
        ]
    return scenarios
