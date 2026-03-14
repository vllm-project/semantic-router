import re

ANSWER_PATTERN_ARC = re.compile(r"(?:answer(?:\sis)?:?\s*)(A|B|C|D)", re.IGNORECASE)
ANSWER_PATTERN_MMLU = re.compile(
    r"(?:answer(?:\sis)?:?\s*)(A|B|C|D|E|F|G|H|I|J)", re.IGNORECASE
)

# Full system signal keys exposed by semantic-router classification pipeline.
SYSTEM_SIGNAL_KEYS = [
    "keyword",
    "embedding",
    "domain",
    "fact_check",
    "user_feedback",
    "preference",
    "language",
    "context",
    "complexity",
    "modality",
    "authz",
    "jailbreak",
    "pii",
]

# matched_signals keys are plural for some signals.
SYSTEM_MATCHED_SIGNAL_KEYS = {
    "keyword": "keywords",
    "embedding": "embeddings",
    "domain": "domains",
    "fact_check": "fact_check",
    "user_feedback": "user_feedback",
    "preference": "preferences",
    "language": "language",
    "context": "context",
    "complexity": "complexity",
    "modality": "modality",
    "authz": "authz",
    "jailbreak": "jailbreak",
    "pii": "pii",
}
