import re

ANSWER_PATTERN_ARC = re.compile(r"(?:answer(?:\sis)?:?\s*)(A|B|C|D)", re.IGNORECASE)
ANSWER_PATTERN_MMLU = re.compile(
    r"(?:answer(?:\sis)?:?\s*)(A|B|C|D|E|F|G|H|I|J)", re.IGNORECASE
)
