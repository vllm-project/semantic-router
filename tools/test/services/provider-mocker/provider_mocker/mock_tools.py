"""Tool name selection shared by the native tool-call fixtures."""

import json
import re
from typing import Any

MOCK_TOOL_NAME = re.compile(r"__mock_tool_name:([A-Za-z0-9_-]+):__")


def mock_tool_name(request_text: Any) -> str:
    """Return the tool a request names with its marker, by default ``lookup``."""
    match = MOCK_TOOL_NAME.search(json.dumps(request_text, default=str))
    return match.group(1) if match else "lookup"
