from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Model configuration"""

    model_name: str
    endpoint: str
    api_key: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    use_cot: bool = False
    seed: int = 42
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Test result"""

    model_name: str
    test_name: str
    score: float
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None
