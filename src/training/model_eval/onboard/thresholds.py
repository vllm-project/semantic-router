from datetime import datetime
from typing import Any, Dict, List


def build_threshold_report(
    summary: List[Dict[str, Any]],
    min_accuracy: float,
    max_latency_ms: float,
) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []
    for item in summary:
        accuracy = item.get("accuracy")
        latency = item.get("avg_latency_ms")
        if accuracy is not None and accuracy < min_accuracy:
            violations.append(
                {
                    "dataset": item.get("dataset"),
                    "dimension": item.get("dimension"),
                    "metric": "accuracy",
                    "value": accuracy,
                    "threshold": min_accuracy,
                }
            )
        if latency is not None and latency > max_latency_ms:
            violations.append(
                {
                    "dataset": item.get("dataset"),
                    "dimension": item.get("dimension"),
                    "metric": "avg_latency_ms",
                    "value": latency,
                    "threshold": max_latency_ms,
                }
            )

    return {
        "generated_at": datetime.now().isoformat() + "Z",
        "policy": {
            "min_accuracy": min_accuracy,
            "max_latency_ms": max_latency_ms,
        },
        "summary": summary,
        "violations": violations,
        "passed": len(violations) == 0,
    }
