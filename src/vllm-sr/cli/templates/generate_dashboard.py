#!/usr/bin/env python3
"""
Generate Grafana Dashboard JSON for vLLM Semantic Router
"""

import json
import sys
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).resolve().parent
if str(_TEMPLATES_DIR) not in sys.path:
    sys.path.insert(0, str(_TEMPLATES_DIR))

from grafana_dashboard_sections import generate_all_dashboard_panels  # noqa: E402


def generate_dashboard():
    """Generate the complete dashboard"""
    return generate_all_dashboard_panels()


def generate_dashboard_document():
    """Generate metadata and panels from the same reproducible source."""
    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": generate_dashboard(),
        "refresh": "10s",
        "schemaVersion": 39,
        "tags": ["llm", "router", "semantic"],
        "templating": {
            "list": [
                {
                    "current": {
                        "selected": False,
                        "text": "Prometheus",
                        "value": "Prometheus",
                    },
                    "hide": 0,
                    "includeAll": False,
                    "label": "Datasource",
                    "multi": False,
                    "name": "DS_PROMETHEUS",
                    "options": [],
                    "query": "prometheus",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "type": "datasource",
                },
                {
                    "name": "router_instance",
                    "label": "Router instance",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                    "query": 'label_values(up{job="semantic-router"}, instance)',
                    "definition": 'label_values(up{job="semantic-router"}, instance)',
                    "refresh": 1,
                    "includeAll": True,
                    "allValue": ".*",
                    "multi": True,
                    "current": {"text": "All", "value": "$__all"},
                    "options": [],
                },
            ]
        },
        "time": {"from": "now-3h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": "vLLM Semantic Router",
        "description": "Measured inference outcomes, recipe routing, reported model usage, plugins and telemetry health. Missing observations remain No data.",
        "uid": "vllm-semantic-router",
        "version": 2,
        "weekStart": "",
    }


def main():
    """Generate the checked-in dashboard without depending on the caller's cwd."""
    dashboard = generate_dashboard_document()
    output_file = _TEMPLATES_DIR / "llm-router-dashboard.serve.json"
    with open(output_file, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"Dashboard generated successfully: {output_file}")
    print(f"Top-level panels: {len(dashboard['panels'])}")


if __name__ == "__main__":
    main()
