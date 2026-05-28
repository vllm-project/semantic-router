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


def main():
    """Main function to generate and save the dashboard"""
    panels = generate_dashboard()

    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": panels,
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
                }
            ]
        },
        "time": {"from": "now-3h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": "vLLM Semantic Router Dashboard",
        "uid": "vllm-semantic-router",
        "version": 1,
        "weekStart": "",
    }

    output_file = "llm-router-dashboard.serve.json"
    with open(output_file, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"Dashboard generated successfully: {output_file}")
    print(f"Total panels: {len(panels)}")


if __name__ == "__main__":
    main()
