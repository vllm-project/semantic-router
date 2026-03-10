import json
from datetime import datetime
from typing import Optional


class ReportMixin:
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate test report

        Args:
            output_path: output path; if None, auto-generate filename

        Returns:
            str: report file path
        """
        if not self.test_results:
            raise ValueError("No test results available")

        report = self._generate_report()

        if output_path is None:
            approach = "CoT" if self.model_config.use_cot else "Direct"
            output_path = (
                f"{self.model_config.model_name.replace('/', '_')}_{approach}.json"
            )

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(report)

        return output_path

    def _generate_report(self) -> str:
        """Generate JSON report"""
        if self.system_eval_responses:
            overall_score = next(
                (r.score for r in self.test_results if r.test_name == "system_eval"),
                0.0,
            )
            report_data = {
                "endpoint": self._normalize_eval_endpoint(),
                "generated_at": datetime.now().isoformat() + "Z",
                "overall_score": overall_score,
                "summary": self.system_eval_summary,
                "results": self.system_eval_responses,
            }
            return json.dumps(report_data, indent=4, ensure_ascii=False)

        arc_result = next(
            (r for r in self.test_results if r.test_name == "arc_challenge"), None
        )
        mmlu_result = next(
            (r for r in self.test_results if r.test_name == "mmlu_pro"), None
        )

        report_data = {
            "model_name": self.model_config.model_name,
            "approach": "Chain-of-Thought" if self.model_config.use_cot else "Direct",
        }

        if arc_result:
            report_data["global_metrics"] = {
                "reasoning": {
                    "benchmark": "ARC-Challenge",
                    "results": {"overall": round(arc_result.score, 3)},
                }
            }

        if mmlu_result:
            report_data["domain_scores"] = {
                "benchmark": "MMLU-Pro",
                "results": {k: round(v, 2) for k, v in mmlu_result.metrics.items()},
            }

        metadata = {
            "test_time": datetime.now().isoformat() + "Z",
            "seed": self.model_config.seed,
        }
        if arc_result and arc_result.details:
            metadata.update(arc_result.details)
        elif mmlu_result and mmlu_result.details:
            metadata.update(mmlu_result.details)

        report_data["metadata"] = metadata
        return json.dumps(report_data, indent=4, ensure_ascii=False)
