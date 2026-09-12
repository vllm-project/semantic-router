"""Observed supervision and actual gradient accounting, without changing loss."""

from collections import Counter, defaultdict


def logit_loss_gradient(logits, targets, mask, global_examples):
    """Exact d(loss)/d(logit) for the trainer's observed-mean BCE objective."""
    if global_examples <= 0 or bool((mask.sum(-1) == 0).any()):
        raise ValueError("Positive global examples and observed supervision required")
    return (
        (logits.detach().float().sigmoid() - targets.float())
        * mask
        / mask.sum(-1, keepdim=True)
        / global_examples
    )


class SupervisionDiagnostics:
    def __init__(self, labels):
        self.labels = labels
        self.sources = {}
        self.head_gradient_norm_sums = defaultdict(lambda: [0.0] * len(labels))
        self.gradient_steps = 0

    def observe(self, rows, logits, targets, mask, global_examples):
        gradients = (
            logit_loss_gradient(logits, targets, mask, global_examples).cpu().tolist()
        )
        for row, gradient in zip(rows, gradients, strict=True):
            source = row["source"]
            if source not in self.sources:
                self.sources[source] = {
                    "draws": 0,
                    "unique_ids": set(),
                    "observed_count_distribution": Counter(),
                    "per_label": {
                        label: {
                            "positive": 0,
                            "negative": 0,
                            "unknown": 0,
                            "positive_logit_gradient_absolute_sum": 0.0,
                            "negative_logit_gradient_absolute_sum": 0.0,
                        }
                        for label in self.labels
                    },
                }
            stats = self.sources[source]
            stats["draws"] += 1
            stats["unique_ids"].add(row["id"])
            stats["observed_count_distribution"][sum(row["label_mask"])] += 1
            for label, target, observed, value in zip(
                self.labels, row["targets"], row["label_mask"], gradient, strict=True
            ):
                state = (
                    "unknown" if not observed else "positive" if target else "negative"
                )
                stats["per_label"][label][state] += 1
                if observed:
                    stats["per_label"][label][
                        f"{state}_logit_gradient_absolute_sum"
                    ] += abs(value)

    def observe_head_gradients(self, model):
        """Actual final-head row gradient norms before optimizer clipping."""
        found = False
        for name, parameter in model.named_parameters():
            if (
                "classifier" in name.split(".")
                and parameter.grad is not None
                and parameter.ndim in {1, 2}
                and parameter.shape[0] == len(self.labels)
            ):
                norms = (
                    parameter.grad.detach()
                    .float()
                    .reshape(len(self.labels), -1)
                    .norm(dim=1)
                    .cpu()
                    .tolist()
                )
                for index, value in enumerate(norms):
                    self.head_gradient_norm_sums[name][index] += value
                found = True
        if not found:
            raise ValueError("No trainable final classifier head gradients found")
        self.gradient_steps += 1

    def snapshot(self):
        return {
            "objective": "Per-example mean over observed labels, then global-example mean",
            "logit_gradient_scope": "Exact BCE gradient before clipping; source-wise absolute sums are not signed net encoder gradients",
            "head_gradient_scope": "Actual final classifier row norm before clipping, summed per optimizer step; mixed-source gradients cannot be attributed to one source",
            "gradient_steps": self.gradient_steps,
            "sources": {
                source: {
                    **stats,
                    "unique_ids": len(stats["unique_ids"]),
                    "observed_count_distribution": dict(
                        stats["observed_count_distribution"]
                    ),
                }
                for source, stats in self.sources.items()
            },
            "head_gradient_norm_sums": {
                name: dict(zip(self.labels, values, strict=True))
                for name, values in self.head_gradient_norm_sums.items()
            },
        }
