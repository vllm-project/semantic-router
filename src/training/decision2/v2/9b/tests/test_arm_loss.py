import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def synthetic_batch():
    torch.manual_seed(3)
    table = torch.randn(9, 4096)
    candidate_ids = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, -1], [0, 8, 2], [6, 7, -1]]
    )
    mask = candidate_ids >= 0
    return table, {
        "rows": torch.arange(5),
        "candidates": table[candidate_ids.clamp(min=0)] * mask[..., None],
        "candidate_mask": mask,
        "candidate_ids": candidate_ids,
        "query": torch.randn(5, 4096),
        "query_ids": torch.arange(5),
        "labels": torch.tensor([1, 0, 1, 2, 0]),
        "task_type": torch.tensor([0, 2, 1, 0, 1]),
        "counts": mask.sum(-1),
        "levels": torch.tensor(
            [[-1, -1, -1], [0, 1, 2], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
        ),
    }


@unittest.skipUnless(HAS_TORCH, "requires torch")
class ArmLossTest(unittest.TestCase):
    def test_every_arm_trains_every_group(self):
        from clm9b.arms import ARMS, ArmModel
        from clm9b.train_heads import arm_loss

        table, batch = synthetic_batch()
        teacher = (
            torch.full((5, 3), 1 / 3).masked_fill(~batch["candidate_mask"], 0.0),
            torch.ones(5, dtype=torch.bool),
        )
        teacher[0][2, :2] = 0.5
        teacher[0][4, :2] = 0.5
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = ArmModel(arm)
                terms = arm_loss(model, batch, table, teacher)
                loss = terms["relative"] + terms["score"]
                self.assertTrue(loss.requires_grad)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                groups = {
                    "relative": list(model.head.parameters()),
                    "score": list(model.score.parameters()),
                }
                for name, params in groups.items():
                    if not params:
                        continue
                    norm = torch.sqrt(
                        sum(
                            (p.grad.detach() ** 2).sum()
                            for p in params
                            if p.grad is not None
                        )
                    )
                    self.assertGreater(
                        float(norm), 0.0, f"{arm}/{name} received no gradient"
                    )
                if ARMS[arm]["objective"].endswith("replay"):
                    self.assertGreater(float(terms["replay_kl"]), 0.0)


if __name__ == "__main__":
    unittest.main()
