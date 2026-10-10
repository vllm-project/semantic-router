"""Tests for the exploration scripts' command lines and scoring loops, with no models."""

import types

import eval_lfm25
import eval_scx
import torch
import train_lfm25_encoder
import train_scx_router
import try_scx_zeroshot


def test_scx_training_defaults_match_the_reported_run():
    args = train_scx_router.build_parser().parse_args([])
    assert (args.epochs, args.batch_size, args.learning_rate, args.seed) == (
        10,
        8,
        2e-5,
        42,
    )
    assert args.output_dir.endswith("runs/scx_router_finetuned")


def test_lfm25_training_defaults_match_the_reported_run():
    args = train_lfm25_encoder.build_parser().parse_args(["--seed", "3"])
    assert (args.epochs, args.batch_size, args.learning_rate, args.seed) == (
        10,
        16,
        3e-5,
        3,
    )
    assert args.output_dir.endswith("runs/lfm25_encoder_finetuned")


def test_outputs_default_outside_the_source_files():
    assert (
        eval_lfm25.build_parser()
        .parse_args([])
        .out_json.endswith("runs/lfm25_finetuned_preds.json")
    )
    assert (
        try_scx_zeroshot.build_parser()
        .parse_args([])
        .out_json.endswith("runs/scx_zeroshot_results.json")
    )


def test_scx_eval_takes_the_model_and_output_path_positionally():
    args = eval_scx.build_parser().parse_args(
        ["some/model", "out.json", "--batch-size", "4"]
    )
    assert (args.model_path, args.out_json, args.batch_size) == (
        "some/model",
        "out.json",
        4,
    )


def test_zero_shot_picks_the_highest_scoring_description():
    descriptions = {
        "AR": "AR: a text-only response",
        "DIFFUSION": "DIFFUSION: generating an image",
        "BOTH": "BOTH: both a text explanation and an image",
    }
    best = iter(["DIFFUSION", "BOTH", "AR"])

    def pipe(text, label_texts, threshold):
        winner = descriptions[next(best)]
        return [
            [{"label": t, "score": 0.9 if t == winner else 0.05} for t in label_texts]
        ]

    assert try_scx_zeroshot.classify_rows(
        pipe, [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    ) == ["DIFFUSION", "BOTH", "AR"]


def test_scx_prediction_uses_the_scores_of_the_labels_each_example_has():
    class Dataset(list):
        pass

    class Collator:
        def __call__(self, examples):
            return {
                "input_ids": torch.zeros(len(examples), 1),
                "labels_text": [["AR", "DIFFUSION", "BOTH"]] * len(examples),
                "input_texts": ["x"] * len(examples),
                "labels": torch.zeros(len(examples)),
            }

    class Model:
        def __call__(self, input_ids):
            # padded to 4 columns; the last is padding and must be ignored
            return types.SimpleNamespace(
                logits=torch.tensor([[0.1, 0.2, 0.9, 5.0], [0.9, 0.1, 0.2, 5.0]])
            )

    preds = eval_scx.predict_scx(
        Model(), Dataset([0, 1]), Collator(), batch_size=2, device="cpu"
    )
    assert preds == ["BOTH", "AR"]
