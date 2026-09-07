package classification

import (
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// This file documents the three-tier classifier-backend abstraction from the
// #2587 pluggable classifier backend mechanism (docs/plans in the main
// worktree). Each tier is matched to a model output shape rather than forcing
// every classifier through one universal interface:
//
//   - SequenceClassifierBackend (classifier_jailbreak_init.go) - text -> full
//     label/probability distribution. Implemented and wired for jailbreak
//     this round; category and hallucination-binary are intended future
//     consumers.
//   - ScoringBackend (below) - text -> a single continuous score, for
//     regression-style models such as a query-difficulty scorer. Reserved:
//     no backend implements it and nothing wires it in yet.
//   - TokenClassifierBackend (below) - text -> spans, for PII / GLiGuard-style
//     entity extraction. Reserved: no backend implements it and nothing
//     wires it in yet.

// ScoringBackend is implemented by classifier backends that report a single
// continuous score for a piece of text, rather than a label distribution
// (e.g. a trained query-difficulty model). A ScoringBackend's output is
// expected to plug into existing threshold-based decision logic unchanged -
// it does not carry a label of its own.
type ScoringBackend interface {
	Score(text string) (float64, error)
}

// TokenClassifierBackend is implemented by classifier backends that return
// per-span entities within a text, rather than a single label or score (e.g.
// PII entity extraction or a GLiGuard-style zero-shot span classifier).
type TokenClassifierBackend interface {
	ClassifyTokens(text string) ([]candle_binding.TokenEntity, error)
}
