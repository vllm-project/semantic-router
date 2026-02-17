//go:build windows || !cgo

// Package nlp_binding provides Go bindings for BM25 and N-gram keyword
// classification. This is the mock implementation for platforms without CGo.
package nlp_binding

import "fmt"

// MatchResult represents the output of a BM25 or N-gram classification.
type MatchResult struct {
	Matched         bool
	RuleName        string
	MatchedKeywords []string
	Scores          []float32
	MatchCount      int
	TotalKeywords   int
}

// ---------------------------------------------------------------------------
// BM25 Classifier (mock)
// ---------------------------------------------------------------------------

// BM25Classifier wraps a Rust-backed BM25 keyword classifier.
type BM25Classifier struct{}

// NewBM25Classifier creates a new BM25 classifier instance (mock).
func NewBM25Classifier() *BM25Classifier {
	return &BM25Classifier{}
}

// AddRule adds a keyword rule (mock - no-op).
func (c *BM25Classifier) AddRule(name, operator string, keywords []string, threshold float32, caseSensitive bool) error {
	return fmt.Errorf("nlp-binding: BM25 classifier not available (built without CGo)")
}

// Classify runs classification (mock - always returns no match).
func (c *BM25Classifier) Classify(text string) MatchResult {
	return MatchResult{}
}

// Free releases resources (mock - no-op).
func (c *BM25Classifier) Free() {}

// ---------------------------------------------------------------------------
// N-gram Classifier (mock)
// ---------------------------------------------------------------------------

// NgramClassifier wraps a Rust-backed N-gram keyword classifier.
type NgramClassifier struct{}

// NewNgramClassifier creates a new N-gram classifier instance (mock).
func NewNgramClassifier() *NgramClassifier {
	return &NgramClassifier{}
}

// AddRule adds a keyword rule (mock - no-op).
func (c *NgramClassifier) AddRule(name, operator string, keywords []string, threshold float32, caseSensitive bool, arity int) error {
	return fmt.Errorf("nlp-binding: N-gram classifier not available (built without CGo)")
}

// Classify runs classification (mock - always returns no match).
func (c *NgramClassifier) Classify(text string) MatchResult {
	return MatchResult{}
}

// Free releases resources (mock - no-op).
func (c *NgramClassifier) Free() {}
