//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

// Package nlp_binding provides Go bindings for BM25 and N-gram keyword
// classification backed by Rust implementations via C FFI.
//
// It follows the same convention as candle-binding:
//   - Rust builds a static/dynamic library via cargo build --release
//   - Go links against it via #cgo LDFLAGS
//   - C strings are passed via CString/CStr
//   - Rust-allocated memory is freed via explicit free functions
package nlp_binding

import (
	"fmt"
	"unsafe"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lnlp_binding -ldl -lm -lpthread
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

// ClassifyResult matches the Rust #[repr(C)] struct.
typedef struct {
    bool matched;
    char* rule_name;
    char** matched_keywords;
    float* scores;
    int match_count;
    int total_keywords;
} ClassifyResult;

// BM25 classifier FFI
extern uint64_t bm25_classifier_new();
extern bool bm25_classifier_add_rule(
    uint64_t handle,
    const char* name,
    const char* op,
    const char** keywords,
    int num_keywords,
    float threshold,
    bool case_sensitive
);
extern ClassifyResult bm25_classifier_classify(uint64_t handle, const char* text);
extern void bm25_classifier_free(uint64_t handle);

// N-gram classifier FFI
extern uint64_t ngram_classifier_new();
extern bool ngram_classifier_add_rule(
    uint64_t handle,
    const char* name,
    const char* op,
    const char** keywords,
    int num_keywords,
    float threshold,
    bool case_sensitive,
    int arity
);
extern ClassifyResult ngram_classifier_classify(uint64_t handle, const char* text);
extern void ngram_classifier_free(uint64_t handle);

// Memory management
extern void free_classify_result(ClassifyResult result);
*/
import "C"

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
// BM25 Classifier
// ---------------------------------------------------------------------------

// BM25Classifier wraps a Rust-backed BM25 keyword classifier.
type BM25Classifier struct {
	handle C.uint64_t
}

// NewBM25Classifier creates a new BM25 classifier instance.
func NewBM25Classifier() *BM25Classifier {
	handle := C.bm25_classifier_new()
	return &BM25Classifier{handle: handle}
}

// AddRule adds a keyword rule to the BM25 classifier.
//
// Parameters:
//   - name: rule name (used as category)
//   - operator: "AND", "OR", or "NOR"
//   - keywords: list of keywords for this rule
//   - threshold: BM25 score threshold for a keyword to count as matched
//   - caseSensitive: whether matching is case-sensitive
func (c *BM25Classifier) AddRule(name, operator string, keywords []string, threshold float32, caseSensitive bool) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	cOp := C.CString(operator)
	defer C.free(unsafe.Pointer(cOp))

	cKeywords := make([]*C.char, len(keywords))
	for i, kw := range keywords {
		cKeywords[i] = C.CString(kw)
		defer C.free(unsafe.Pointer(cKeywords[i]))
	}

	var kwPtr **C.char
	if len(cKeywords) > 0 {
		kwPtr = &cKeywords[0]
	}

	ok := C.bm25_classifier_add_rule(
		c.handle,
		cName,
		cOp,
		kwPtr,
		C.int(len(keywords)),
		C.float(threshold),
		C.bool(caseSensitive),
	)

	if !bool(ok) {
		return fmt.Errorf("failed to add BM25 rule %q", name)
	}
	return nil
}

// Classify runs BM25 classification on the input text.
func (c *BM25Classifier) Classify(text string) MatchResult {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.bm25_classifier_classify(c.handle, cText)
	defer C.free_classify_result(result)

	return convertResult(result)
}

// Free releases the Rust-side resources for this classifier.
func (c *BM25Classifier) Free() {
	C.bm25_classifier_free(c.handle)
}

// ---------------------------------------------------------------------------
// N-gram Classifier
// ---------------------------------------------------------------------------

// NgramClassifier wraps a Rust-backed N-gram keyword classifier.
type NgramClassifier struct {
	handle C.uint64_t
}

// NewNgramClassifier creates a new N-gram classifier instance.
func NewNgramClassifier() *NgramClassifier {
	handle := C.ngram_classifier_new()
	return &NgramClassifier{handle: handle}
}

// AddRule adds a keyword rule to the N-gram classifier.
//
// Parameters:
//   - name: rule name (used as category)
//   - operator: "AND", "OR", or "NOR"
//   - keywords: list of keywords for this rule
//   - threshold: similarity threshold (0.0-1.0) for n-gram matching
//   - caseSensitive: whether matching is case-sensitive
//   - arity: n-gram arity (2 for bigrams, 3 for trigrams, etc.)
func (c *NgramClassifier) AddRule(name, operator string, keywords []string, threshold float32, caseSensitive bool, arity int) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	cOp := C.CString(operator)
	defer C.free(unsafe.Pointer(cOp))

	cKeywords := make([]*C.char, len(keywords))
	for i, kw := range keywords {
		cKeywords[i] = C.CString(kw)
		defer C.free(unsafe.Pointer(cKeywords[i]))
	}

	var kwPtr **C.char
	if len(cKeywords) > 0 {
		kwPtr = &cKeywords[0]
	}

	ok := C.ngram_classifier_add_rule(
		c.handle,
		cName,
		cOp,
		kwPtr,
		C.int(len(keywords)),
		C.float(threshold),
		C.bool(caseSensitive),
		C.int(arity),
	)

	if !bool(ok) {
		return fmt.Errorf("failed to add N-gram rule %q", name)
	}
	return nil
}

// Classify runs N-gram classification on the input text.
func (c *NgramClassifier) Classify(text string) MatchResult {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ngram_classifier_classify(c.handle, cText)
	defer C.free_classify_result(result)

	return convertResult(result)
}

// Free releases the Rust-side resources for this classifier.
func (c *NgramClassifier) Free() {
	C.ngram_classifier_free(c.handle)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func convertResult(result C.ClassifyResult) MatchResult {
	mr := MatchResult{
		Matched:       bool(result.matched),
		MatchCount:    int(result.match_count),
		TotalKeywords: int(result.total_keywords),
	}

	if !mr.Matched {
		return mr
	}

	if result.rule_name != nil {
		mr.RuleName = C.GoString(result.rule_name)
	}

	count := int(result.match_count)
	if count > 0 && result.matched_keywords != nil {
		cKwSlice := unsafe.Slice(result.matched_keywords, count)
		mr.MatchedKeywords = make([]string, count)
		for i := 0; i < count; i++ {
			if cKwSlice[i] != nil {
				mr.MatchedKeywords[i] = C.GoString(cKwSlice[i])
			}
		}
	}

	if count > 0 && result.scores != nil {
		cScoreSlice := unsafe.Slice(result.scores, count)
		mr.Scores = make([]float32, count)
		for i := 0; i < count; i++ {
			mr.Scores[i] = float32(cScoreSlice[i])
		}
	}

	return mr
}
