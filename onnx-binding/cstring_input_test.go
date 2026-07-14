//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

import (
	"errors"
	"testing"
)

type onnxNULBoundaryCase struct {
	name string
	call func() error
}

func onnxNULBoundaryCases(invalid string) []onnxNULBoundaryCase {
	return []onnxNULBoundaryCase{
		{
			name: "mmBERT model path",
			call: func() error { return InitMmBertEmbeddingModel(invalid, true) },
		},
		{
			name: "sequence classifier model path",
			call: func() error { return InitMmBert32KIntentClassifier(invalid, true) },
		},
		{
			name: "token classifier model path",
			call: func() error { return InitMmBert32KPIIClassifier(invalid, true) },
		},
		{
			name: "classifier name",
			call: func() error { return initClassifier(invalid, "unused", false) },
		},
		{
			name: "token classifier name",
			call: func() error { return initTokenClassifier(invalid, "unused", false) },
		},
		{
			name: "embedding text",
			call: func() error {
				_, err := GetEmbeddingWithMetadata(invalid, 0, 0, 0)
				return err
			},
		},
		{
			name: "2D matryoshka text",
			call: func() error {
				_, err := GetEmbedding2DMatryoshka(invalid, "mmbert", 0, 0)
				return err
			},
		},
		{
			name: "embedding batch item",
			call: func() error {
				_, err := GetEmbeddingsBatch([]string{"valid", invalid}, 0, 0)
				return err
			},
		},
		{
			name: "similarity first text",
			call: func() error {
				_, err := CalculateEmbeddingSimilarity(invalid, "valid", "mmbert", 0)
				return err
			},
		},
		{
			name: "similarity second text",
			call: func() error {
				_, err := CalculateEmbeddingSimilarity("valid", invalid, "mmbert", 0)
				return err
			},
		},
		{
			name: "batch similarity query",
			call: func() error {
				_, err := CalculateSimilarityBatch(invalid, []string{"valid"}, 1, "mmbert", 0)
				return err
			},
		},
		{
			name: "batch similarity candidate",
			call: func() error {
				_, err := CalculateSimilarityBatch("valid", []string{"valid", invalid}, 1, "mmbert", 0)
				return err
			},
		},
		{
			name: "classification text",
			call: func() error {
				_, err := ClassifyMmBert32KIntent(invalid)
				return err
			},
		},
		{
			name: "PII classification text",
			call: func() error {
				_, err := ClassifyMmBert32KPII(invalid)
				return err
			},
		},
		{
			name: "multi-modal model path",
			call: func() error { return InitMultiModalEmbeddingModel(invalid, true) },
		},
		{
			name: "multi-modal text",
			call: func() error {
				_, err := MultiModalEncodeText(invalid, 0)
				return err
			},
		},
	}
}

func TestONNXGoBoundariesRejectEmbeddedNULBeforeFFI(t *testing.T) {
	t.Parallel()

	for _, test := range onnxNULBoundaryCases("trusted-prefix\x00hidden-suffix") {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := test.call(); !errors.Is(err, errEmbeddedNULByte) {
				t.Fatalf("expected embedded-NUL validation error, got %v", err)
			}
		})
	}
}

func TestCheckedCStringRejectsEmbeddedNULWithoutAllocation(t *testing.T) {
	t.Parallel()

	pointer, err := checkedCString("prefix\x00suffix", "test field")
	if pointer != nil {
		t.Fatal("invalid input must not allocate a C string")
	}
	if !errors.Is(err, errEmbeddedNULByte) {
		t.Fatalf("expected embedded-NUL validation error, got %v", err)
	}
}

func TestIsClassifierLoadedRejectsEmbeddedNUL(t *testing.T) {
	t.Parallel()

	if IsClassifierLoaded("intent\x00other") {
		t.Fatal("embedded-NUL classifier name must fail closed")
	}
}
