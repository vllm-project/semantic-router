//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

import (
	"errors"
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestONNXEmbeddingModelTypeFailsClosed(t *testing.T) {
	t.Parallel()

	unsupported := []string{"", "qwen3", "gemma", "auto", "unknown"}
	for _, modelType := range unsupported {
		modelType := modelType
		t.Run(modelType, func(t *testing.T) {
			t.Parallel()
			calls := []func() error{
				func() error {
					_, err := GetEmbeddingWithModelType("text", modelType, 0)
					return err
				},
				func() error {
					_, err := GetEmbedding2DMatryoshka("text", modelType, 0, 0)
					return err
				},
			}
			for _, call := range calls {
				if err := call(); err == nil || !strings.Contains(err.Error(), "unsupported ONNX embedding model type") {
					t.Fatalf("model type %q must fail before inference, got %v", modelType, err)
				}
			}
		})
	}
}

func TestONNXPublicEmbeddingControlsFailBeforeFFI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		call func() error
	}{
		{name: "metadata negative dimension", call: func() error { _, err := GetEmbeddingWithMetadata("text", 0.5, 0.5, -1); return err }},
		{name: "matryoshka negative layer", call: func() error { _, err := GetEmbedding2DMatryoshka("text", "mmbert", -1, 256); return err }},
		{name: "batch negative dimension", call: func() error { _, err := GetEmbeddingsBatch([]string{"text"}, 0, -1); return err }},
		{name: "batch similarity negative top-k", call: func() error {
			_, err := CalculateSimilarityBatch("query", []string{"candidate"}, -1, "mmbert", 0)
			return err
		}},
		{name: "legacy embedding negative max length", call: func() error { _, err := GetEmbedding("text", -1); return err }},
		{name: "multimodal text negative dimension", call: func() error { _, err := MultiModalEncodeText("text", -1); return err }},
		{name: "multimodal image negative dimension", call: func() error { _, err := MultiModalEncodeImage([]float32{0, 0, 0}, 1, 1, -1); return err }},
		{name: "multimodal audio negative dimension", call: func() error { _, err := MultiModalEncodeAudio([]float32{0}, 1, 1, -1); return err }},
		{name: "metadata non-finite priority", call: func() error { _, err := GetEmbeddingWithMetadata("text", float32(math.Inf(1)), 0.5, 0); return err }},
	}
	if strconv.IntSize == 64 {
		tooLargeValue := int64(1) << 31
		tooLarge := int(tooLargeValue)
		tests = append(tests,
			struct {
				name string
				call func() error
			}{name: "matryoshka layer exceeds C int", call: func() error {
				_, err := GetEmbedding2DMatryoshka("text", "mmbert", tooLarge, 0)
				return err
			}},
			struct {
				name string
				call func() error
			}{name: "batch top-k exceeds C int", call: func() error {
				_, err := CalculateSimilarityBatch("query", []string{"candidate"}, tooLarge, "mmbert", 0)
				return err
			}},
		)
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := test.call(); err == nil {
				t.Fatal("invalid public embedding controls unexpectedly reached FFI")
			}
		})
	}
}

func TestONNXEncodedImageControlsFailBeforeDecode(t *testing.T) {
	t.Parallel()

	for name, call := range map[string]func() error{
		"bytes":  func() error { _, err := MultiModalEncodeImageFromBytes(nil, -1); return err },
		"base64": func() error { _, err := MultiModalEncodeImageFromBase64("not-base64", -1); return err },
	} {
		if err := call(); err == nil || !strings.Contains(err.Error(), "target dimension") {
			t.Fatalf("%s invalid dimension error = %v, want fail-fast target validation", name, err)
		}
	}
}

func TestONNXLegacySimilarityControlsFailClosed(t *testing.T) {
	t.Parallel()

	if got := CalculateSimilarity("left", "right", -1); got != -1 {
		t.Fatalf("CalculateSimilarity() = %v, want fail-closed sentinel -1", got)
	}
	if got := FindMostSimilar("query", []string{"candidate"}, -1); got.Index != -1 || got.Score != -1 {
		t.Fatalf("FindMostSimilar() = %#v, want fail-closed sentinel", got)
	}
}

func TestONNXSimilarityModelTypeContract(t *testing.T) {
	t.Parallel()

	for _, modelType := range []string{"mmbert", "MMBERT", "auto", " AUTO "} {
		normalized, err := validateONNXSimilarityModelType(modelType)
		if err != nil {
			t.Fatalf("similarity model type %q must be accepted: %v", modelType, err)
		}
		if normalized != "mmbert" {
			t.Fatalf("similarity model type %q normalized to %q, want mmbert", modelType, normalized)
		}
	}

	for _, modelType := range []string{"", "qwen3", "gemma", "multimodal", "unknown"} {
		if _, err := validateONNXSimilarityModelType(modelType); err == nil ||
			!strings.Contains(err.Error(), "unsupported ONNX embedding model type") {
			t.Fatalf("similarity model type %q must fail closed, got %v", modelType, err)
		}
	}

	// Exercise both exported APIs without requiring a model: an embedded NUL is
	// rejected after model normalization, proving that auto reaches input
	// validation instead of the unsupported-model branch.
	for name, call := range map[string]func() error{
		"pair": func() error {
			_, err := CalculateEmbeddingSimilarity("invalid\x00text", "valid", "auto", 0)
			return err
		},
		"batch": func() error {
			_, err := CalculateSimilarityBatch("invalid\x00query", []string{"valid"}, 1, "auto", 0)
			return err
		},
	} {
		if err := call(); !errors.Is(err, errEmbeddedNULByte) {
			t.Fatalf("%s similarity API did not accept auto before input validation: %v", name, err)
		}
	}
}

func TestONNXMultimodalTensorShapesFailBeforeFFI(t *testing.T) {
	t.Parallel()

	imageCases := []struct {
		name          string
		pixels        []float32
		height, width int
	}{
		{name: "negative height", pixels: []float32{1}, height: -1, width: 1},
		{name: "oversized geometry", pixels: []float32{1}, height: 8193, width: 1},
		{name: "length mismatch", pixels: []float32{1}, height: 1, width: 1},
	}
	for _, test := range imageCases {
		if _, err := MultiModalEncodeImage(test.pixels, test.height, test.width, 0); err == nil {
			t.Fatalf("%s must fail before inference", test.name)
		}
	}

	audioCases := []struct {
		name              string
		mel               []float32
		nMels, timeFrames int
	}{
		{name: "negative dimension", mel: []float32{1}, nMels: -1, timeFrames: 1},
		{name: "zero time frames", mel: []float32{1}, nMels: 1, timeFrames: 0},
		{name: "short backing array", mel: []float32{1}, nMels: 80, timeFrames: 3000},
		{name: "long backing array", mel: []float32{1, 2}, nMels: 1, timeFrames: 1},
		{name: "mel C int narrowing", mel: []float32{1}, nMels: 1 << 31, timeFrames: 1},
		{name: "frame C int narrowing", mel: []float32{1}, nMels: 1, timeFrames: 1 << 31},
	}
	for _, test := range audioCases {
		if _, err := MultiModalEncodeAudio(test.mel, test.nMels, test.timeFrames, 0); err == nil {
			t.Fatalf("%s must fail before inference", test.name)
		}
	}
}
