package onnx_binding

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestNormalizeONNXSimilarityOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		options SimilarityOptions
	}{
		{
			name: "auto resolves to mmbert and preserves controls",
			options: SimilarityOptions{
				ModelType:       " AUTO ",
				TargetLayer:     6,
				TargetDim:       256,
				QualityPriority: 0.25,
				LatencyPriority: 0.75,
			},
		},
		{
			name: "explicit mmbert",
			options: SimilarityOptions{
				ModelType: "MMBERT",
			},
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			got, err := normalizeONNXSimilarityOptions(test.options)
			if err != nil {
				t.Fatal(err)
			}
			if got.ModelType != "mmbert" {
				t.Fatalf("normalized model = %q, want mmbert", got.ModelType)
			}
			if got.TargetLayer != test.options.TargetLayer || got.TargetDim != test.options.TargetDim {
				t.Fatalf("matryoshka controls changed: got %#v, want %#v", got, test.options)
			}
			if got.QualityPriority != test.options.QualityPriority ||
				got.LatencyPriority != test.options.LatencyPriority {
				t.Fatalf("single-model priorities changed: got %#v, want %#v", got, test.options)
			}
		})
	}
}

func TestNormalizeONNXSimilarityOptionsFailsClosed(t *testing.T) {
	t.Parallel()

	type optionTest struct {
		name    string
		options SimilarityOptions
		want    string
	}
	tests := []optionTest{
		{name: "empty model", options: SimilarityOptions{}, want: "unsupported ONNX embedding model type"},
		{name: "unsupported model", options: SimilarityOptions{ModelType: "qwen3"}, want: "unsupported ONNX embedding model type"},
		{name: "negative layer", options: SimilarityOptions{ModelType: "mmbert", TargetLayer: -1}, want: "target layer cannot be negative"},
		{name: "negative dimension", options: SimilarityOptions{ModelType: "mmbert", TargetDim: -1}, want: "target dimension cannot be negative"},
		{name: "negative quality", options: SimilarityOptions{ModelType: "auto", QualityPriority: -0.1}, want: "between 0 and 1"},
		{name: "high latency", options: SimilarityOptions{ModelType: "auto", LatencyPriority: 1.1}, want: "between 0 and 1"},
		{name: "nan quality", options: SimilarityOptions{ModelType: "auto", QualityPriority: float32(math.NaN())}, want: "finite"},
		{name: "infinite latency", options: SimilarityOptions{ModelType: "auto", LatencyPriority: float32(math.Inf(1))}, want: "finite"},
	}
	if strconv.IntSize == 64 {
		tooLarge := int64(1) << 31
		tests = append(tests,
			optionTest{name: "layer exceeds C int", options: SimilarityOptions{ModelType: "mmbert", TargetLayer: int(tooLarge)}, want: "signed 32-bit C int"},
			optionTest{name: "dimension exceeds C int", options: SimilarityOptions{ModelType: "mmbert", TargetDim: int(tooLarge)}, want: "signed 32-bit C int"},
		)
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			_, err := normalizeONNXSimilarityOptions(test.options)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("normalizeONNXSimilarityOptions(%#v) error = %v, want %q", test.options, err, test.want)
			}
		})
	}
}
