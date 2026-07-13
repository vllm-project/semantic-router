package candle_binding

import (
	"math"
	"strconv"
	"testing"
)

func TestEmbeddingModelTypeName(t *testing.T) {
	t.Parallel()

	tests := []struct {
		modelType int
		want      string
	}{
		{modelType: 0, want: "qwen3"},
		{modelType: 1, want: "gemma"},
		{modelType: 2, want: "mmbert"},
		{modelType: 3, want: "multimodal"},
		{modelType: -1, want: "unknown"},
	}

	for _, test := range tests {
		if got := embeddingModelTypeName(test.modelType); got != test.want {
			t.Fatalf("embeddingModelTypeName(%d) = %q, want %q", test.modelType, got, test.want)
		}
	}
}

func TestValidateSimilarityModelType(t *testing.T) {
	t.Parallel()

	for _, modelType := range []string{"auto", "qwen3", "gemma", "mmbert"} {
		if err := validateSimilarityModelType(modelType); err != nil {
			t.Fatalf("validateSimilarityModelType(%q) = %v", modelType, err)
		}
	}
	for _, modelType := range []string{"", "multimodal", "unknown", "MMBERT"} {
		if err := validateSimilarityModelType(modelType); err == nil {
			t.Fatalf("validateSimilarityModelType(%q) unexpectedly succeeded", modelType)
		}
	}
}

func TestNormalizeSimilarityOptions(t *testing.T) {
	t.Parallel()

	options, err := normalizeSimilarityOptions(SimilarityOptions{ModelType: "auto"})
	if err != nil {
		t.Fatal(err)
	}
	if options.QualityPriority != 0.5 || options.LatencyPriority != 0.5 {
		t.Fatalf("auto defaults = %#v", options)
	}
	for _, invalid := range []SimilarityOptions{
		{ModelType: "qwen3", TargetLayer: 6},
		{ModelType: "mmbert", TargetLayer: -1},
		{ModelType: "mmbert", TargetDim: -1},
		{ModelType: "auto", QualityPriority: 1.1},
		{ModelType: "auto", LatencyPriority: -0.1},
		{ModelType: "auto", QualityPriority: float32(math.NaN())},
		{ModelType: "auto", LatencyPriority: float32(math.Inf(1))},
	} {
		if _, err := normalizeSimilarityOptions(invalid); err == nil {
			t.Fatalf("normalizeSimilarityOptions(%#v) unexpectedly succeeded", invalid)
		}
	}

	if strconv.IntSize == 64 {
		tooLarge := int64(1) << 31
		for _, invalid := range []SimilarityOptions{
			{ModelType: "mmbert", TargetLayer: int(tooLarge)},
			{ModelType: "mmbert", TargetDim: int(tooLarge)},
		} {
			if _, err := normalizeSimilarityOptions(invalid); err == nil {
				t.Fatalf("normalizeSimilarityOptions(%#v) accepted a value that cannot fit C.int", invalid)
			}
		}
	}
}

func TestPublicEmbeddingControlsFailBeforeNativeDispatch(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		call func() error
	}{
		{name: "tokenize negative length", call: func() error { _, err := TokenizeText("text", -1); return err }},
		{name: "bert negative length", call: func() error { _, err := GetEmbedding("text", -1); return err }},
		{name: "batched negative dimension", call: func() error { _, err := GetEmbeddingBatched("text", "qwen3", -1); return err }},
		{name: "explicit model negative dimension", call: func() error { _, err := GetEmbeddingWithModelType("text", "mmbert", -1); return err }},
		{name: "metadata negative dimension", call: func() error { _, err := GetEmbeddingWithMetadata("text", 0.5, 0.5, -1); return err }},
		{name: "matryoshka negative layer", call: func() error { _, err := GetEmbedding2DMatryoshka("text", "mmbert", -1, 256); return err }},
		{name: "batch similarity negative top-k", call: func() error {
			_, err := CalculateSimilarityBatch("query", []string{"candidate"}, -1, "mmbert", 0)
			return err
		}},
		{name: "multimodal text negative dimension", call: func() error { _, err := MultiModalEncodeText("text", -1); return err }},
		{name: "multimodal image negative dimension", call: func() error { _, err := MultiModalEncodeImage([]float32{0, 0, 0}, 1, 1, -1); return err }},
		{name: "multimodal audio negative dimension", call: func() error { _, err := MultiModalEncodeAudio([]float32{0}, 1, 1, -1); return err }},
		{name: "smart non-finite priority", call: func() error { _, err := GetEmbeddingSmart("text", float32(math.NaN()), 0.5); return err }},
	}
	if strconv.IntSize == 64 {
		tooLargeValue := int64(1) << 31
		tooLarge := int(tooLargeValue)
		tests = append(tests,
			struct {
				name string
				call func() error
			}{name: "metadata dimension exceeds C int", call: func() error {
				_, err := GetEmbeddingWithMetadata("text", 0.5, 0.5, tooLarge)
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
				t.Fatal("invalid public embedding controls unexpectedly reached native dispatch")
			}
		})
	}
}

func TestLegacySimilarityControlsFailClosed(t *testing.T) {
	t.Parallel()

	if got := CalculateSimilarity("left", "right", -1); got != -1 {
		t.Fatalf("CalculateSimilarity() = %v, want fail-closed sentinel -1", got)
	}
	if got := FindMostSimilar("query", []string{"candidate"}, -1); got.Index != -1 || got.Score != -1 {
		t.Fatalf("FindMostSimilar() = %#v, want fail-closed sentinel", got)
	}
}
