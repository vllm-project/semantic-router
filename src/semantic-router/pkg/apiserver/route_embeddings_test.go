//go:build !windows && cgo

package apiserver

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// stubMultiModalContract overrides the loaded model's dimension contract so
// request validation can be exercised without loading a checkpoint.
func stubMultiModalContract(t *testing.T, contract embeddingDimensionContract) {
	t.Helper()
	orig := multiModalDimensionContract
	t.Cleanup(func() { multiModalDimensionContract = orig })
	multiModalDimensionContract = func() embeddingDimensionContract { return contract }
}

func TestBuildBatchSimilarityMatchesRejectsInvalidNativeIndex(t *testing.T) {
	result := &candle_binding.BatchSimilarityOutput{
		Matches: []candle_binding.BatchSimilarityMatch{
			{Index: 2, Similarity: 0.9},
		},
	}

	if _, err := buildBatchSimilarityMatches(result, []string{"a", "b"}); err == nil {
		t.Fatalf("expected invalid native match index to return an error")
	}
}

func TestBuildBatchSimilarityMatchesIncludesCandidateText(t *testing.T) {
	result := &candle_binding.BatchSimilarityOutput{
		Matches: []candle_binding.BatchSimilarityMatch{
			{Index: 1, Similarity: 0.9},
			{Index: 0, Similarity: 0.7},
		},
	}

	matches, err := buildBatchSimilarityMatches(result, []string{"first", "second"})
	if err != nil {
		t.Fatalf("expected valid native matches, got %v", err)
	}

	if matches[0].Text != "second" || matches[1].Text != "first" {
		t.Fatalf("expected candidate text to follow native indexes, got %+v", matches)
	}
}

func TestValidateEmbeddingRequestRequiresTextsOrImages(t *testing.T) {
	req := EmbeddingRequest{Dimension: defaultEmbeddingDimension}

	code, message, ok := validateEmbeddingRequest(req, nil)
	if ok {
		t.Fatalf("expected empty texts and images to be invalid")
	}
	if code != "INVALID_INPUT" || message != "at least one of texts or images must be provided" {
		t.Fatalf("unexpected validation error %q: %q", code, message)
	}
}

func TestValidateEmbeddingRequestAcceptsImagesOnly(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{ModelID: "multi-modal-embed-small", Default: 384, Supported: []int{384, 256, 128, 64, 32}})
	req := EmbeddingRequest{
		Model:     "multimodal",
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 384,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected image-only request to be valid")
	}
}

func TestValidateEmbeddingRequestRejectsUnsafeImage(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{ModelID: "multi-modal-embed-small", Default: 384, Supported: []int{384, 256, 128, 64, 32}})
	req := EmbeddingRequest{
		Model:     "multimodal",
		Images:    []string{"https://example.com/cat.png"},
		Dimension: 384,
	}

	code, message, ok := validateEmbeddingRequest(req, nil)
	if ok {
		t.Fatalf("expected non-data-URI image to be rejected (SSRF guard)")
	}
	if code != "INVALID_IMAGE" {
		t.Fatalf("unexpected validation error code %q: %q", code, message)
	}
}

func TestValidateEmbeddingRequestRejectsMalformedBase64(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{ModelID: "multi-modal-embed-small", Default: 384, Supported: []int{384, 256, 128, 64, 32}})
	req := EmbeddingRequest{
		Model:     "multimodal",
		Images:    []string{"data:image/png;base64,!!!!"},
		Dimension: 384,
	}

	code, message, ok := validateEmbeddingRequest(req, nil)
	if ok {
		t.Fatalf("expected malformed base64 image to be rejected as a client error, not surface as a 500")
	}
	if code != "INVALID_IMAGE" {
		t.Fatalf("unexpected validation error code %q: %q", code, message)
	}
}

func TestValidateEmbeddingRequestAcceptsUppercaseDataURIScheme(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{ModelID: "multi-modal-embed-small", Default: 384, Supported: []int{384, 256, 128, 64, 32}})
	// "DATA:IMAGE/PNG;BASE64,..." passes the safety gate; it must also pass
	// decode-validation so it is not accepted here only to 500 at the FFI, whose
	// marker scan is case-sensitive (CanonicalDataURL normalizes it downstream).
	req := EmbeddingRequest{
		Model:     "multimodal",
		Images:    []string{"DATA:IMAGE/PNG;BASE64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 384,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected uppercase-scheme data URI to be accepted")
	}
}

func TestBuildEmbeddingResultsWrapsImageEncodeFailure(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{ModelID: "multi-modal-embed-small", Default: 384, Supported: []int{384, 256, 128, 64, 32}})
	// A validated safe data URI whose bytes are not a decodable image fails at the
	// FFI; buildEmbeddingResults must tag it as an imageEncodeError so the handler
	// maps it to 400 instead of 500.
	orig := multiModalEncodeImage
	defer func() { multiModalEncodeImage = orig }()
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return nil, errors.New("failed to decode image")
	}

	req := EmbeddingRequest{
		Model:     "multimodal",
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: 384,
	}
	_, _, err := buildEmbeddingResults(req, multiModalModelFallbackID)
	if err == nil {
		t.Fatalf("expected an error from a failing image encode")
	}
	var imgErr *imageEncodeError
	if !errors.As(err, &imgErr) {
		t.Fatalf("expected imageEncodeError, got %T: %v", err, err)
	}
	if imgErr.index != 0 {
		t.Fatalf("expected image index 0, got %d", imgErr.index)
	}
}

func TestClassifyEmbeddingErrorMapsImageEncodeFailureTo400(t *testing.T) {
	status, code, _ := classifyEmbeddingError(&imageEncodeError{index: 2, err: errors.New("bad image")})
	if status != http.StatusBadRequest || code != "INVALID_IMAGE" {
		t.Fatalf("expected 400 INVALID_IMAGE, got %d %q", status, code)
	}
}

func TestClassifyEmbeddingErrorMapsInternalFailureTo500(t *testing.T) {
	status, code, _ := classifyEmbeddingError(errors.New("model not loaded"))
	if status != http.StatusInternalServerError || code != "EMBEDDING_GENERATION_FAILED" {
		t.Fatalf("expected 500 EMBEDDING_GENERATION_FAILED, got %d %q", status, code)
	}
}

func TestValidateEmbeddingRequestRejectsTooManyImages(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{ModelID: "multi-modal-embed-small", Default: 384, Supported: []int{384, 256, 128, 64, 32}})
	images := make([]string, maxImagesPerRequest+1)
	for i := range images {
		images[i] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
	}
	req := EmbeddingRequest{Model: "multimodal", Images: images, Dimension: 384}

	code, _, ok := validateEmbeddingRequest(req, nil)
	if ok {
		t.Fatalf("expected more than %d images to be rejected", maxImagesPerRequest)
	}
	if code != "INVALID_INPUT" {
		t.Fatalf("unexpected validation error code %q", code)
	}
}

func TestAverageEmbeddingProcessingTimeUsesInputCount(t *testing.T) {
	req := EmbeddingRequest{
		Texts:  []string{"first", "second"},
		Images: []string{"data:image/png;base64,iVBORw0KGgo="},
	}

	got := averageEmbeddingProcessingTime(90, req)

	if got != 30 {
		t.Fatalf("expected average processing time to use text plus image inputs, got %.2f", got)
	}
}

func TestNormalizeBatchSimilarityLimitCapsTopKAtCandidateCount(t *testing.T) {
	req := BatchSimilarityRequest{
		Candidates: []string{"a", "b"},
		TopK:       10,
	}

	normalizeBatchSimilarityLimit(&req)

	if req.TopK != 2 {
		t.Fatalf("expected top_k to be capped at candidate count, got %d", req.TopK)
	}
}

func TestValidateSimilarityRequest(t *testing.T) {
	cases := []struct {
		name     string
		req      SimilarityRequest
		wantOK   bool
		wantCode string
	}{
		{"valid", SimilarityRequest{Text1: "a", Text2: "b", Dimension: defaultEmbeddingDimension}, true, ""},
		{"empty_text1", SimilarityRequest{Text1: "", Text2: "b", Dimension: defaultEmbeddingDimension}, false, "INVALID_INPUT"},
		{"whitespace_text2", SimilarityRequest{Text1: "a", Text2: "   ", Dimension: defaultEmbeddingDimension}, false, "INVALID_INPUT"},
		{"bad_dimension", SimilarityRequest{Text1: "a", Text2: "b", Dimension: 999}, false, "INVALID_DIMENSION"},
		{"dimension_64_allowed", SimilarityRequest{Text1: "a", Text2: "b", Dimension: 64}, true, ""},
		{"quality_priority_too_high", SimilarityRequest{Text1: "a", Text2: "b", Dimension: defaultEmbeddingDimension, QualityPriority: 1.5}, false, "INVALID_PARAMETER"},
		{"latency_priority_negative", SimilarityRequest{Text1: "a", Text2: "b", Dimension: defaultEmbeddingDimension, LatencyPriority: -0.1}, false, "INVALID_PARAMETER"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			code, _, ok := validateSimilarityRequest(tc.req)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if code != tc.wantCode {
				t.Fatalf("code = %q, want %q", code, tc.wantCode)
			}
		})
	}
}

func TestValidateBatchSimilarityRequestRejectsBlankAndOutOfRange(t *testing.T) {
	base := func() BatchSimilarityRequest {
		return BatchSimilarityRequest{Query: "q", Candidates: []string{"a", "b"}, Dimension: defaultEmbeddingDimension}
	}
	cases := []struct {
		name     string
		mutate   func(*BatchSimilarityRequest)
		wantOK   bool
		wantCode string
	}{
		{"valid", func(*BatchSimilarityRequest) {}, true, ""},
		{"whitespace_query", func(r *BatchSimilarityRequest) { r.Query = "  " }, false, "INVALID_INPUT"},
		{"blank_candidate", func(r *BatchSimilarityRequest) { r.Candidates = []string{"a", " "} }, false, "INVALID_INPUT"},
		{"quality_priority_too_high", func(r *BatchSimilarityRequest) { r.QualityPriority = 2 }, false, "INVALID_PARAMETER"},
		{"latency_priority_negative", func(r *BatchSimilarityRequest) { r.LatencyPriority = -1 }, false, "INVALID_PARAMETER"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := base()
			tc.mutate(&req)
			code, _, ok := validateBatchSimilarityRequest(req)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if code != tc.wantCode {
				t.Fatalf("code = %q, want %q", code, tc.wantCode)
			}
		})
	}
}

func TestValidateBatchSimilarityRequestRejectsNegativeTopK(t *testing.T) {
	req := BatchSimilarityRequest{
		Query:      "query",
		Candidates: []string{"a", "b"},
		TopK:       -1,
		Dimension:  defaultEmbeddingDimension,
	}

	code, message, ok := validateBatchSimilarityRequest(req)
	if ok {
		t.Fatalf("expected negative top_k to be invalid")
	}
	if code != "INVALID_INPUT" || message != "top_k cannot be negative" {
		t.Fatalf("unexpected validation error %q: %q", code, message)
	}
}

// target_layer must be validated against the layers the loaded model actually
// advertises, not a hardcoded list. For the official
// mmbert-embed-32k-2d-matryoshka (available_layers [6, 11, 16, 22]), layer 16
// must be accepted (it ships and is loadable) and layer 3 must be rejected
// (it is not on disk and previously fell back silently to the full model).
func TestValidateEmbeddingRequestTargetLayerFollowsModelManifest(t *testing.T) {
	available := []int{6, 11, 16, 22}

	if _, _, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "mmbert",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 16,
	}, available); !ok {
		t.Fatalf("expected target_layer=16 to be valid for %v", available)
	}

	code, message, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "mmbert",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 3,
	}, available)
	if ok {
		t.Fatalf("expected target_layer=3 to be rejected for %v", available)
	}
	if code != "INVALID_LAYER" {
		t.Fatalf("expected INVALID_LAYER, got %q", code)
	}
	if !strings.Contains(message, "6, 11, 16, 22") {
		t.Fatalf("error message should list the model's real layers, got %q", message)
	}
}

// When a model ships without a manifest the validator falls back to the legacy
// layer set, so target_layer=3 stays valid for that set.
func TestValidateEmbeddingRequestTargetLayerLegacyFallback(t *testing.T) {
	if _, _, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "mmbert",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 3,
	}, []int{3, 6, 11, 22}); !ok {
		t.Fatalf("expected target_layer=3 to be valid for the legacy fallback set")
	}
}

// target_layer is only meaningful for mmbert; other models must reject it.
func TestValidateEmbeddingRequestTargetLayerRejectedForNonMmbert(t *testing.T) {
	code, _, ok := validateEmbeddingRequest(EmbeddingRequest{
		Model:       "qwen3",
		Texts:       []string{"hello"},
		Dimension:   defaultEmbeddingDimension,
		TargetLayer: 6,
	}, []int{6, 11, 16, 22})
	if ok {
		t.Fatalf("expected target_layer on non-mmbert model to be rejected")
	}
	if code != "INVALID_PARAMETER" {
		t.Fatalf("expected INVALID_PARAMETER, got %q", code)
	}
}

func TestApplyEmbeddingDefaultsUsesResolvedMultimodalDefault(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{
		ModelID:   "multi-modal-embed-small",
		Default:   384,
		Supported: []int{384, 256, 128, 64, 32},
	})

	for _, req := range []*EmbeddingRequest{
		{Model: "multimodal", Texts: []string{"hello"}},
		{Images: []string{"data:image/png;base64,aGVsbG8="}},
		{Model: "multimodal", Texts: []string{"hello"}, Images: []string{"data:image/png;base64,aGVsbG8="}},
	} {
		applyEmbeddingDefaults(req)
		if req.Dimension != 384 {
			t.Fatalf("expected resolved multimodal default 384, got %d for %+v", req.Dimension, req)
		}
	}
}

func TestApplyEmbeddingDefaultsTextAutoUsesLegacyDefault(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{Default: 384, Supported: []int{384, 256, 128, 64, 32}})
	req := EmbeddingRequest{Texts: []string{"hello"}}

	applyEmbeddingDefaults(&req)

	if req.Dimension != defaultEmbeddingDimension {
		t.Fatalf("expected unresolved text request to retain default %d, got %d", defaultEmbeddingDimension, req.Dimension)
	}
}

func TestValidateEmbeddingRequestDefersUnavailableMultimodalModel(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{})
	req := EmbeddingRequest{Model: "multimodal", Images: []string{"data:image/png;base64,aGVsbG8="}, Dimension: 384}

	if code, message, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected readiness handling to remain downstream, got %q: %q", code, message)
	}
}

func TestValidateEmbeddingRequestUsesDeclaredMultimodalDimensions(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{
		ModelID:   "multi-modal-embed-small",
		Default:   384,
		Supported: []int{384, 256, 128, 64, 32},
	})

	for _, dim := range []int{384, 256, 128, 64, 32} {
		req := EmbeddingRequest{
			Model: "multimodal", Images: []string{"data:image/png;base64,aGVsbG8="}, Dimension: dim,
		}
		if code, message, ok := validateEmbeddingRequest(req, nil); !ok {
			t.Fatalf("dimension %d should be supported, got %q: %q", dim, code, message)
		}
	}

	for _, dim := range []int{100, 512, 768} {
		req := EmbeddingRequest{
			Model: "multimodal", Images: []string{"data:image/png;base64,aGVsbG8="}, Dimension: dim,
		}
		code, message, ok := validateEmbeddingRequest(req, nil)
		if ok || code != "INVALID_DIMENSION" {
			t.Fatalf("dimension %d should be rejected, got %q: %q", dim, code, message)
		}
		for _, want := range []string{"multi-modal-embed-small", "384, 256, 128, 64, 32"} {
			if !strings.Contains(message, want) {
				t.Fatalf("error %q should contain %q", message, want)
			}
		}
	}
}

func TestValidateEmbeddingRequestRejectsMixedEmbeddingSpaces(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{
		ModelID:   "multi-modal-embed-small",
		Default:   384,
		Supported: []int{384, 256, 128, 64, 32},
	})
	req := EmbeddingRequest{
		Texts: []string{"hello"}, Images: []string{"data:image/png;base64,aGVsbG8="}, Dimension: 384,
	}

	code, message, ok := validateEmbeddingRequest(req, nil)
	if ok || code != "INVALID_PARAMETER" || !strings.Contains(message, "model='multimodal'") {
		t.Fatalf("expected incompatible mixed request rejection, got %q: %q", code, message)
	}
}

func TestValidateEmbeddingRequestAcceptsSharedMultimodalSpace(t *testing.T) {
	stubMultiModalContract(t, embeddingDimensionContract{
		ModelID:   "multi-modal-embed-small",
		Default:   384,
		Supported: []int{384, 256, 128, 64, 32},
	})
	req := EmbeddingRequest{
		Model: "multimodal", Texts: []string{"hello"}, Images: []string{"data:image/png;base64,aGVsbG8="}, Dimension: 256,
	}

	if code, message, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected shared-space mixed request to pass, got %q: %q", code, message)
	}
}

func TestBuildEmbeddingResultsPassesResolvedDimensionToImages(t *testing.T) {
	var gotImageDim int
	orig := multiModalEncodeImage
	defer func() { multiModalEncodeImage = orig }()
	multiModalEncodeImage = func(_ string, targetDim int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		gotImageDim = targetDim
		return &candle_binding.MultiModalEmbeddingOutput{
			Embedding: make([]float32, targetDim),
			Modality:  "image",
		}, nil
	}

	req := EmbeddingRequest{Images: []string{"data:image/png;base64,aGVsbG8="}, Dimension: 128}
	if _, _, err := buildEmbeddingResults(req, "multi-modal-embed-small"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotImageDim != 128 {
		t.Fatalf("expected image encoder dimension 128, got %d", gotImageDim)
	}
}

func TestBuildEmbeddingResultsUsesProvidedModelID(t *testing.T) {
	orig := multiModalEncodeImage
	defer func() { multiModalEncodeImage = orig }()
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return &candle_binding.MultiModalEmbeddingOutput{
			Embedding: make([]float32, 384),
			Modality:  "image",
		}, nil
	}

	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: 384,
	}
	results, _, err := buildEmbeddingResults(req, "multi-modal-embed-small")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ModelUsed != "multi-modal-embed-small" {
		t.Fatalf("expected model_used to be the provided id, got %q", results[0].ModelUsed)
	}
}

func TestMultiModalModelIDBaseNamesConfiguredPath(t *testing.T) {
	s := &ClassificationAPIServer{
		config: &config.RouterConfig{
			InlineModels: config.InlineModels{
				EmbeddingModels: config.EmbeddingModels{MultiModalModelPath: "models/mom-embedding-multimodal"},
			},
		},
	}

	if got := s.multiModalModelID(); got != "multi-modal-embed-small" {
		t.Fatalf("expected configured path to reduce to its base name, got %q", got)
	}
}

func TestMultiModalModelIDFallsBackWhenUnconfigured(t *testing.T) {
	s := &ClassificationAPIServer{config: &config.RouterConfig{}}

	if got := s.multiModalModelID(); got != multiModalModelFallbackID {
		t.Fatalf("expected fallback id %q, got %q", multiModalModelFallbackID, got)
	}
}
