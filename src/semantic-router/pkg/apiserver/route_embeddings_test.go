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

// stubMultiModalDim overrides the native-dimension getter for a test so the
// image over-dimension contract can be exercised without a loaded model.
func stubMultiModalDim(t *testing.T, dim int) {
	t.Helper()
	orig := multiModalEmbeddingDim
	t.Cleanup(func() { multiModalEmbeddingDim = orig })
	multiModalEmbeddingDim = func() int { return dim }
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
	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: defaultEmbeddingDimension,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected image-only request to be valid")
	}
}

func TestValidateEmbeddingRequestRejectsUnsafeImage(t *testing.T) {
	req := EmbeddingRequest{
		Images:    []string{"https://example.com/cat.png"},
		Dimension: defaultEmbeddingDimension,
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
	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,!!!!"},
		Dimension: defaultEmbeddingDimension,
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
	// "DATA:IMAGE/PNG;BASE64,..." passes the safety gate; it must also pass
	// decode-validation so it is not accepted here only to 500 at the FFI, whose
	// marker scan is case-sensitive (CanonicalDataURL normalizes it downstream).
	req := EmbeddingRequest{
		Images:    []string{"DATA:IMAGE/PNG;BASE64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: defaultEmbeddingDimension,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected uppercase-scheme data URI to be accepted")
	}
}

func TestBuildEmbeddingResultsWrapsImageEncodeFailure(t *testing.T) {
	// A validated safe data URI whose bytes are not a decodable image fails at the
	// FFI; buildEmbeddingResults must tag it as an imageEncodeError so the handler
	// maps it to 400 instead of 500.
	orig := multiModalEncodeImage
	defer func() { multiModalEncodeImage = orig }()
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return nil, errors.New("failed to decode image")
	}

	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: defaultEmbeddingDimension,
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
	images := make([]string, maxImagesPerRequest+1)
	for i := range images {
		images[i] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
	}
	req := EmbeddingRequest{Images: images, Dimension: defaultEmbeddingDimension}

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

func TestIsValidDimensionAcceptsNativeMultimodalDimension(t *testing.T) {
	// 384 is the multimodal model's native dimension and must be accepted so a
	// caller can request the image model's full-width vector explicitly.
	if !isValidDimension(384) {
		t.Fatalf("expected dimension 384 (multimodal native) to be valid")
	}
}

func TestApplyEmbeddingDefaultsImageOnlyUsesNativeDimension(t *testing.T) {
	// An image-only request has no text side to honor the text default, so it
	// defaults to the multimodal native dimension (the ceiling MRL can produce).
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{Images: []string{"data:image/png;base64,aGVsbG8="}}

	applyEmbeddingDefaults(&req)

	if req.Dimension != 384 {
		t.Fatalf("expected image-only request to default to native dimension 384, got %d", req.Dimension)
	}
}

func TestApplyEmbeddingDefaultsMixedUsesTextDefault(t *testing.T) {
	// A mixed text+image request defaults req.Dimension to the text default
	// (768). req.Dimension controls the text side only; the image side always
	// encodes at its native dimension regardless. This preserves text recall
	// rather than silently downgrading text to 384 for a false-consistency win
	// (cross-encoder vectors do not align across modalities regardless of dim).
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Texts:  []string{"hello"},
		Images: []string{"data:image/png;base64,aGVsbG8="},
	}

	applyEmbeddingDefaults(&req)

	if req.Dimension != defaultEmbeddingDimension {
		t.Fatalf("expected mixed request to default to text default %d, got %d", defaultEmbeddingDimension, req.Dimension)
	}
}

func TestApplyEmbeddingDefaultsTextsUseTextDefault(t *testing.T) {
	req := EmbeddingRequest{Texts: []string{"hello"}}

	applyEmbeddingDefaults(&req)

	if req.Dimension != defaultEmbeddingDimension {
		t.Fatalf("expected text-only request to default to %d, got %d", defaultEmbeddingDimension, req.Dimension)
	}
}

func TestApplyEmbeddingDefaultsImageOnlyFallsBackWhenModelUnloaded(t *testing.T) {
	// When no multimodal model is loaded the getter reports <= 0; defaulting an
	// image-only request falls back to the text default and the encode fails
	// downstream rather than this layer guessing a native dimension.
	stubMultiModalDim(t, -1)
	req := EmbeddingRequest{Images: []string{"data:image/png;base64,aGVsbG8="}}

	applyEmbeddingDefaults(&req)

	if req.Dimension != defaultEmbeddingDimension {
		t.Fatalf("expected fallback to %d when multimodal model unloaded, got %d", defaultEmbeddingDimension, req.Dimension)
	}
}

func TestValidateEmbeddingRequestRejectsImageOnlyDimensionAboveNative(t *testing.T) {
	// Image-only: req.Dimension is the image target and must be <= native.
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 768,
	}

	code, message, ok := validateEmbeddingRequest(req, nil)
	if ok {
		t.Fatalf("expected dimension above the multimodal native dimension to be rejected for image-only")
	}
	if code != "INVALID_DIMENSION" {
		t.Fatalf("unexpected error code %q: %q", code, message)
	}
	if !strings.Contains(message, "384") {
		t.Fatalf("error should mention the native dimension, got %q", message)
	}
}

func TestValidateEmbeddingRequestAllowsAboveNativeDimensionForMixed(t *testing.T) {
	// Mixed: req.Dimension controls the text side, and the image side always
	// encodes at its native dimension regardless. A text-legal req.Dimension
	// like 768 must be accepted even though it exceeds the image ceiling.
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Texts:     []string{"hello"},
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 768,
	}

	if _, message, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected mixed request with text-legal dimension 768 to be accepted; got %q", message)
	}
}

func TestValidateEmbeddingRequestAcceptsImageNativeDimension(t *testing.T) {
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 384,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected native dimension 384 to be accepted for image inputs")
	}
}

func TestValidateEmbeddingRequestRejectsTargetLayerWithImages(t *testing.T) {
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Images:      []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension:   384,
		TargetLayer: 6,
	}

	code, message, ok := validateEmbeddingRequest(req, []int{6, 11, 16, 22})
	if ok {
		t.Fatalf("expected target_layer with image inputs to be rejected")
	}
	if code != "INVALID_PARAMETER" || !strings.Contains(message, "image inputs") {
		t.Fatalf("unexpected error %q: %q", code, message)
	}
}

func TestValidateEmbeddingRequestRejectsTextModelForImageOnly(t *testing.T) {
	// An image-only request naming a text model would have that model silently
	// ignored; reject so the contract is explicit.
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Model:     "qwen3",
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 384,
	}

	code, _, ok := validateEmbeddingRequest(req, nil)
	if ok {
		t.Fatalf("expected image-only request with a text model to be rejected")
	}
	if code != "INVALID_PARAMETER" {
		t.Fatalf("expected INVALID_PARAMETER, got %q", code)
	}
}

func TestValidateEmbeddingRequestAllowsMultimodalModelForImageOnly(t *testing.T) {
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Model:     "multimodal",
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 384,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected an explicit multimodal model selector to be accepted for image inputs")
	}
}

func TestValidateEmbeddingRequestAllowsTextModelForMixed(t *testing.T) {
	// A mixed request still honors req.Model for the text side, so a text model
	// is not "ignored" and must be accepted.
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Model:     "qwen3",
		Texts:     []string{"hello"},
		Images:    []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"},
		Dimension: 384,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected mixed text+image request with a text model to be accepted")
	}
}

func TestImageTargetDimensionMixedUsesNative(t *testing.T) {
	// Mixed: req.Dimension is the text-side target, so the image side must be
	// resolved to the multimodal native dimension regardless of req.Dimension.
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Texts:     []string{"hello"},
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: 768,
	}
	if got := imageTargetDimension(req); got != 384 {
		t.Fatalf("expected mixed request image encode dim to be 384 (native), got %d", got)
	}
}

func TestImageTargetDimensionImageOnlyPassesThrough(t *testing.T) {
	// Image-only: req.Dimension IS the image target (already validated <= native).
	stubMultiModalDim(t, 384)
	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: 128,
	}
	if got := imageTargetDimension(req); got != 128 {
		t.Fatalf("expected image-only request image encode dim to be 128, got %d", got)
	}
}

func TestImageTargetDimensionMixedFallsBackWhenModelUnloaded(t *testing.T) {
	// Mixed with no multimodal model loaded: helper returns req.Dimension so the
	// downstream encode surfaces "model not loaded" rather than the helper
	// silently choosing a value.
	stubMultiModalDim(t, -1)
	req := EmbeddingRequest{
		Texts:     []string{"hello"},
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: 768,
	}
	if got := imageTargetDimension(req); got != 768 {
		t.Fatalf("expected fallback to req.Dimension 768 when multimodal unloaded, got %d", got)
	}
}

func TestBuildEmbeddingResultsImageOnlyHonorsRequestDimension(t *testing.T) {
	// Image-only: req.Dimension is the image target (validated <= native) and
	// must be passed through to the encoder verbatim.
	stubMultiModalDim(t, 384)

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

	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: 128,
	}
	if _, _, err := buildEmbeddingResults(req, "multi-modal-embed-small"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotImageDim != 128 {
		t.Fatalf("expected image encode to receive image-only request dimension 128, got %d", gotImageDim)
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
				EmbeddingModels: config.EmbeddingModels{MultiModalModelPath: "models/multi-modal-embed-small"},
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
