//go:build !windows && cgo

package apiserver

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

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
	_, _, err := buildEmbeddingResults(req)
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

func TestClassifyEmbeddingErrorMapsModelNotReadyTo503(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		wantText   string
		wantCode   string
		wantStatus int
	}{
		{
			name:       "services sentinel",
			err:        services.ErrModelNotReady,
			wantText:   services.ErrModelNotReady.Error(),
			wantCode:   "EMBEDDING_NOT_READY",
			wantStatus: http.StatusServiceUnavailable,
		},
		{
			name:       "candle sentinel",
			err:        candle_binding.ErrEmbeddingModelNotReady,
			wantText:   candle_binding.ErrEmbeddingModelNotReady.Error(),
			wantCode:   "EMBEDDING_NOT_READY",
			wantStatus: http.StatusServiceUnavailable,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			status, code, message := classifyEmbeddingError(tc.err)
			if status != tc.wantStatus || code != tc.wantCode {
				t.Fatalf("expected %d %q, got %d %q", tc.wantStatus, tc.wantCode, status, code)
			}
			if !strings.Contains(message, tc.wantText) {
				t.Fatalf("expected message to include %q, got %q", tc.wantText, message)
			}
		})
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

func TestEmbeddingEndpointsReturn503WhenNotReady(t *testing.T) {
	if candle_binding.IsEmbeddingReady() {
		t.Skip("test requires embedding models to be uninitialized")
	}

	s := &ClassificationAPIServer{}

	tests := []struct {
		name    string
		path    string
		body    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{
			"embeddings",
			"/api/v1/embeddings",
			`{"texts":["hi"]}`,
			s.handleEmbeddings,
		},
		{
			"similarity",
			"/api/v1/similarity",
			`{"text1":"hello","text2":"world"}`,
			s.handleSimilarity,
		},
		{
			"batch similarity",
			"/api/v1/similarity/batch",
			`{"query":"hello","candidates":["world"]}`,
			s.handleBatchSimilarity,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()

			tc.handler(rr, req)

			if rr.Code != http.StatusServiceUnavailable {
				t.Fatalf("expected 503, got %d: %s", rr.Code, rr.Body.String())
			}

			if !strings.Contains(rr.Body.String(), "EMBEDDING_NOT_READY") {
				t.Fatalf("expected EMBEDDING_NOT_READY, got: %s", rr.Body.String())
			}
		})
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

// When text models are ready, text embedding requests must not return 503.
func TestEmbeddingHandlerSucceedsWhenTextModelsReady(t *testing.T) {
	orig := candle_binding.IsEmbeddingReady()
	candle_binding.SetEmbeddingReady(true)
	defer candle_binding.SetEmbeddingReady(orig)

	s := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings",
		strings.NewReader(`{"texts":["hello"]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleEmbeddings(rr, req)

	if rr.Code == http.StatusServiceUnavailable {
		t.Fatalf("expected non-503 when text models are ready, got 503: %s", rr.Body.String())
	}
}

// When multimodal is not ready, image embedding must return 503 via the error path.
func TestEmbeddingHandlerReturns503WhenMultimodalNotReady(t *testing.T) {
	origText := candle_binding.IsEmbeddingReady()
	origImage := candle_binding.IsMultiModalReady()
	candle_binding.SetEmbeddingReady(true)
	candle_binding.SetMultiModalReady(false)
	defer candle_binding.SetEmbeddingReady(origText)
	defer candle_binding.SetMultiModalReady(origImage)

	s := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings",
		strings.NewReader(`{"images":["data:image/png;base64,aGVsbG8="]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleEmbeddings(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when multimodal not ready, got %d: %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "EMBEDDING_NOT_READY") {
		t.Fatalf("expected EMBEDDING_NOT_READY, got: %s", rr.Body.String())
	}
}

// When both text and multimodal are ready, image embedding must not return 503.
func TestEmbeddingHandlerSucceedsWhenMultimodalReady(t *testing.T) {
	origText := candle_binding.IsEmbeddingReady()
	origImage := candle_binding.IsMultiModalReady()
	candle_binding.SetEmbeddingReady(true)
	candle_binding.SetMultiModalReady(true)
	defer candle_binding.SetEmbeddingReady(origText)
	defer candle_binding.SetMultiModalReady(origImage)

	s := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings",
		strings.NewReader(`{"images":["data:image/png;base64,aGVsbG8="]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleEmbeddings(rr, req)

	if rr.Code == http.StatusServiceUnavailable {
		t.Fatalf("expected non-503 when multimodal is ready, got 503: %s", rr.Body.String())
	}
}

// When only multimodal is ready (text models absent), text embedding must
// return 503 — a multimodal-only deployment must not serve text requests.
func TestEmbeddingHandlerReturns503WhenTextModelsNotReady(t *testing.T) {
	origText := candle_binding.IsEmbeddingReady()
	origImage := candle_binding.IsMultiModalReady()
	candle_binding.SetEmbeddingReady(false)
	candle_binding.SetMultiModalReady(true)
	defer candle_binding.SetEmbeddingReady(origText)
	defer candle_binding.SetMultiModalReady(origImage)

	s := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings",
		strings.NewReader(`{"texts":["hello"]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleEmbeddings(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when text models not ready, got %d: %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "EMBEDDING_NOT_READY") {
		t.Fatalf("expected EMBEDDING_NOT_READY, got: %s", rr.Body.String())
	}
}

// When only multimodal is ready, image embedding must succeed — a
// multimodal-only deployment must serve image requests.
func TestEmbeddingHandlerSucceedsForImagesInMultimodalOnlyDeployment(t *testing.T) {
	origText := candle_binding.IsEmbeddingReady()
	origImage := candle_binding.IsMultiModalReady()
	candle_binding.SetEmbeddingReady(false)
	candle_binding.SetMultiModalReady(true)
	defer candle_binding.SetEmbeddingReady(origText)
	defer candle_binding.SetMultiModalReady(origImage)

	s := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings",
		strings.NewReader(`{"images":["data:image/png;base64,aGVsbG8="]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleEmbeddings(rr, req)

	if rr.Code == http.StatusServiceUnavailable {
		t.Fatalf("expected non-503 for images in multimodal-only deployment, got 503: %s", rr.Body.String())
	}
}

// When only text models are ready, a mixed text+image request must return 503
// because the multimodal gate blocks the image portion.
func TestEmbeddingHandlerReturns503ForMixedRequestWhenMultimodalNotReady(t *testing.T) {
	origText := candle_binding.IsEmbeddingReady()
	origImage := candle_binding.IsMultiModalReady()
	candle_binding.SetEmbeddingReady(true)
	candle_binding.SetMultiModalReady(false)
	defer candle_binding.SetEmbeddingReady(origText)
	defer candle_binding.SetMultiModalReady(origImage)

	s := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings",
		strings.NewReader(`{"texts":["hello"],"images":["data:image/png;base64,aGVsbG8="]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleEmbeddings(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 for mixed request when multimodal not ready, got %d: %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "EMBEDDING_NOT_READY") {
		t.Fatalf("expected EMBEDDING_NOT_READY, got: %s", rr.Body.String())
	}
}

// When only multimodal is ready, a mixed text+image request must return 503
// because the text gate blocks the text portion.
func TestEmbeddingHandlerReturns503ForMixedRequestWhenTextModelsNotReady(t *testing.T) {
	origText := candle_binding.IsEmbeddingReady()
	origImage := candle_binding.IsMultiModalReady()
	candle_binding.SetEmbeddingReady(false)
	candle_binding.SetMultiModalReady(true)
	defer candle_binding.SetEmbeddingReady(origText)
	defer candle_binding.SetMultiModalReady(origImage)

	s := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings",
		strings.NewReader(`{"texts":["hello"],"images":["data:image/png;base64,aGVsbG8="]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	s.handleEmbeddings(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 for mixed request when text models not ready, got %d: %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "EMBEDDING_NOT_READY") {
		t.Fatalf("expected EMBEDDING_NOT_READY, got: %s", rr.Body.String())
	}
}
