//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"net/http"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func mustEmbeddingImageDataURI(t *testing.T, mime string) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.RGBA{R: 32, G: 128, B: 255, A: 255})
	var encoded bytes.Buffer
	var err error
	switch mime {
	case "image/png":
		err = png.Encode(&encoded, img)
	case "image/jpeg":
		err = jpeg.Encode(&encoded, img, &jpeg.Options{Quality: 90})
	default:
		t.Fatalf("unsupported test MIME %q", mime)
	}
	if err != nil {
		t.Fatalf("encode %s fixture: %v", mime, err)
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
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
		Images:    []string{mustEmbeddingImageDataURI(t, "image/png")},
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
	pngURI := mustEmbeddingImageDataURI(t, "image/png")
	req := EmbeddingRequest{
		Images:    []string{strings.Replace(pngURI, "data:image/png;base64,", "DATA:IMAGE/PNG;BASE64,", 1)},
		Dimension: defaultEmbeddingDimension,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected uppercase-scheme data URI to be accepted")
	}
}

func TestValidateEmbeddingRequestRejectsDecodedBytesThatAreNotDeclaredImage(t *testing.T) {
	pngURI := mustEmbeddingImageDataURI(t, "image/png")
	tests := []struct {
		name  string
		image string
	}{
		{name: "valid base64 non-image", image: "data:image/png;base64,aGVsbG8="},
		{name: "MIME format mismatch", image: strings.Replace(pngURI, "image/png", "image/jpeg", 1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, message, ok := validateEmbeddingRequest(EmbeddingRequest{
				Images:    []string{tt.image},
				Dimension: defaultEmbeddingDimension,
			}, nil)
			if ok || code != "INVALID_IMAGE" || !strings.Contains(message, "decodable JPEG or PNG") {
				t.Fatalf("unexpected validation result ok=%v code=%q message=%q", ok, code, message)
			}
		})
	}
}

func TestValidateEmbeddingRequestRejectsImageFormatsUnsupportedByEmbeddingDecoder(t *testing.T) {
	for _, mime := range []string{"image/gif", "image/webp"} {
		t.Run(mime, func(t *testing.T) {
			req := EmbeddingRequest{
				Images:    []string{"data:" + mime + ";base64,AAAA"},
				Dimension: defaultEmbeddingDimension,
			}

			code, message, ok := validateEmbeddingRequest(req, nil)
			if ok {
				t.Fatalf("expected %s to be rejected before reaching the JPEG/PNG-only decoder", mime)
			}
			if code != "INVALID_IMAGE" || !strings.Contains(message, "JPEG or PNG") {
				t.Fatalf("unexpected validation error %q: %q", code, message)
			}
		})
	}
}

func TestBuildEmbeddingResultsWrapsImageEncodeFailure(t *testing.T) {
	// A validated safe data URI whose bytes are not a decodable image fails at the
	// FFI; buildEmbeddingResults must tag it as an imageEncodeError so the handler
	// maps it to 400 instead of 500.
	orig := multiModalEncodeImage
	defer func() { multiModalEncodeImage = orig }()
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return nil, fmt.Errorf("decoder rejected payload: %w", candle_binding.ErrInvalidImageInput)
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

func TestBuildEmbeddingResultsKeepsInternalImageEncodeFailureAs500(t *testing.T) {
	orig := multiModalEncodeImage
	defer func() { multiModalEncodeImage = orig }()
	internalErr := errors.New("model is not initialized: private detail")
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return nil, internalErr
	}

	req := EmbeddingRequest{
		Images:    []string{"data:image/png;base64,aGVsbG8="},
		Dimension: defaultEmbeddingDimension,
	}
	_, _, err := buildEmbeddingResults(req)
	if err == nil {
		t.Fatal("expected an internal image encode error")
	}
	var imgErr *imageEncodeError
	if errors.As(err, &imgErr) {
		t.Fatalf("internal error must not be wrapped as imageEncodeError: %v", err)
	}
	if !errors.Is(err, internalErr) {
		t.Fatalf("expected internal cause to remain available to server code: %v", err)
	}

	status, code, message := classifyEmbeddingError(err)
	if status != http.StatusInternalServerError || code != "EMBEDDING_GENERATION_FAILED" {
		t.Fatalf("expected 500 EMBEDDING_GENERATION_FAILED, got %d %q", status, code)
	}
	if message != "failed to generate embedding" || strings.Contains(message, "private detail") {
		t.Fatalf("500 response exposed internal error detail: %q", message)
	}
}

func TestClassifyEmbeddingErrorMapsImageEncodeFailureTo400(t *testing.T) {
	status, code, _ := classifyEmbeddingError(&imageEncodeError{index: 2, err: candle_binding.ErrInvalidImageInput})
	if status != http.StatusBadRequest || code != "INVALID_IMAGE" {
		t.Fatalf("expected 400 INVALID_IMAGE, got %d %q", status, code)
	}
}

func TestClassifyEmbeddingErrorMapsInternalFailureTo500(t *testing.T) {
	status, code, message := classifyEmbeddingError(errors.New("model not loaded"))
	if status != http.StatusInternalServerError || code != "EMBEDDING_GENERATION_FAILED" {
		t.Fatalf("expected 500 EMBEDDING_GENERATION_FAILED, got %d %q", status, code)
	}
	if message != "failed to generate embedding" {
		t.Fatalf("expected fixed client-safe 500 message, got %q", message)
	}
}

func TestValidateEmbeddingRequestRejectsTooManyImages(t *testing.T) {
	images := make([]string, maxImagesPerRequest+1)
	for i := range images {
		images[i] = mustEmbeddingImageDataURI(t, "image/png")
	}
	req := EmbeddingRequest{Images: images, Dimension: defaultEmbeddingDimension}

	code, _, ok := validateEmbeddingRequest(req, nil)
	if ok {
		t.Fatalf("expected more than %d images to be rejected", maxImagesPerRequest)
	}
	if code != embeddingInputTooLargeCode {
		t.Fatalf("unexpected validation error code %q", code)
	}
}

func TestAverageEmbeddingProcessingTimeUsesInputCount(t *testing.T) {
	req := EmbeddingRequest{
		Texts:  []string{"first", "second"},
		Images: []string{mustEmbeddingImageDataURI(t, "image/png")},
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
