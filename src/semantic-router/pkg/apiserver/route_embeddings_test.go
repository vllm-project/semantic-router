//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
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
		Dimension: defaultImageEmbeddingDimension,
	}

	if _, _, ok := validateEmbeddingRequest(req, nil); !ok {
		t.Fatalf("expected image-only request to be valid")
	}
}

func TestValidateEmbeddingRequestRejectsUnsafeImage(t *testing.T) {
	req := EmbeddingRequest{
		Images:    []string{"https://example.com/cat.png"},
		Dimension: defaultImageEmbeddingDimension,
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
		Dimension: defaultImageEmbeddingDimension,
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
		Dimension: defaultImageEmbeddingDimension,
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
				Dimension: defaultImageEmbeddingDimension,
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
				Dimension: defaultImageEmbeddingDimension,
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

func TestValidateEmbeddingRequestRejectsTooManyImages(t *testing.T) {
	images := make([]string, maxImagesPerRequest+1)
	for i := range images {
		images[i] = mustEmbeddingImageDataURI(t, "image/png")
	}
	req := EmbeddingRequest{Images: images, Dimension: defaultImageEmbeddingDimension}

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

	code, message, ok := validateBatchSimilarityRequest(req, nil)
	if ok {
		t.Fatalf("expected negative top_k to be invalid")
	}
	if code != "INVALID_INPUT" || message != "top_k cannot be negative" {
		t.Fatalf("unexpected validation error %q: %q", code, message)
	}
}
