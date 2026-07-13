//go:build !windows && cgo

package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

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

func TestBuildEmbeddingResultsRejectsNativeDimensionMismatch(t *testing.T) {
	original := embeddingOutputForRequest
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		return &candle_binding.EmbeddingOutput{
			Embedding: make([]float32, 256),
			ModelType: "mmbert",
		}, nil
	}
	t.Cleanup(func() { embeddingOutputForRequest = original })

	_, _, err := buildEmbeddingResults(EmbeddingRequest{
		Texts:     []string{"hello"},
		Model:     "mmbert",
		Dimension: defaultEmbeddingDimension,
	})
	var contractErr *embeddingOutputContractError
	if !errors.As(err, &contractErr) {
		t.Fatalf("dimension mismatch error = %T %v, want embeddingOutputContractError", err, err)
	}
	if status, code, message := classifyEmbeddingError(err); status != http.StatusInternalServerError ||
		code != "EMBEDDING_GENERATION_FAILED" || message != "failed to generate embedding" {
		t.Fatalf("dimension mismatch classification = %d %q %q", status, code, message)
	}
}

func TestBuildEmbeddingResultsRejectsNativeModelMismatch(t *testing.T) {
	original := embeddingOutputForRequest
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		return &candle_binding.EmbeddingOutput{
			Embedding: make([]float32, defaultEmbeddingDimension),
			ModelType: "qwen3",
		}, nil
	}
	t.Cleanup(func() { embeddingOutputForRequest = original })

	_, _, err := buildEmbeddingResults(EmbeddingRequest{
		Texts:     []string{"hello"},
		Model:     "mmbert",
		Dimension: defaultEmbeddingDimension,
	})
	var contractErr *embeddingOutputContractError
	if !errors.As(err, &contractErr) {
		t.Fatalf("model mismatch error = %T %v, want embeddingOutputContractError", err, err)
	}
}

func TestBuildEmbeddingResultsRejectsNativeImageDimensionMismatch(t *testing.T) {
	original := multiModalEncodeImage
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return &candle_binding.MultiModalEmbeddingOutput{
			Embedding: make([]float32, 256),
			Modality:  "image",
		}, nil
	}
	t.Cleanup(func() { multiModalEncodeImage = original })

	_, _, err := buildEmbeddingResults(EmbeddingRequest{
		Images:    []string{"data:image/png;base64,placeholder"},
		Dimension: defaultImageEmbeddingDimension,
	})
	var contractErr *embeddingOutputContractError
	if !errors.As(err, &contractErr) {
		t.Fatalf("image dimension mismatch error = %T %v, want embeddingOutputContractError", err, err)
	}
}
