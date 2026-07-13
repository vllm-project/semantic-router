//go:build !windows && cgo

package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestSimilarityHandlersDoNotExposeNativeErrors(t *testing.T) {
	originalSimilarity := calculateEmbeddingSimilarityNative
	originalBatch := calculateSimilarityBatchNative
	t.Cleanup(func() {
		calculateEmbeddingSimilarityNative = originalSimilarity
		calculateSimilarityBatchNative = originalBatch
	})
	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}

	calculateEmbeddingSimilarityNative = func(string, string, string, int) (*candle_binding.SimilarityOutput, error) {
		return nil, errors.New("private device and model path")
	}
	assertInferenceErrorResponse(
		t,
		server.handleSimilarity,
		"/api/v1/similarity",
		`{"text1":"hello","text2":"world"}`,
		"SIMILARITY_CALCULATION_FAILED",
		"failed to calculate similarity",
	)

	calculateSimilarityBatchNative = func(string, []string, int, string, int) (*candle_binding.BatchSimilarityOutput, error) {
		return nil, errors.New("private batch runtime detail")
	}
	assertInferenceErrorResponse(
		t,
		server.handleBatchSimilarity,
		"/api/v1/similarity/batch",
		`{"query":"hello","candidates":["world"]}`,
		"BATCH_SIMILARITY_FAILED",
		"failed to calculate batch similarity",
	)

	calculateSimilarityBatchNative = func(string, []string, int, string, int) (*candle_binding.BatchSimilarityOutput, error) {
		return &candle_binding.BatchSimilarityOutput{
			Matches: []candle_binding.BatchSimilarityMatch{{Index: 99, Similarity: 0.5}},
		}, nil
	}
	assertInferenceErrorResponse(
		t,
		server.handleBatchSimilarity,
		"/api/v1/similarity/batch",
		`{"query":"hello","candidates":["world"]}`,
		"BATCH_SIMILARITY_INVALID_RESULT",
		"batch similarity returned an invalid result",
	)
}

func TestEmbeddingHandlersMapNativeTokenLimitsToStable413(t *testing.T) {
	originalText := embeddingOutputForRequest
	originalSimilarity := calculateEmbeddingSimilarityNative
	originalBatch := calculateSimilarityBatchNative
	t.Cleanup(func() {
		embeddingOutputForRequest = originalText
		calculateEmbeddingSimilarityNative = originalSimilarity
		calculateSimilarityBatchNative = originalBatch
	})
	limitError := fmt.Errorf("%w: private token counts", candle_binding.ErrEmbeddingInputTooLong)
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		return nil, limitError
	}
	calculateEmbeddingSimilarityNative = func(string, string, string, int) (*candle_binding.SimilarityOutput, error) {
		return nil, limitError
	}
	calculateSimilarityBatchNative = func(string, []string, int, string, int) (*candle_binding.BatchSimilarityOutput, error) {
		return nil, limitError
	}

	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}
	tests := []struct {
		path    string
		body    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{path: "/api/v1/embeddings", body: `{"texts":["hello"]}`, handler: server.handleEmbeddings},
		{path: "/api/v1/similarity", body: `{"text1":"hello","text2":"world"}`, handler: server.handleSimilarity},
		{path: "/api/v1/similarity/batch", body: `{"query":"hello","candidates":["world"]}`, handler: server.handleBatchSimilarity},
	}
	for _, test := range tests {
		req := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
		req.Header.Set("Content-Type", "application/json")
		recorder := httptest.NewRecorder()

		test.handler(recorder, req)

		if recorder.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("%s: expected 413, got %d: %s", test.path, recorder.Code, recorder.Body.String())
		}
		assertJSONErrorCode(t, recorder.Body.Bytes(), embeddingInputTooLargeCode)
		assertJSONErrorMessage(t, recorder.Body.Bytes(), "embedding input exceeds the selected model context")
		if strings.Contains(recorder.Body.String(), "private") || strings.Contains(recorder.Body.String(), "counts") {
			t.Fatalf("%s exposed token-limit details: %s", test.path, recorder.Body.String())
		}
	}
}

func assertInferenceErrorResponse(
	t *testing.T,
	handler func(http.ResponseWriter, *http.Request),
	path string,
	body string,
	code string,
	message string,
) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	handler(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rr.Code, rr.Body.String())
	}
	assertJSONErrorCode(t, rr.Body.Bytes(), code)
	assertJSONErrorMessage(t, rr.Body.Bytes(), message)
	if strings.Contains(rr.Body.String(), "private") || strings.Contains(rr.Body.String(), "99") {
		t.Fatalf("response exposed native detail: %s", rr.Body.String())
	}
}
