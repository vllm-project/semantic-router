//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestValidateEmbeddingRequestInputCountBoundary(t *testing.T) {
	texts := repeatedEmbeddingTexts(maxEmbeddingInputs, "x")
	if code, message, ok := validateEmbeddingRequest(EmbeddingRequest{
		Texts:     texts,
		Dimension: defaultEmbeddingDimension,
	}, nil); !ok {
		t.Fatalf("exact input limit rejected: %q %q", code, message)
	}

	texts = append(texts, "overflow")
	code, _, ok := validateEmbeddingRequest(EmbeddingRequest{
		Texts:     texts,
		Dimension: defaultEmbeddingDimension,
	}, nil)
	if ok || code != embeddingInputTooLargeCode {
		t.Fatalf("expected %s above input limit, got ok=%v code=%q", embeddingInputTooLargeCode, ok, code)
	}
}

func TestValidateEmbeddingTextsSizeBoundaries(t *testing.T) {
	exact := strings.Repeat("x", maxEmbeddingTextBytes)
	if code, message, ok := validateEmbeddingTexts([]string{exact}); !ok {
		t.Fatalf("exact per-text limit rejected: %q %q", code, message)
	}

	tests := []struct {
		name  string
		texts []string
	}{
		{name: "per text bytes plus one", texts: []string{exact + "x"}},
		{name: "total bytes plus one", texts: append(repeatedEmbeddingTexts(8, exact), "x")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, _, ok := validateEmbeddingTexts(tt.texts)
			if ok || code != embeddingInputTooLargeCode {
				t.Fatalf("expected %s, got ok=%v code=%q", embeddingInputTooLargeCode, ok, code)
			}
		})
	}

	if code, message, ok := validateEmbeddingTexts(repeatedEmbeddingTexts(8, exact)); !ok {
		t.Fatalf("exact aggregate text limit rejected: %q %q", code, message)
	}
}

func TestValidateEmbeddingTextsRejectsInvalidContent(t *testing.T) {
	tests := []struct {
		name string
		text string
	}{
		{name: "empty", text: ""},
		{name: "whitespace", text: " \t\n"},
		{name: "NUL", text: "safe\x00suffix"},
		{name: "invalid UTF-8", text: string([]byte{0xff})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, _, ok := validateEmbeddingTexts([]string{tt.text})
			if ok || code != "INVALID_INPUT" {
				t.Fatalf("expected INVALID_INPUT, got ok=%v code=%q", ok, code)
			}
		})
	}
	if code, message, ok := validateEmbeddingTexts([]string{"valid\ufffdtext"}); !ok {
		t.Fatalf("legal replacement character rejected: %q %q", code, message)
	}
}

func TestValidateBatchSimilarityCandidateAndTextBudgets(t *testing.T) {
	req := BatchSimilarityRequest{
		Query:      "query",
		Candidates: repeatedEmbeddingTexts(maxBatchSimilarityCandidates, "candidate"),
		TopK:       1,
		Dimension:  defaultEmbeddingDimension,
	}
	if code, message, ok := validateBatchSimilarityRequest(req); !ok {
		t.Fatalf("exact candidate limit rejected: %q %q", code, message)
	}

	req.Candidates = append(req.Candidates, "overflow")
	code, _, ok := validateBatchSimilarityRequest(req)
	if ok || code != embeddingInputTooLargeCode {
		t.Fatalf("expected candidate limit rejection, got ok=%v code=%q", ok, code)
	}

	req.Candidates = []string{strings.Repeat("x", maxEmbeddingTextBytes+1)}
	code, _, ok = validateBatchSimilarityRequest(req)
	if ok || code != embeddingInputTooLargeCode {
		t.Fatalf("expected text size rejection, got ok=%v code=%q", ok, code)
	}
}

func TestValidateSimilarityTextsUsesSharedContract(t *testing.T) {
	tests := []SimilarityRequest{
		{Text1: " ", Text2: "valid"},
		{Text1: "valid", Text2: "bad\x00text"},
	}
	for _, req := range tests {
		code, _, ok := validateSimilarityTexts(req)
		if ok || code != "INVALID_INPUT" {
			t.Fatalf("expected INVALID_INPUT, got ok=%v code=%q", ok, code)
		}
	}

	code, _, ok := validateSimilarityTexts(SimilarityRequest{
		Text1: strings.Repeat("x", maxEmbeddingTextBytes+1),
		Text2: "valid",
	})
	if ok || code != embeddingInputTooLargeCode {
		t.Fatalf("expected text size rejection, got ok=%v code=%q", ok, code)
	}
}

func TestEmbeddingHandlersEnforceInputLimitsBeforeNativeAdmission(t *testing.T) {
	embeddingsBody, err := json.Marshal(map[string]interface{}{
		"texts": repeatedEmbeddingTexts(maxEmbeddingInputs+1, "text"),
	})
	if err != nil {
		t.Fatalf("marshal embeddings request: %v", err)
	}
	batchBody, err := json.Marshal(map[string]interface{}{
		"query":      "query",
		"candidates": repeatedEmbeddingTexts(maxBatchSimilarityCandidates+1, "candidate"),
	})
	if err != nil {
		t.Fatalf("marshal batch request: %v", err)
	}
	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}
	tests := []struct {
		name    string
		path    string
		body    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{name: "embeddings input count", path: "/api/v1/embeddings", body: string(embeddingsBody), handler: server.handleEmbeddings},
		{name: "similarity text size", path: "/api/v1/similarity", body: `{"text1":"` + strings.Repeat("x", maxEmbeddingTextBytes+1) + `","text2":"valid"}`, handler: server.handleSimilarity},
		{name: "batch candidate count", path: "/api/v1/similarity/batch", body: string(batchBody), handler: server.handleBatchSimilarity},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			tt.handler(rr, req)

			if rr.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("expected 413, got %d: %s", rr.Code, rr.Body.String())
			}
			assertJSONErrorCode(t, rr.Body.Bytes(), embeddingInputTooLargeCode)
		})
	}
}

func repeatedEmbeddingTexts(count int, text string) []string {
	texts := make([]string, count)
	for i := range texts {
		texts[i] = text
	}
	return texts
}
