package embedding

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func TestOpenAICompatibleProviderRejectsOversizedSuccessResponse(t *testing.T) {
	limit := embeddingResponseByteLimit(1, 1)
	tests := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{
			name: "content length",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Length", strconv.FormatInt(limit+1, 10))
				w.WriteHeader(http.StatusOK)
			},
		},
		{
			name: "chunked",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				body := `{"data":[],"private":"LEAK-ME-API-KEY","padding":"` + strings.Repeat("x", int(limit))
				_, _ = w.Write([]byte(body))
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(tt.handler)
			defer server.Close()
			provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
				BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
			})

			_, err := provider.Embed(context.Background(), "hello")
			assertEmbeddingResponseError(t, err, embeddingResponseTooLarge)
			if strings.Contains(err.Error(), "LEAK-ME") {
				t.Fatalf("Embed() error leaked response content: %v", err)
			}
		})
	}
}

func TestOpenAICompatibleProviderResponseBudgetScalesWithBatch(t *testing.T) {
	oneLimit := embeddingResponseByteLimit(1, 1)
	twoLimit := embeddingResponseByteLimit(2, 1)
	base := paddedEmbeddingResponse(t, 1, 0)
	paddingLength := int(oneLimit) + 1 - len(base)
	onePayload := paddedEmbeddingResponse(t, 1, paddingLength)
	twoPayload := paddedEmbeddingResponse(t, 2, paddingLength)
	if int64(len(onePayload)) != oneLimit+1 || int64(len(twoPayload)) > twoLimit {
		t.Fatalf("invalid test payload sizes: one=%d/%d two=%d/%d", len(onePayload), oneLimit, len(twoPayload), twoLimit)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request embeddingsRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		payload := onePayload
		if len(request.Input) == 2 {
			payload = twoPayload
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
	})

	_, err := provider.Embed(context.Background(), "one")
	assertEmbeddingResponseError(t, err, embeddingResponseTooLarge)
	if embeddings, err := provider.EmbedBatch(context.Background(), []string{"one", "two"}); err != nil || len(embeddings) != 2 {
		t.Fatalf("EmbedBatch() = %#v, %v, want two embeddings", embeddings, err)
	}
}

func TestEmbeddingResponseByteLimitUsesUnknownBudgetAndAbsoluteCap(t *testing.T) {
	unknown := embeddingResponseByteLimit(1, 0)
	known := embeddingResponseByteLimit(1, 1)
	if unknown <= known || unknown >= maxEmbeddingResponseBytes {
		t.Fatalf("unknown dimension budget = %d, want between %d and %d", unknown, known, maxEmbeddingResponseBytes)
	}
	maxInt := int(^uint(0) >> 1)
	if got := embeddingResponseByteLimit(maxInt, maxInt); got != maxEmbeddingResponseBytes {
		t.Fatalf("overflow-safe budget = %d, want absolute cap %d", got, maxEmbeddingResponseBytes)
	}
}

func TestOpenAICompatibleProviderRejectsTrailingJSON(t *testing.T) {
	secret := "LEAK-ME-RESPONSE-CREDENTIAL"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[0.5]}]} {"secret":"`+secret+`"}`)
	}))
	defer server.Close()
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
	})

	_, err := provider.Embed(context.Background(), "hello")
	assertEmbeddingResponseError(t, err, embeddingResponseTrailingData)
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("Embed() error leaked trailing response content: %v", err)
	}
}

func TestOpenAICompatibleProviderRejectsNonJSONNumericValues(t *testing.T) {
	for _, value := range []string{"NaN", "Infinity", "-Infinity"} {
		t.Run(value, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[`+value+`]}]}`)
			}))
			defer server.Close()
			provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
				BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
			})

			_, err := provider.Embed(context.Background(), "hello")
			assertEmbeddingResponseError(t, err, embeddingResponseInvalidJSON)
		})
	}
}

func TestOpenAICompatibleProviderDoesNotExposeSuccessfulErrorBody(t *testing.T) {
	privateMarker := "LEAK-ME-UPSTREAM-BODY"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"error":{"message":"`+privateMarker+`"}}`)
	}))
	defer server.Close()
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
	})

	_, err := provider.Embed(context.Background(), "hello")
	assertEmbeddingResponseError(t, err, embeddingResponseInvalidData)
	if strings.Contains(err.Error(), privateMarker) {
		t.Fatalf("Embed() error leaked provider error body: %v", err)
	}
}

func TestOpenAICompatibleProviderRejectsFloat32Overflow(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[1e40]}]}`)
	}))
	defer server.Close()
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
	})

	_, err := provider.Embed(context.Background(), "hello")
	assertEmbeddingResponseError(t, err, embeddingResponseInvalidData)
	if !strings.Contains(err.Error(), "outside the float32 range") {
		t.Fatalf("Embed() error = %v, want float32 range error", err)
	}
}

func TestOpenAICompatibleProviderAcceptsFloat32Boundary(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeEmbeddingResponse(t, w, [][]float64{{float64(math.MaxFloat32), -float64(math.MaxFloat32)}})
	}))
	defer server.Close()
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 2,
	})

	embedding, err := provider.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if embedding[0] != math.MaxFloat32 || embedding[1] != -math.MaxFloat32 {
		t.Fatalf("Embed() = %#v, want finite float32 boundaries", embedding)
	}
}

func TestOpenAICompatibleProviderRejectsNonFiniteEmbeddingValues(t *testing.T) {
	provider := &OpenAICompatibleProvider{expectedDimension: 1}
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := provider.convertEmbedding([]float64{value}); err == nil || !strings.Contains(err.Error(), "not finite") {
			t.Fatalf("convertEmbedding(%v) error = %v, want non-finite error", value, err)
		}
	}
}

func TestOpenAICompatibleProviderRejectsEmbeddingCardinalityMismatch(t *testing.T) {
	for _, count := range []int{1, 3} {
		t.Run(strconv.Itoa(count), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				embeddings := make([][]float64, count)
				for i := range embeddings {
					embeddings[i] = []float64{float64(i)}
				}
				writeEmbeddingResponse(t, w, embeddings)
			}))
			defer server.Close()
			provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
				BaseURL: server.URL, Model: "embedding-model", ExpectedDimension: 1,
			})

			_, err := provider.EmbedBatch(context.Background(), []string{"one", "two"})
			assertEmbeddingResponseError(t, err, embeddingResponseInvalidData)
			if !strings.Contains(err.Error(), "expected 2") {
				t.Fatalf("EmbedBatch() error = %v, want cardinality error", err)
			}
		})
	}
}

func TestOpenAICompatibleProviderHonorsExplicitUnorderedIndexes(t *testing.T) {
	provider := &OpenAICompatibleProvider{expectedDimension: 1}
	data := []embeddingDatum{
		{Index: intPointer(1), Embedding: []float64{1}},
		{Index: intPointer(0), Embedding: []float64{0}},
	}

	embeddings, err := provider.parseEmbeddings(data, 2)
	if err != nil {
		t.Fatalf("parseEmbeddings() error = %v", err)
	}
	if embeddings[0][0] != 0 || embeddings[1][0] != 1 {
		t.Fatalf("parseEmbeddings() = %#v, want index-ordered embeddings", embeddings)
	}
}

func TestOpenAICompatibleProviderFallsBackOnlyForMissingIndexes(t *testing.T) {
	provider := &OpenAICompatibleProvider{expectedDimension: 1}
	data := []embeddingDatum{
		{Embedding: []float64{0}},
		{Embedding: []float64{1}},
	}

	embeddings, err := provider.parseEmbeddings(data, 2)
	if err != nil {
		t.Fatalf("parseEmbeddings() error = %v", err)
	}
	if embeddings[0][0] != 0 || embeddings[1][0] != 1 {
		t.Fatalf("parseEmbeddings() = %#v, want response-order fallback", embeddings)
	}
}

func TestEmbeddingProviderErrorBodyRemainsBoundedAndPrivate(t *testing.T) {
	body := &countingReadCloser{Reader: strings.NewReader(strings.Repeat("LEAK-ME-API-KEY", maxErrorBodyDrainBytes))}
	err := responseError(&http.Response{StatusCode: http.StatusBadGateway, Body: body})
	if body.bytesRead != maxErrorBodyDrainBytes {
		t.Fatalf("responseError() read %d bytes, want %d", body.bytesRead, maxErrorBodyDrainBytes)
	}
	if body.sawEOF {
		t.Fatal("responseError() read past the bounded error-body budget")
	}
	if strings.Contains(err.Error(), "LEAK-ME") {
		t.Fatalf("responseError() leaked provider body: %v", err)
	}
}

func TestEmbeddingProviderErrorBodyDrainsSmallBodyToEOF(t *testing.T) {
	payload := "private upstream detail"
	body := &countingReadCloser{Reader: strings.NewReader(payload)}
	_ = responseError(&http.Response{StatusCode: http.StatusBadRequest, Body: body})
	if body.bytesRead != len(payload) || !body.sawEOF {
		t.Fatalf("responseError() read=%d eof=%v, want read=%d eof=true", body.bytesRead, body.sawEOF, len(payload))
	}
}
