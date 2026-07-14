//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

func TestInferenceJSONUnicodeScannerDistinguishesLossyAndValidInputs(t *testing.T) {
	invalid := [][]byte{
		{'{', '"', 'v', '"', ':', '"', 0xff, '"', '}'},
		[]byte(`{"v":"\ud800"}`),
		[]byte(`{"v":"\udc00"}`),
	}
	for _, body := range invalid {
		if jsonunicode.Valid(body) {
			t.Fatalf("invalid Unicode accepted: %q", body)
		}
	}

	valid := [][]byte{
		[]byte(`{"v":"\ud83d\ude00"}`),
		[]byte(`{"v":"\ufffd"}`),
		[]byte(`{"v":"�"}`),
	}
	for _, body := range valid {
		if !jsonunicode.Valid(body) {
			t.Fatalf("valid Unicode rejected: %q", body)
		}
	}
}

func TestInferenceHandlersRejectLossyUnicodeBeforeUnmarshal(t *testing.T) {
	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}
	endpoints := inferenceUnicodeEndpoints(server)
	invalidValues := []struct {
		name  string
		value []byte
	}{
		{name: "invalid UTF-8 byte", value: []byte{0xff}},
		{name: "unpaired high surrogate", value: []byte(`\ud800`)},
		{name: "unpaired low surrogate", value: []byte(`\udc00`)},
	}

	for _, endpoint := range endpoints {
		for _, invalid := range invalidValues {
			t.Run(endpoint.name+"/"+invalid.name, func(t *testing.T) {
				body := endpoint.body(invalid.value)
				req := httptest.NewRequest(http.MethodPost, endpoint.path, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				rr := httptest.NewRecorder()

				endpoint.handler(rr, req)

				if rr.Code != http.StatusBadRequest {
					t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
				}
				assertJSONErrorCode(t, rr.Body.Bytes(), "INVALID_INPUT")
				assertJSONErrorMessage(t, rr.Body.Bytes(), errInvalidInferenceJSONUnicode.Error())
			})
		}
	}
}

func TestInferenceHandlersAcceptSurrogatePairsAndReplacementCharacter(t *testing.T) {
	admission := newEmbeddingProcessAdmission(1)
	release, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("saturate admission: %v", err)
	}
	defer release()
	server := &ClassificationAPIServer{embeddingAdmission: admission}
	validValues := []struct {
		name  string
		value []byte
	}{
		{name: "surrogate pair", value: []byte(`\ud83d\ude00`)},
		{name: "escaped replacement character", value: []byte(`\ufffd`)},
		{name: "literal replacement character", value: []byte("�")},
	}

	for _, endpoint := range inferenceUnicodeEndpoints(server) {
		for _, valid := range validValues {
			t.Run(endpoint.name+"/"+valid.name, func(t *testing.T) {
				req := httptest.NewRequest(
					http.MethodPost,
					endpoint.path,
					bytes.NewReader(endpoint.body(valid.value)),
				)
				req.Header.Set("Content-Type", "application/json")
				rr := httptest.NewRecorder()

				endpoint.handler(rr, req)

				if rr.Code != http.StatusServiceUnavailable {
					t.Fatalf("valid Unicode did not reach admission: %d: %s", rr.Code, rr.Body.String())
				}
				assertJSONErrorCode(t, rr.Body.Bytes(), "EMBEDDING_OVERLOADED")
			})
		}
	}
}

type inferenceUnicodeEndpoint struct {
	name    string
	path    string
	body    func([]byte) []byte
	handler func(http.ResponseWriter, *http.Request)
}

func inferenceUnicodeEndpoints(server *ClassificationAPIServer) []inferenceUnicodeEndpoint {
	wrap := func(prefix, suffix string) func([]byte) []byte {
		return func(value []byte) []byte {
			body := append([]byte(prefix), value...)
			return append(body, suffix...)
		}
	}
	return []inferenceUnicodeEndpoint{
		{name: "embeddings", path: "/api/v1/embeddings", body: wrap(`{"texts":["`, `"]}`), handler: server.handleEmbeddings},
		{name: "similarity", path: "/api/v1/similarity", body: wrap(`{"text1":"`, `","text2":"valid"}`), handler: server.handleSimilarity},
		{name: "batch similarity", path: "/api/v1/similarity/batch", body: wrap(`{"query":"`, `","candidates":["valid"]}`), handler: server.handleBatchSimilarity},
	}
}
