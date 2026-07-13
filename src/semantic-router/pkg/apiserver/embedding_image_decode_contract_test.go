//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestEmbeddingHandlerMapsTruncatedImageNativeRejectionTo400(t *testing.T) {
	pngURI := mustEmbeddingImageDataURI(t, "image/png")
	payload := strings.TrimPrefix(pngURI, "data:image/png;base64,")
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		t.Fatalf("decode PNG fixture: %v", err)
	}
	truncatedURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data[:len(data)-8])
	if code, message, ok := validateEmbeddingImages([]string{truncatedURI}); !ok {
		t.Fatalf("metadata inspection rejected truncated stream before Rust: %s %s", code, message)
	}

	original := multiModalEncodeImage
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return nil, fmt.Errorf("%w: truncated pixel stream", candle_binding.ErrInvalidImageInput)
	}
	t.Cleanup(func() { multiModalEncodeImage = original })
	body, err := json.Marshal(EmbeddingRequest{Images: []string{truncatedURI}})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	server.handleEmbeddings(recorder, req)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", recorder.Code, recorder.Body.String())
	}
	assertJSONErrorCode(t, recorder.Body.Bytes(), "INVALID_IMAGE")
}
