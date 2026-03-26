//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func stubKnowledgeBaseMapEmbeddings(t *testing.T) {
	t.Helper()

	restore := knowledgeBaseMapEmbeddingFunc
	knowledgeBaseMapEmbeddingFunc = func(text string, modelType string, _ int) (*candle_binding.EmbeddingOutput, error) {
		text = strings.TrimSpace(text)
		length := float32(len(text))
		vector := []float32{
			length,
			float32(len(strings.Fields(text))) + 1,
			float32((len(text) % 7) + 1),
		}
		return &candle_binding.EmbeddingOutput{
			Embedding: vector,
			ModelType: modelType,
		}, nil
	}
	t.Cleanup(func() {
		knowledgeBaseMapEmbeddingFunc = restore
	})
}

func TestHandleKnowledgeBaseMapEndpoints(t *testing.T) {
	apiServer, _, _ := newTestKnowledgeBaseAPIServer(t)
	stubKnowledgeBaseMapEmbeddings(t)

	metadataReq := httptest.NewRequest(http.MethodGet, "/config/kbs/privacy_kb/map/metadata", nil)
	metadataReq.SetPathValue("name", "privacy_kb")
	metadataRR := httptest.NewRecorder()
	apiServer.handleGetKnowledgeBaseMapMetadata(metadataRR, metadataReq)

	if metadataRR.Code != http.StatusOK {
		t.Fatalf("expected metadata 200, got %d: %s", metadataRR.Code, metadataRR.Body.String())
	}

	var metadata knowledgeBaseMapMetadataResponse
	if err := json.Unmarshal(metadataRR.Body.Bytes(), &metadata); err != nil {
		t.Fatalf("json.Unmarshal metadata: %v", err)
	}
	if metadata.Name != "privacy_kb" {
		t.Fatalf("expected privacy_kb metadata, got %+v", metadata)
	}
	if metadata.Projection != "umap_2d" {
		t.Fatalf("expected umap_2d projection, got %+v", metadata)
	}
	if metadata.PointCount <= 0 || metadata.LabelCount <= 0 {
		t.Fatalf("expected populated metadata, got %+v", metadata)
	}
	if len(metadata.Groups) == 0 {
		t.Fatalf("expected kb groups in metadata, got %+v", metadata)
	}

	dataReq := httptest.NewRequest(http.MethodGet, "/config/kbs/privacy_kb/map/data.ndjson", nil)
	dataReq.SetPathValue("name", "privacy_kb")
	dataRR := httptest.NewRecorder()
	apiServer.handleGetKnowledgeBaseMapData(dataRR, dataReq)
	if dataRR.Code != http.StatusOK {
		t.Fatalf("expected data 200, got %d: %s", dataRR.Code, dataRR.Body.String())
	}
	if got := dataRR.Header().Get("Content-Type"); got != "application/x-ndjson" {
		t.Fatalf("expected ndjson content type, got %q", got)
	}
	if !strings.Contains(dataRR.Body.String(), "proprietary_code") && len(strings.TrimSpace(dataRR.Body.String())) == 0 {
		t.Fatalf("expected non-empty ndjson body, got %q", dataRR.Body.String())
	}
	firstLine := strings.TrimSpace(strings.Split(dataRR.Body.String(), "\n")[0])
	var point kbRawPoint
	if err := json.Unmarshal([]byte(firstLine), &point); err != nil {
		t.Fatalf("json.Unmarshal point: %v", err)
	}
	if point.LabelName == "" || point.Text == "" || len(point.Vector) == 0 {
		t.Fatalf("expected raw kb point payload, got %+v", point)
	}
}

func TestHandleKnowledgeBaseMapMissingKnowledgeBase(t *testing.T) {
	apiServer, _, _ := newTestKnowledgeBaseAPIServer(t)
	stubKnowledgeBaseMapEmbeddings(t)

	req := httptest.NewRequest(http.MethodGet, "/config/kbs/missing/map/metadata", nil)
	req.SetPathValue("name", "missing")
	rr := httptest.NewRecorder()
	apiServer.handleGetKnowledgeBaseMapMetadata(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", rr.Code, rr.Body.String())
	}
}
