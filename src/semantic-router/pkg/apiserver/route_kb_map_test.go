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
	if metadata.PointCount <= 0 || metadata.LabelCount <= 0 {
		t.Fatalf("expected populated metadata, got %+v", metadata)
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

	gridReq := httptest.NewRequest(http.MethodGet, "/config/kbs/privacy_kb/map/grid.json", nil)
	gridReq.SetPathValue("name", "privacy_kb")
	gridRR := httptest.NewRecorder()
	apiServer.handleGetKnowledgeBaseMapGrid(gridRR, gridReq)
	if gridRR.Code != http.StatusOK {
		t.Fatalf("expected grid 200, got %d: %s", gridRR.Code, gridRR.Body.String())
	}
	var grid knowledgeBaseMapGridResponse
	if err := json.Unmarshal(gridRR.Body.Bytes(), &grid); err != nil {
		t.Fatalf("json.Unmarshal grid: %v", err)
	}
	if len(grid.Grid) == 0 {
		t.Fatalf("expected populated grid response, got %+v", grid)
	}
	if len(grid.GroupNames) != 0 {
		t.Fatalf("expected kb map grid to ship without grouped overlays, got %+v", grid.GroupNames)
	}

	topicReq := httptest.NewRequest(http.MethodGet, "/config/kbs/privacy_kb/map/topic.json", nil)
	topicReq.SetPathValue("name", "privacy_kb")
	topicRR := httptest.NewRecorder()
	apiServer.handleGetKnowledgeBaseMapTopic(topicRR, topicReq)
	if topicRR.Code != http.StatusOK {
		t.Fatalf("expected topic 200, got %d: %s", topicRR.Code, topicRR.Body.String())
	}
	var topic knowledgeBaseMapTopicResponse
	if err := json.Unmarshal(topicRR.Body.Bytes(), &topic); err != nil {
		t.Fatalf("json.Unmarshal topic: %v", err)
	}
	if len(topic.Data) == 0 {
		t.Fatalf("expected topic data, got %+v", topic)
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
