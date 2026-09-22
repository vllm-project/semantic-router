//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func stubKnowledgeBaseMapEmbeddings(t *testing.T, server *ClassificationAPIServer) {
	t.Helper()
	provider, err := embedding.NewFuncProvider("test", 3, func(_ context.Context, text string) ([]float32, error) {
		text = strings.TrimSpace(text)
		return []float32{float32(len(text)), float32(len(strings.Fields(text))) + 1, float32(len(text)%7) + 1}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	cfg := server.currentConfig()
	model := knowledgeBaseMapModelType(cfg)
	prepared := embedding.NewSet(map[string]embedding.Provider{model: provider}, model)
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Embeddings: prepared})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	server.classificationSvc = services.NewRecipeClassificationService(classifiers, cfg)
}

func TestHandleKnowledgeBaseMapMetadataEndpoint(t *testing.T) {
	apiServer, _, _ := newTestKnowledgeBaseAPIServer(t)
	stubKnowledgeBaseMapEmbeddings(t, apiServer)

	metadataReq := httptest.NewRequest(http.MethodGet, "/api/v1/storage/knowledge-bases/privacy_kb/map/metadata", nil)
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
}

func TestHandleKnowledgeBaseMapDataEndpoint(t *testing.T) {
	apiServer, _, _ := newTestKnowledgeBaseAPIServer(t)
	stubKnowledgeBaseMapEmbeddings(t, apiServer)

	dataReq := httptest.NewRequest(http.MethodGet, "/api/v1/storage/knowledge-bases/privacy_kb/map/data.ndjson", nil)
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
	stubKnowledgeBaseMapEmbeddings(t, apiServer)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/storage/knowledge-bases/missing/map/metadata", nil)
	req.SetPathValue("name", "missing")
	rr := httptest.NewRecorder()
	apiServer.handleGetKnowledgeBaseMapMetadata(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", rr.Code, rr.Body.String())
	}
}
