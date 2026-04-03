//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func newTestKnowledgeBaseAPIServer(t *testing.T) (*ClassificationAPIServer, string, string) {
	t.Helper()
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("default_route")), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("config.Parse: %v", err)
	}
	return &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
		configPath:        configPath,
	}, tempDir, configPath
}

func withStubbedRuntimeConfigSync(t *testing.T) {
	t.Helper()
	restoreRuntimeSync := runtimeConfigSyncRunner
	runtimeConfigSyncRunner = func(sourceConfigPath string) (string, error) {
		return sourceConfigPath, nil
	}
	t.Cleanup(func() { runtimeConfigSyncRunner = restoreRuntimeSync })
}

func mustMarshalKnowledgeBasePayload(t *testing.T, payload knowledgeBaseUpsertRequest) []byte {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal payload: %v", err)
	}
	return body
}

func mustDecodeKnowledgeBaseDocument(t *testing.T, rr *httptest.ResponseRecorder) knowledgeBaseDocument {
	t.Helper()
	var document knowledgeBaseDocument
	if err := json.Unmarshal(rr.Body.Bytes(), &document); err != nil {
		t.Fatalf("json.Unmarshal response: %v", err)
	}
	return document
}

func mustDecodeKnowledgeBaseList(t *testing.T, rr *httptest.ResponseRecorder) knowledgeBaseListResponse {
	t.Helper()
	var list knowledgeBaseListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &list); err != nil {
		t.Fatalf("json.Unmarshal list response: %v", err)
	}
	return list
}

func TestHandleKnowledgeBaseLifecycle(t *testing.T) {
	apiServer, tempDir, configPath := newTestKnowledgeBaseAPIServer(t)
	withStubbedRuntimeConfigSync(t)

	createPayload := testKnowledgeBasePayload()
	created := createKnowledgeBaseDocument(t, apiServer, createPayload)
	assertCreatedKnowledgeBase(t, created)

	customDir := filepath.Join(tempDir, ".vllm-sr", "knowledge_bases", "research_kb")
	assertKnowledgeBaseManifestExists(t, customDir)

	updatedPayload := createPayload
	updatedPayload.Threshold = 0.52
	updatedPayload.Labels = append(updatedPayload.Labels, knowledgeBaseLabelPayload{
		Name:        "source_code",
		Description: "Internal source code",
		Exemplars:   []string{"Inspect our private source repository"},
	})
	updatedPayload.Groups["private"] = []string{"lab_notes", "source_code"}

	updated := updateKnowledgeBaseDocument(t, apiServer, "research_kb", updatedPayload)
	assertUpdatedKnowledgeBase(t, updated)
	assertKnowledgeBaseListContainsBuiltInAndCustom(t, apiServer)
	deleteKnowledgeBase(t, apiServer, "research_kb")
	assertKnowledgeBaseRemoved(t, customDir, configPath)
}

func testKnowledgeBasePayload() knowledgeBaseUpsertRequest {
	return knowledgeBaseUpsertRequest{
		Name:        "research_kb",
		Threshold:   0.41,
		Description: "Custom knowledge base for research workflows.",
		Labels: []knowledgeBaseLabelPayload{
			{
				Name:        "papers",
				Description: "Public research papers",
				Exemplars:   []string{"Summarize this published paper"},
			},
			{
				Name:        "lab_notes",
				Description: "Private notes",
				Exemplars:   []string{"Review our private experiment notes"},
			},
		},
		Groups: map[string][]string{
			"public":  {"papers"},
			"private": {"lab_notes"},
		},
		Metrics: []config.KnowledgeBaseMetricConfig{
			{
				Name:          "private_vs_public",
				Type:          config.KBMetricTypeGroupMargin,
				PositiveGroup: "private",
				NegativeGroup: "public",
			},
		},
	}
}

func createKnowledgeBaseDocument(t *testing.T, apiServer *ClassificationAPIServer, payload knowledgeBaseUpsertRequest) knowledgeBaseDocument {
	t.Helper()
	createBody := mustMarshalKnowledgeBasePayload(t, payload)
	createReq := httptest.NewRequest(http.MethodPost, "/config/kbs", bytes.NewReader(createBody))
	createRR := httptest.NewRecorder()
	apiServer.handleCreateKnowledgeBase(createRR, createReq)
	if createRR.Code != http.StatusCreated {
		t.Fatalf("expected 201 Created, got %d: %s", createRR.Code, createRR.Body.String())
	}
	return mustDecodeKnowledgeBaseDocument(t, createRR)
}

func assertCreatedKnowledgeBase(t *testing.T, created knowledgeBaseDocument) {
	t.Helper()
	if created.Name != "research_kb" {
		t.Fatalf("expected created KB name research_kb, got %q", created.Name)
	}
	if !created.Managed || !created.Editable || created.Builtin {
		t.Fatalf("expected managed editable custom KB, got %+v", created)
	}
	if got := created.Source.Path; got != "knowledge_bases/research_kb/" {
		t.Fatalf("expected managed source path, got %q", got)
	}
}

func assertKnowledgeBaseManifestExists(t *testing.T, customDir string) {
	t.Helper()
	_, statErr := os.Stat(filepath.Join(customDir, knowledgeBaseManifestName))
	if statErr != nil {
		t.Fatalf("expected labels.json in %s: %v", customDir, statErr)
	}
}

func updateKnowledgeBaseDocument(t *testing.T, apiServer *ClassificationAPIServer, name string, payload knowledgeBaseUpsertRequest) knowledgeBaseDocument {
	t.Helper()
	updateBody := mustMarshalKnowledgeBasePayload(t, payload)
	updateReq := httptest.NewRequest(http.MethodPut, "/config/kbs/"+name, bytes.NewReader(updateBody))
	updateReq.SetPathValue("name", name)
	updateRR := httptest.NewRecorder()
	apiServer.handleUpdateKnowledgeBase(updateRR, updateReq)
	if updateRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", updateRR.Code, updateRR.Body.String())
	}
	return mustDecodeKnowledgeBaseDocument(t, updateRR)
}

func assertUpdatedKnowledgeBase(t *testing.T, updated knowledgeBaseDocument) {
	t.Helper()
	if updated.Threshold != 0.52 {
		t.Fatalf("expected updated threshold 0.52, got %v", updated.Threshold)
	}
	if len(updated.Labels) != 3 {
		t.Fatalf("expected 3 labels after update, got %d", len(updated.Labels))
	}
}

func assertKnowledgeBaseListContainsBuiltInAndCustom(t *testing.T, apiServer *ClassificationAPIServer) {
	t.Helper()
	listReq := httptest.NewRequest(http.MethodGet, "/config/kbs", nil)
	listRR := httptest.NewRecorder()
	apiServer.handleListKnowledgeBases(listRR, listReq)
	if listRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for list, got %d: %s", listRR.Code, listRR.Body.String())
	}
	list := mustDecodeKnowledgeBaseList(t, listRR)
	expected := append(defaultKnowledgeBaseNames(), "research_kb")
	assertKnowledgeBaseNames(t, list.Items, expected)
}

func deleteKnowledgeBase(t *testing.T, apiServer *ClassificationAPIServer, name string) {
	t.Helper()
	deleteReq := httptest.NewRequest(http.MethodDelete, "/config/kbs/"+name, nil)
	deleteReq.SetPathValue("name", name)
	deleteRR := httptest.NewRecorder()
	apiServer.handleDeleteKnowledgeBase(deleteRR, deleteReq)
	if deleteRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK on delete, got %d: %s", deleteRR.Code, deleteRR.Body.String())
	}
}

func assertKnowledgeBaseRemoved(t *testing.T, customDir string, configPath string) {
	t.Helper()
	_, statErr := os.Stat(customDir)
	if !os.IsNotExist(statErr) {
		t.Fatalf("expected custom KB dir to be removed, got err=%v", statErr)
	}

	reloaded, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("config.Parse after delete: %v", err)
	}
	assertKnowledgeBaseConfigNames(t, reloaded.KnowledgeBases, defaultKnowledgeBaseNames())
}

func TestHandleKnowledgeBaseAllowsBuiltinMutationAndDeletion(t *testing.T) {
	apiServer, tempDir, configPath := newTestKnowledgeBaseAPIServer(t)
	withStubbedRuntimeConfigSync(t)

	updatePayload := knowledgeBaseUpsertRequest{
		Name:        "privacy_kb",
		Threshold:   0.9,
		Description: "Updated privacy KB",
		Labels: []knowledgeBaseLabelPayload{
			{Name: "proprietary_code", Exemplars: []string{"Inspect our internal code"}},
		},
		Groups: map[string][]string{
			"private": {"proprietary_code"},
		},
	}
	body := mustMarshalKnowledgeBasePayload(t, updatePayload)

	req := httptest.NewRequest(http.MethodPut, "/config/kbs/privacy_kb", bytes.NewReader(body))
	req.SetPathValue("name", "privacy_kb")
	rr := httptest.NewRecorder()
	apiServer.handleUpdateKnowledgeBase(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	updated := mustDecodeKnowledgeBaseDocument(t, rr)
	if updated.Name != "privacy_kb" {
		t.Fatalf("expected updated built-in KB, got %+v", updated)
	}
	if updated.Source.Path != "knowledge_bases/privacy/" {
		t.Fatalf("expected built-in mutation to keep the canonical source dir, got %q", updated.Source.Path)
	}
	if !updated.Editable {
		t.Fatalf("expected updated built-in KB to remain editable, got %+v", updated)
	}

	customDir := filepath.Join(tempDir, ".vllm-sr", "knowledge_bases", "privacy")
	if _, manifestErr := os.Stat(filepath.Join(customDir, knowledgeBaseManifestName)); manifestErr != nil {
		t.Fatalf("expected managed KB assets for built-in update: %v", manifestErr)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/config/kbs/privacy_kb", nil)
	deleteReq.SetPathValue("name", "privacy_kb")
	deleteRR := httptest.NewRecorder()
	apiServer.handleDeleteKnowledgeBase(deleteRR, deleteReq)
	if deleteRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK on delete, got %d: %s", deleteRR.Code, deleteRR.Body.String())
	}

	if _, statErr := os.Stat(customDir); !os.IsNotExist(statErr) {
		t.Fatalf("expected built-in managed asset dir to be removed, got err=%v", statErr)
	}

	reloaded, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("config.Parse after delete: %v", err)
	}
	assertKnowledgeBaseConfigNames(t, reloaded.KnowledgeBases, []string{"mmlu_kb"})
}

func defaultKnowledgeBaseNames() []string {
	defaults := config.DefaultCanonicalGlobal().ModelCatalog.KBs
	names := make([]string, 0, len(defaults))
	for _, kb := range defaults {
		names = append(names, kb.Name)
	}
	slices.Sort(names)
	return names
}

func assertKnowledgeBaseNames(t *testing.T, items []knowledgeBaseDocument, want []string) {
	t.Helper()
	got := make([]string, 0, len(items))
	for _, item := range items {
		got = append(got, item.Name)
	}
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("knowledge base names mismatch\nwant: %v\ngot:  %v", want, got)
	}
}

func assertKnowledgeBaseConfigNames(t *testing.T, items []config.KnowledgeBaseConfig, want []string) {
	t.Helper()
	got := make([]string, 0, len(items))
	for _, item := range items {
		got = append(got, item.Name)
	}
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("knowledge base config names mismatch\nwant: %v\ngot:  %v", want, got)
	}
}
