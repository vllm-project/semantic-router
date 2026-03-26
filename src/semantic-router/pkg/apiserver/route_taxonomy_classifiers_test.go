//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestHandleKnowledgeBaseLifecycle(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("default_route")), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("config.Parse: %v", err)
	}

	restoreRuntimeSync := runtimeConfigSyncRunner
	runtimeConfigSyncRunner = func(sourceConfigPath string) (string, error) {
		return sourceConfigPath, nil
	}
	defer func() { runtimeConfigSyncRunner = restoreRuntimeSync }()

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
		configPath:        configPath,
	}

	createPayload := knowledgeBaseUpsertRequest{
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

	createBody, err := json.Marshal(createPayload)
	if err != nil {
		t.Fatalf("json.Marshal create payload: %v", err)
	}
	createReq := httptest.NewRequest(http.MethodPost, "/config/kbs", bytes.NewReader(createBody))
	createRR := httptest.NewRecorder()
	apiServer.handleCreateKnowledgeBase(createRR, createReq)

	if createRR.Code != http.StatusCreated {
		t.Fatalf("expected 201 Created, got %d: %s", createRR.Code, createRR.Body.String())
	}

	var created knowledgeBaseDocument
	unmarshalErr := json.Unmarshal(createRR.Body.Bytes(), &created)
	if unmarshalErr != nil {
		t.Fatalf("json.Unmarshal create response: %v", unmarshalErr)
	}
	if created.Name != "research_kb" {
		t.Fatalf("expected created KB name research_kb, got %q", created.Name)
	}
	if !created.Managed || !created.Editable || created.Builtin {
		t.Fatalf("expected managed editable custom KB, got %+v", created)
	}
	if got := created.Source.Path; got != "kbs/custom/research_kb/" {
		t.Fatalf("expected managed source path, got %q", got)
	}

	customDir := filepath.Join(tempDir, "kbs", "custom", "research_kb")
	_, statErr := os.Stat(filepath.Join(customDir, knowledgeBaseManifestName))
	if statErr != nil {
		t.Fatalf("expected labels.json in %s: %v", customDir, statErr)
	}

	updatedPayload := createPayload
	updatedPayload.Threshold = 0.52
	updatedPayload.Labels = append(updatedPayload.Labels, knowledgeBaseLabelPayload{
		Name:        "source_code",
		Description: "Internal source code",
		Exemplars:   []string{"Inspect our private source repository"},
	})
	updatedPayload.Groups["private"] = []string{"lab_notes", "source_code"}

	updateBody, err := json.Marshal(updatedPayload)
	if err != nil {
		t.Fatalf("json.Marshal update payload: %v", err)
	}
	updateReq := httptest.NewRequest(http.MethodPut, "/config/kbs/research_kb", bytes.NewReader(updateBody))
	updateReq.SetPathValue("name", "research_kb")
	updateRR := httptest.NewRecorder()
	apiServer.handleUpdateKnowledgeBase(updateRR, updateReq)

	if updateRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", updateRR.Code, updateRR.Body.String())
	}

	var updated knowledgeBaseDocument
	unmarshalErr = json.Unmarshal(updateRR.Body.Bytes(), &updated)
	if unmarshalErr != nil {
		t.Fatalf("json.Unmarshal update response: %v", unmarshalErr)
	}
	if updated.Threshold != 0.52 {
		t.Fatalf("expected updated threshold 0.52, got %v", updated.Threshold)
	}
	if len(updated.Labels) != 3 {
		t.Fatalf("expected 3 labels after update, got %d", len(updated.Labels))
	}

	listReq := httptest.NewRequest(http.MethodGet, "/config/kbs", nil)
	listRR := httptest.NewRecorder()
	apiServer.handleListKnowledgeBases(listRR, listReq)
	if listRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for list, got %d: %s", listRR.Code, listRR.Body.String())
	}
	var list knowledgeBaseListResponse
	unmarshalErr = json.Unmarshal(listRR.Body.Bytes(), &list)
	if unmarshalErr != nil {
		t.Fatalf("json.Unmarshal list response: %v", unmarshalErr)
	}
	if len(list.Items) != 2 {
		t.Fatalf("expected built-in + custom KBs in list, got %d", len(list.Items))
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/config/kbs/research_kb", nil)
	deleteReq.SetPathValue("name", "research_kb")
	deleteRR := httptest.NewRecorder()
	apiServer.handleDeleteKnowledgeBase(deleteRR, deleteReq)
	if deleteRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK on delete, got %d: %s", deleteRR.Code, deleteRR.Body.String())
	}

	_, statErr = os.Stat(customDir)
	if !os.IsNotExist(statErr) {
		t.Fatalf("expected custom classifier dir to be removed, got err=%v", statErr)
	}

	reloaded, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("config.Parse after delete: %v", err)
	}
	if len(reloaded.KnowledgeBases) != 1 || reloaded.KnowledgeBases[0].Name != "privacy_kb" {
		t.Fatalf("expected only built-in KB after delete, got %+v", reloaded.KnowledgeBases)
	}
}

func TestHandleKnowledgeBaseAllowsBuiltinMutationAndDeletion(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("default_route")), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("config.Parse: %v", err)
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
		configPath:        configPath,
	}

	restoreRuntimeSync := runtimeConfigSyncRunner
	runtimeConfigSyncRunner = func(sourceConfigPath string) (string, error) {
		return sourceConfigPath, nil
	}
	defer func() { runtimeConfigSyncRunner = restoreRuntimeSync }()

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
	body, err := json.Marshal(updatePayload)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	req := httptest.NewRequest(http.MethodPut, "/config/kbs/privacy_kb", bytes.NewReader(body))
	req.SetPathValue("name", "privacy_kb")
	rr := httptest.NewRecorder()
	apiServer.handleUpdateKnowledgeBase(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var updated knowledgeBaseDocument
	if unmarshalErr := json.Unmarshal(rr.Body.Bytes(), &updated); unmarshalErr != nil {
		t.Fatalf("json.Unmarshal update response: %v", unmarshalErr)
	}
	if updated.Name != "privacy_kb" {
		t.Fatalf("expected updated built-in KB, got %+v", updated)
	}
	if updated.Source.Path != "kbs/custom/privacy_kb/" {
		t.Fatalf("expected built-in mutation to materialize managed source, got %q", updated.Source.Path)
	}
	if !updated.Editable {
		t.Fatalf("expected updated built-in KB to remain editable, got %+v", updated)
	}

	customDir := filepath.Join(tempDir, "kbs", "custom", "privacy_kb")
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
	if len(reloaded.KnowledgeBases) != 0 {
		t.Fatalf("expected deleting built-in KB to persist removal, got %+v", reloaded.KnowledgeBases)
	}
}
