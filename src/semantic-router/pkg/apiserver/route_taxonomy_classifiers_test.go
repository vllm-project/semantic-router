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

func TestHandleTaxonomyClassifierLifecycle(t *testing.T) {
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

	createPayload := taxonomyClassifierUpsertRequest{
		Name:              "research_classifier",
		Threshold:         0.41,
		SecurityThreshold: 0.28,
		Description:       "Custom taxonomy for research workflows.",
		Tiers: []taxonomyClassifierTierPayload{
			{Name: "internal"},
			{Name: "external"},
		},
		Categories: []taxonomyClassifierCategoryPayload{
			{
				Name:        "papers",
				Tier:        "external",
				Description: "Public research papers",
				Exemplars:   []string{"Summarize this published paper"},
			},
			{
				Name:        "lab_notes",
				Tier:        "internal",
				Description: "Private notes",
				Exemplars:   []string{"Review our private experiment notes"},
			},
		},
		TierGroups: map[string][]string{
			"privacy_categories": {"lab_notes"},
		},
	}

	createBody, err := json.Marshal(createPayload)
	if err != nil {
		t.Fatalf("json.Marshal create payload: %v", err)
	}
	createReq := httptest.NewRequest(http.MethodPost, "/config/classifiers", bytes.NewReader(createBody))
	createRR := httptest.NewRecorder()
	apiServer.handleCreateTaxonomyClassifier(createRR, createReq)

	if createRR.Code != http.StatusCreated {
		t.Fatalf("expected 201 Created, got %d: %s", createRR.Code, createRR.Body.String())
	}

	var created taxonomyClassifierDocument
	unmarshalErr := json.Unmarshal(createRR.Body.Bytes(), &created)
	if unmarshalErr != nil {
		t.Fatalf("json.Unmarshal create response: %v", unmarshalErr)
	}
	if created.Name != "research_classifier" {
		t.Fatalf("expected created classifier name research_classifier, got %q", created.Name)
	}
	if !created.Managed || !created.Editable || created.Builtin {
		t.Fatalf("expected managed editable custom classifier, got %+v", created)
	}
	if got := created.Source.Path; got != "classifiers/custom/research_classifier/" {
		t.Fatalf("expected managed source path, got %q", got)
	}

	customDir := filepath.Join(tempDir, "classifiers", "custom", "research_classifier")
	_, statErr := os.Stat(filepath.Join(customDir, "taxonomy.json"))
	if statErr != nil {
		t.Fatalf("expected taxonomy.json in %s: %v", customDir, statErr)
	}

	updatedPayload := createPayload
	updatedPayload.Threshold = 0.52
	updatedPayload.Categories = append(updatedPayload.Categories, taxonomyClassifierCategoryPayload{
		Name:        "source_code",
		Tier:        "internal",
		Description: "Internal source code",
		Exemplars:   []string{"Inspect our private source repository"},
	})

	updateBody, err := json.Marshal(updatedPayload)
	if err != nil {
		t.Fatalf("json.Marshal update payload: %v", err)
	}
	updateReq := httptest.NewRequest(http.MethodPut, "/config/classifiers/research_classifier", bytes.NewReader(updateBody))
	updateReq.SetPathValue("name", "research_classifier")
	updateRR := httptest.NewRecorder()
	apiServer.handleUpdateTaxonomyClassifier(updateRR, updateReq)

	if updateRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", updateRR.Code, updateRR.Body.String())
	}

	var updated taxonomyClassifierDocument
	unmarshalErr = json.Unmarshal(updateRR.Body.Bytes(), &updated)
	if unmarshalErr != nil {
		t.Fatalf("json.Unmarshal update response: %v", unmarshalErr)
	}
	if updated.Threshold != 0.52 {
		t.Fatalf("expected updated threshold 0.52, got %v", updated.Threshold)
	}
	if len(updated.Categories) != 3 {
		t.Fatalf("expected 3 categories after update, got %d", len(updated.Categories))
	}

	listReq := httptest.NewRequest(http.MethodGet, "/config/classifiers", nil)
	listRR := httptest.NewRecorder()
	apiServer.handleListTaxonomyClassifiers(listRR, listReq)
	if listRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for list, got %d: %s", listRR.Code, listRR.Body.String())
	}
	var list taxonomyClassifierListResponse
	unmarshalErr = json.Unmarshal(listRR.Body.Bytes(), &list)
	if unmarshalErr != nil {
		t.Fatalf("json.Unmarshal list response: %v", unmarshalErr)
	}
	if len(list.Items) != 2 {
		t.Fatalf("expected built-in + custom classifiers in list, got %d", len(list.Items))
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/config/classifiers/research_classifier", nil)
	deleteReq.SetPathValue("name", "research_classifier")
	deleteRR := httptest.NewRecorder()
	apiServer.handleDeleteTaxonomyClassifier(deleteRR, deleteReq)
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
	if len(reloaded.TaxonomyClassifiers) != 1 || reloaded.TaxonomyClassifiers[0].Name != "privacy_classifier" {
		t.Fatalf("expected only built-in classifier after delete, got %+v", reloaded.TaxonomyClassifiers)
	}
}

func TestHandleTaxonomyClassifierRejectsBuiltinMutation(t *testing.T) {
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

	updatePayload := taxonomyClassifierUpsertRequest{
		Name:              "privacy_classifier",
		Threshold:         0.9,
		SecurityThreshold: 0.8,
		Categories: []taxonomyClassifierCategoryPayload{
			{Name: "proprietary_code", Tier: "privacy_policy", Exemplars: []string{"Inspect our internal code"}},
		},
	}
	body, err := json.Marshal(updatePayload)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	req := httptest.NewRequest(http.MethodPut, "/config/classifiers/privacy_classifier", bytes.NewReader(body))
	req.SetPathValue("name", "privacy_classifier")
	rr := httptest.NewRecorder()
	apiServer.handleUpdateTaxonomyClassifier(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403 Forbidden, got %d: %s", rr.Code, rr.Body.String())
	}
}
