//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type metadataInventoryService struct {
	classificationService
	entries []binding.PreparedBinding
}

func (s metadataInventoryService) PreparedBindings() ([]binding.PreparedBinding, bool) {
	return s.entries, true
}

func TestPreparedInventoryPublishesSourceNamesAndActualInputLimits(t *testing.T) {
	var entries []binding.PreparedBinding
	for _, name := range []string{"Domain", "FactCheck", "Feedback", "Guard", "PII", "Safety", "Hazard", "Embedding"} {
		spec := config.GetModelByPath("models/Vela-1.0-Encoder-307M-" + name)
		if spec == nil {
			t.Fatalf("fixture source checkpoint %s is not registered", name)
		}
		entries = append(entries, binding.PreparedBinding{
			Identity: binding.Identity{Recipe: "example", Name: name, Deployment: name, Contract: "label_distribution.v1", Adapter: "mmbert"},
			Artifact: "models/derived/opaque-" + name, Revision: spec.Revision + ":" + strings.Repeat("a", 64),
			Capability: binding.Capability{Provider: "ort", Device: "rocm:0", Precision: "native",
				Limits: binding.Limits{ModelTokens: 32768, TaskTokens: 32768, DocumentTokens: 262144, DeploymentTokens: 262144, Overflow: "window"},
				Window: &binding.WindowCapability{Size: 32768, Overlap: 256}},
		})
	}
	api := &ClassificationAPIServer{classificationSvc: metadataInventoryService{entries: entries}}
	response := httptest.NewRecorder()
	api.handleModelsInfo(response, httptest.NewRequest("GET", "/api/v1/inventory/models", nil))
	var inventory ModelsInfoResponse
	if err := json.Unmarshal(response.Body.Bytes(), &inventory); err != nil {
		t.Fatal(err)
	}
	if len(inventory.Models) != 8 || inventory.Summary.LoadedModels != 8 {
		t.Fatalf("inventory lost prepared models: %+v", inventory.Summary)
	}
	for _, model := range inventory.Models {
		want := "llm-semantic-router/Vela-1.0-Encoder-307M-" + model.Name
		if model.Registry == nil || model.Registry.RepoID != want || model.Metadata["registry_match"] != "source_revision" {
			t.Fatalf("derived artifact lost its declared source identity: %+v", model)
		}
		for key, value := range map[string]string{"forward_max_tokens": "32768", "input_max_tokens": "262144", "document_max_tokens": "262144", "window_size": "32768", "window_overlap": "256", "max_sequence_length": "262144"} {
			if model.Metadata[key] != value {
				t.Errorf("%s %s = %q, want %q", model.Name, key, model.Metadata[key], value)
			}
		}
		if !strings.HasPrefix(model.ModelPath, "models/derived/") {
			t.Fatalf("source metadata replaced execution artifact: %+v", model)
		}
	}
}

func TestPreparedInventoryDoesNotGuessUnknownOrExternalSource(t *testing.T) {
	spec := config.GetModelByPath("models/Vela-1.0-Encoder-307M-Domain")
	for _, test := range []struct{ revision, device string }{
		{"", "rocm:0"}, {"main:digest", "rocm:0"}, {strings.Repeat("a", 40) + ":digest", "rocm:0"},
		{spec.Revision + ":digest", "external"},
	} {
		models, ok := preparedModelsInfo(metadataInventoryService{entries: []binding.PreparedBinding{{
			Artifact: "models/derived/Domain-0ba69f4fb67387a2", Revision: test.revision,
			Capability: binding.Capability{Device: test.device},
		}}})
		if !ok || len(models) != 1 || models[0].Registry != nil {
			t.Fatalf("guessed checkpoint from an artifact name: %+v", models)
		}
		for _, key := range []string{"forward_max_tokens", "input_max_tokens", "document_max_tokens", "window_size", "window_overlap"} {
			if _, exists := models[0].Metadata[key]; exists {
				t.Errorf("unknown capability published %s", key)
			}
		}
	}
}

func TestPreparedInputLimitsSeparateTruncationBudgetFromForwardCapacity(t *testing.T) {
	metadata := map[string]string{}
	addPreparedInputLimits(metadata, binding.Capability{Limits: binding.Limits{ModelTokens: 32768, TaskTokens: 32768, DeploymentTokens: 512, Overflow: "truncate"}})
	if metadata["forward_max_tokens"] != "32768" || metadata["input_max_tokens"] != "512" || metadata["document_max_tokens"] != "" || metadata["window_size"] != "" {
		t.Fatalf("non-window budget misreported as context window: %+v", metadata)
	}
}
