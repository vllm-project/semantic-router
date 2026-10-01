package testcases

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOmniEmbeddingOwnerInventory(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func([]omniInventoryModel) []omniInventoryModel
		valid  bool
	}{
		{name: "named owners without default recipe", valid: true},
		{name: "global owner retained", valid: true, mutate: func(models []omniInventoryModel) []omniInventoryModel {
			global := models[0]
			global.Recipe = "@global"
			return append(models, global)
		}},
		{name: "empty default-only inventory", mutate: func([]omniInventoryModel) []omniInventoryModel { return nil }},
		{name: "mini missing", mutate: func(models []omniInventoryModel) []omniInventoryModel { return models[:1] }},
		{name: "duplicate owner", mutate: func(models []omniInventoryModel) []omniInventoryModel { return append(models, models[0]) }},
		{name: "wrong provider", mutate: func(models []omniInventoryModel) []omniInventoryModel {
			models[1].Metadata["provider"] = "candle"
			return models
		}},
		{name: "nano width assigned to mini", mutate: func(models []omniInventoryModel) []omniInventoryModel {
			models[1].Metadata["default_dimension"] = "384"
			return models
		}},
		{name: "missing audio modality", mutate: func(models []omniInventoryModel) []omniInventoryModel {
			models[1].Metadata["modalities"] = "text,image"
			return models
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			models := []omniInventoryModel{
				{Recipe: "nano", Type: "embedding", Loaded: true, Metadata: map[string]string{
					"binding": "embedding", "deployment": "nano", "contract": "embedding.v1",
					"model_type": "vela_omni", "provider": "ort", "default_dimension": "384", "modalities": "text,image,audio",
				}},
				{Recipe: "mini", Type: "embedding", Loaded: true, Metadata: map[string]string{
					"binding": "embedding", "deployment": "mini", "contract": "embedding.v1",
					"model_type": "vela_omni", "provider": "ort", "default_dimension": "768", "modalities": "text,image,audio",
				}},
			}
			if tc.mutate != nil {
				models = tc.mutate(models)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/api/v1/inventory/embedding-models" {
					t.Errorf("unexpected inventory request: %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
					return
				}
				if err := json.NewEncoder(w).Encode(map[string]interface{}{"models": models, "count": len(models)}); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(server.Close)
			probe := omniProbe{client: server.Client(), apiURL: server.URL}
			if err := probe.checkEmbeddingOwners(context.Background()); (err == nil) != tc.valid {
				t.Fatalf("inventory acceptance error=%v, want valid=%t", err, tc.valid)
			}
		})
	}
}
