package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchRouterModelsInfoSuccess(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/info/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(RouterModelsInfo{
			Models: []RouterModelInfo{
				{
					Name:      "category_classifier",
					Type:      "intent_classification",
					Loaded:    true,
					State:     "ready",
					ModelPath: "models/mmbert32k-intent-classifier-merged",
				},
			},
			Summary: RouterModelsSummary{
				Ready:        true,
				Phase:        "ready",
				LoadedModels: 1,
				TotalModels:  1,
			},
		})
	}))
	defer server.Close()

	info := fetchRouterModelsInfo(server.URL + "/")
	if info == nil {
		t.Fatalf("expected router models info")
	}
	if info.Summary.Phase != "ready" {
		t.Fatalf("expected ready phase, got %+v", info.Summary)
	}
	if len(info.Models) != 1 || info.Models[0].Name != "category_classifier" {
		t.Fatalf("unexpected models payload %+v", info.Models)
	}
}

func TestFetchRouterModelsInfoIgnoresNonOKResponses(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "router unavailable", http.StatusBadGateway)
	}))
	defer server.Close()

	if info := fetchRouterModelsInfo(server.URL); info != nil {
		t.Fatalf("expected nil info on non-200 response, got %+v", info)
	}
}
