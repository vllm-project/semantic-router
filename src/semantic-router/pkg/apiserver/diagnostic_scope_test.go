//go:build !windows

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

func TestConvenienceDiagnosticsHTTPRejectsForeignRecipe(t *testing.T) {
	cfg, service := preparedInventoryService(t, serving.New(nil, nil), &config.RouterConfig{Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "private"}}})
	api := &ClassificationAPIServer{config: cfg, classificationSvc: service}
	for _, test := range []struct {
		name, body string
		handler    func(http.ResponseWriter, *http.Request)
	}{
		{"batch", `{"recipe":"foreign","texts":["hello"]}`, api.handleBatchClassification},
		{"combined", `{"recipe":"foreign","text":"hello"}`, api.handleCombinedClassification},
	} {
		t.Run(test.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			test.handler(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(test.body)))
			if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "INVALID_RECIPE") {
				t.Fatalf("requested recipe silently fell back: %d %s", w.Code, w.Body.String())
			}
		})
	}
}
