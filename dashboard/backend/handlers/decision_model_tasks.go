package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// DecisionTaskCatalogHandler keeps task authoring available in both serving
// modes and while the frontend is unavailable. Only current observations may mark deployments ready; a saved
// configuration is sufficient for templates and declared capability previews.
func DecisionTaskCatalogHandler(configPath, upstream string, providers ...routerauth.CredentialProvider) http.HandlerFunc {
	transport := decisionModelTransport{upstream: strings.TrimRight(upstream, "/"), client: &http.Client{Timeout: 5 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	if len(providers) > 0 {
		transport.provider = providers[0]
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			decisionModelError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Task discovery requires GET")
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		ctx, cancel := context.WithTimeout(r.Context(), 6*time.Second)
		defer cancel()
		response, body, err := transport.request(ctx, http.MethodGet, "/api/v1/diagnostics/models/tasks", nil)
		if err == nil && response != nil && response.StatusCode == http.StatusOK {
			var catalog modelservice.TaskCatalogResponse
			if json.Unmarshal(body, &catalog) == nil && catalog.Tasks != nil && catalog.Bindings != nil {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write(body)
				return
			}
		}
		data, err := readPersistedDashboardConfig(configPath)
		if err != nil {
			decisionModelError(w, 503, "configuration_unavailable", "The saved task configuration is unavailable")
			return
		}
		cfg, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(data)
		if err != nil {
			decisionModelError(w, 503, "configuration_unavailable", "The saved task configuration could not be read")
			return
		}
		catalog := modelservice.ProjectTaskCatalog(cfg, nil)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			modelservice.TaskCatalogResponse
			RuntimeObserved bool `json:"runtime_observed"`
		}{TaskCatalogResponse: catalog, RuntimeObserved: false})
	}
}
