package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"os"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

// PublicSystemOneHandler keeps the native surface stable across instance modes.
// The shared handler enforces the same listener grant as the standalone gateway.
func PublicSystemOneHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		data, err := readPersistedDashboardConfig(configPath)
		if err != nil {
			http.Error(w, "Configuration unavailable", 503)
			return
		}
		cfg, err := routerconfig.ParseYAMLBytes(data)
		if err != nil {
			http.Error(w, "Configuration unavailable", 503)
			return
		}
		listener, err := systemone.SelectListener(cfg.Listeners, os.Getenv("VLLM_SR_SYSTEMONE_LISTENER"))
		if err != nil {
			http.NotFound(w, r)
			return
		}
		systemone.Handler(cfg, listener, func(ctx context.Context, deployment string, body json.RawMessage) (int, []byte, error) {
			payload, err := json.Marshal(map[string]any{"deployment": deployment, "request": body, "expected_artifact": cfg.ModelDeployments[deployment].Artifact})
			if err != nil {
				return 0, nil, err
			}
			return instanceRequest(ctx, http.MethodPost, "/systemone", payload)
		})(w, r)
	}
}
