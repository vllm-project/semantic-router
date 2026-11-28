package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const routerReplayModelsPath = routerReplayAPIBasePath + "/models"

// handleRouterReplayModelsAPI serves GET /api/v1/observability/replays/models:
// each configured model's name, as records name it, and the label Insights
// shows for it, resolved from the catalog the Router routes with. No two
// models share a label.
func (r *OpenAIRouter) handleRouterReplayModelsAPI(method string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	payload := routerreplay.ModelListResponse{Object: "router_replay.models", Data: []routerreplay.ModelLabel{}}
	if r.Config != nil && r.Config.EffectiveModelRegistry != nil {
		registry := r.Config.EffectiveModelRegistry
		labels := registry.ModelLabels()
		for _, name := range registry.ModelNames() {
			payload.Data = append(payload.Data, routerreplay.ModelLabel{Name: name, Label: labels[name]})
		}
	}
	return r.createJSONResponse(200, payload)
}
