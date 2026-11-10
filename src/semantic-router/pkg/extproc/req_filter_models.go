package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

type (
	OpenAIModel     = publicmodels.OpenAIModel
	OpenAIModelList = publicmodels.OpenAIModelList
)

// handleModelsRequest handles GET /v1/models requests and returns a direct response
// Whether to include configured models is controlled by the config's ListBackendModels setting (default: false)
// A listener restricted to some models lists only those of them the catalog has.
func (r *OpenAIRouter) handleModelsRequest(_ string, allowed routing.ListenerModels) (*ext_proc.ProcessingResponse, error) {
	resp := publicmodels.NewOpenAIModelList(r.Config, time.Now().Unix())
	if allowed != nil {
		listed := resp.Data[:0:0]
		for _, model := range resp.Data {
			if allowed.Allows(model.ID) {
				listed = append(listed, model)
			}
		}
		resp.Data = listed
	}
	return r.createJSONResponse(200, resp), nil
}
