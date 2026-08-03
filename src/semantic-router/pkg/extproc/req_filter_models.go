package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
)

type (
	OpenAIModel     = publicmodels.OpenAIModel
	OpenAIModelList = publicmodels.OpenAIModelList
)

// handleModelsRequest handles GET /v1/models requests and returns a direct response
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (r *OpenAIRouter) handleModelsRequest(_ string) (*ext_proc.ProcessingResponse, error) {
	resp := publicmodels.NewOpenAIModelList(r.Config, time.Now().Unix())
	return r.createJSONResponse(200, resp), nil
}
