//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
)

// handleOpenAIModels handles OpenAI-compatible model listing at /v1/models
// It returns the configured auto model name and optionally the underlying models from config.
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (s *ClassificationAPIServer) handleOpenAIModels(w http.ResponseWriter, _ *http.Request) {
	resp := publicmodels.NewOpenAIModelList(s.currentConfig(), time.Now().Unix())
	s.writeJSONResponse(w, http.StatusOK, resp)
}
