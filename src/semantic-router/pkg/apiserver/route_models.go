//go:build !windows

package apiserver

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
)

// handleOpenAIModels handles OpenAI-compatible model listing at /v1/models
// It returns effective entrypoints and optionally concrete backend models.
// Whether to include configured models is controlled by the config's ListBackendModels setting (default: false)
func (s *ClassificationAPIServer) handleOpenAIModels(w http.ResponseWriter, _ *http.Request) {
	resp := publicmodels.NewOpenAIModelList(s.currentConfig(), time.Now().Unix())
	s.writeJSONResponse(w, http.StatusOK, resp)
}
