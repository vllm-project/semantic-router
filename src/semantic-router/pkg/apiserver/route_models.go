//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
)

// handleOpenAIModels handles OpenAI-compatible model listing at /v1/models
// It returns only explicit Entrypoint aliases and, when enabled, concrete
// backend Models. Routing algorithms never create additional public names.
func (s *ClassificationAPIServer) handleOpenAIModels(w http.ResponseWriter, _ *http.Request) {
	resp := publicmodels.NewOpenAIModelList(s.currentConfig(), time.Now().Unix())
	s.writeJSONResponse(w, http.StatusOK, resp)
}
