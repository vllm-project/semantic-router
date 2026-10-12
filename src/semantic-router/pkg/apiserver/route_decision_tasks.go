//go:build !windows

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func (s *ClassificationAPIServer) handleDecisionTasks(w http.ResponseWriter, _ *http.Request) {
	var statuses []modelservice.DeploymentStatus
	if manager := modelservice.DefaultManager(); manager != nil {
		statuses = manager.Statuses()
	}
	s.writeJSONResponse(w, http.StatusOK, modelservice.ProjectTaskCatalog(s.currentConfig(), statuses))
}
