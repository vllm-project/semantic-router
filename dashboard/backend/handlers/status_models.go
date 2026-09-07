package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	modelinventory "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
)

type (
	RouterModelsInfo    = modelinventory.ModelsInfoResponse
	RouterModelsSummary = modelinventory.ModelsInfoSummary
	RouterModelInfo     = modelinventory.ModelInfo
	RouterModelsSystem  = modelinventory.SystemInfo
)

func fetchRouterModelsInfo(routerAPIURL string, credentialProvider ...routerauth.CredentialProvider) *RouterModelsInfo {
	if routerAPIURL == "" {
		return nil
	}

	resp, err := routerManagementGET(strings.TrimSuffix(routerAPIURL, "/")+"/info/models", 2*time.Second, credentialProvider...)
	if err != nil {
		return nil
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil
	}

	var info RouterModelsInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil
	}

	return &info
}
