package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	modelinventory "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
)

type (
	RouterModelsInfo    = modelinventory.ModelsInfoResponse
	RouterModelsSummary = modelinventory.ModelsInfoSummary
	RouterModelInfo     = modelinventory.ModelInfo
	RouterModelsSystem  = modelinventory.SystemInfo
)

func fetchRouterModelsInfo(routerAPIURL string) *RouterModelsInfo {
	if routerAPIURL == "" {
		return nil
	}

	req, err := http.NewRequest(http.MethodGet, strings.TrimSuffix(routerAPIURL, "/")+"/info/models", nil)
	if err != nil {
		return nil
	}

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
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
