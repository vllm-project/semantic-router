package modelresearch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	modelinventory "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
)

type openAIModelList struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

func fetchRuntimeModelsInfo(client *http.Client, apiBase string) (*modelinventory.ModelsInfoResponse, error) {
	if strings.TrimSpace(apiBase) == "" {
		return nil, nil
	}

	req, err := http.NewRequest(http.MethodGet, strings.TrimRight(apiBase, "/")+"/info/models", nil)
	if err != nil {
		return nil, err
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("runtime models endpoint returned %d", resp.StatusCode)
	}

	var payload modelinventory.ModelsInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	return &payload, nil
}

func fetchAvailableModels(client *http.Client, apiBase string) ([]string, error) {
	if strings.TrimSpace(apiBase) == "" {
		return nil, nil
	}

	req, err := http.NewRequest(http.MethodGet, strings.TrimRight(apiBase, "/")+"/v1/models", nil)
	if err != nil {
		return nil, err
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("models endpoint returned %d", resp.StatusCode)
	}

	var payload openAIModelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}

	models := make([]string, 0, len(payload.Data))
	for _, item := range payload.Data {
		models = append(models, strings.TrimSpace(item.ID))
	}
	return models, nil
}

func newHTTPClient() *http.Client {
	return &http.Client{Timeout: 5 * time.Second}
}
