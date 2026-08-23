package providerdiscovery

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

const openAIModelsAdapterID = "openai.models.v1"

type OpenAIModelsAdapter struct{}

func (OpenAIModelsAdapter) AdapterID() string { return openAIModelsAdapterID }

func (OpenAIModelsAdapter) ValidateDiscovery(_ context.Context, plan providercatalog.DiscoveryPlan) error {
	return validateCommonPlan(plan, openAIModelsAdapterID)
}

func (OpenAIModelsAdapter) Query(plan providercatalog.DiscoveryPlan) (url.Values, error) {
	if err := validateCommonPlan(plan, openAIModelsAdapterID); err != nil {
		return nil, err
	}
	return make(url.Values), nil
}

func (OpenAIModelsAdapter) Decode(plan providercatalog.DiscoveryPlan, body io.Reader) (AdapterPage, error) {
	if err := validateCommonPlan(plan, openAIModelsAdapterID); err != nil {
		return AdapterPage{}, err
	}
	decoder := json.NewDecoder(body)
	var response struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := decoder.Decode(&response); err != nil {
		return AdapterPage{}, fmt.Errorf("%w: decode OpenAI-compatible model list: %w", ErrInvalidResponse, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return AdapterPage{}, fmt.Errorf("%w: response contains trailing JSON", ErrInvalidResponse)
	}
	if response.Object != "" && response.Object != "list" {
		return AdapterPage{}, fmt.Errorf("%w: model list object is invalid", ErrInvalidResponse)
	}
	models := make([]AdapterModel, len(response.Data))
	for index, model := range response.Data {
		if model.Object != "" && model.Object != "model" {
			return AdapterPage{}, fmt.Errorf("%w: models[%d].object is invalid", ErrInvalidResponse, index)
		}
		models[index] = AdapterModel{ProviderModelID: model.ID, DisplayName: model.ID}
	}
	return localPage(plan, models)
}
