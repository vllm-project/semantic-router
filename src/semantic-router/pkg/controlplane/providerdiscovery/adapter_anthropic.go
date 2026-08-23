package providerdiscovery

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

const anthropicModelsAdapterID = "anthropic.models.v1"

type AnthropicModelsAdapter struct{}

func (AnthropicModelsAdapter) AdapterID() string { return anthropicModelsAdapterID }

func (AnthropicModelsAdapter) ValidateDiscovery(_ context.Context, plan providercatalog.DiscoveryPlan) error {
	return validateCommonPlan(plan, anthropicModelsAdapterID)
}

func (AnthropicModelsAdapter) Query(plan providercatalog.DiscoveryPlan) (url.Values, error) {
	if err := validateCommonPlan(plan, anthropicModelsAdapterID); err != nil {
		return nil, err
	}
	query := make(url.Values)
	query.Set("limit", strconv.Itoa(plan.PageSize))
	if plan.ProviderCursor != "" {
		query.Set("after_id", plan.ProviderCursor)
	}
	return query, nil
}

func (AnthropicModelsAdapter) Decode(plan providercatalog.DiscoveryPlan, body io.Reader) (AdapterPage, error) {
	if err := validateCommonPlan(plan, anthropicModelsAdapterID); err != nil {
		return AdapterPage{}, err
	}
	decoder := json.NewDecoder(body)
	var response struct {
		Data []struct {
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
			Type        string `json:"type"`
		} `json:"data"`
		HasMore bool   `json:"has_more"`
		LastID  string `json:"last_id"`
	}
	if err := decoder.Decode(&response); err != nil {
		return AdapterPage{}, fmt.Errorf("%w: decode Anthropic model list: %w", ErrInvalidResponse, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return AdapterPage{}, fmt.Errorf("%w: response contains trailing JSON", ErrInvalidResponse)
	}
	models := make([]AdapterModel, 0, len(response.Data))
	search := strings.ToLower(plan.Search)
	for index, model := range response.Data {
		if model.Type != "" && model.Type != "model" {
			return AdapterPage{}, fmt.Errorf("%w: models[%d].type is invalid", ErrInvalidResponse, index)
		}
		if search != "" && !strings.Contains(strings.ToLower(model.ID), search) &&
			!strings.Contains(strings.ToLower(model.DisplayName), search) {
			continue
		}
		models = append(models, AdapterModel{ProviderModelID: model.ID, DisplayName: model.DisplayName})
	}
	models, err := normalizeModels(models)
	if err != nil {
		return AdapterPage{}, err
	}
	page := AdapterPage{Models: models, HasMore: response.HasMore}
	if response.HasMore {
		if !canonicalModelText(response.LastID, 1, 512) {
			return AdapterPage{}, fmt.Errorf("%w: last_id is required when has_more is true", ErrInvalidResponse)
		}
		page.NextCursor = response.LastID
	}
	return page, nil
}
