package providerdiscovery

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

const maximumDecodedModels = 10000

func validateCommonPlan(plan providercatalog.DiscoveryPlan, adapterID string) error {
	if plan.DiscoveryAdapterID != adapterID || plan.NormalizedOrigin == "" || plan.Path == "" ||
		plan.PageSize < 1 || plan.PageSize > 200 {
		return ErrInvalidRequest
	}
	if len(plan.ConnectionFields) != 0 {
		return fmt.Errorf("%w: adapter %q does not define connection fields", ErrInvalidRequest, adapterID)
	}
	return nil
}

func normalizeModels(models []AdapterModel) ([]AdapterModel, error) {
	if len(models) > maximumDecodedModels {
		return nil, fmt.Errorf("%w: model count exceeds %d", ErrInvalidResponse, maximumDecodedModels)
	}
	result := make([]AdapterModel, 0, len(models))
	seen := make(map[string]struct{}, len(models))
	for index, model := range models {
		if !canonicalModelText(model.ProviderModelID, 1, 512) {
			return nil, fmt.Errorf("%w: models[%d].id is invalid", ErrInvalidResponse, index)
		}
		if _, duplicate := seen[model.ProviderModelID]; duplicate {
			return nil, fmt.Errorf("%w: duplicate model id %q", ErrInvalidResponse, model.ProviderModelID)
		}
		seen[model.ProviderModelID] = struct{}{}
		if model.DisplayName == "" {
			model.DisplayName = model.ProviderModelID
		}
		if !canonicalModelText(model.DisplayName, 1, 512) {
			return nil, fmt.Errorf("%w: models[%d].display_name is invalid", ErrInvalidResponse, index)
		}
		model.Capabilities = append([]string(nil), model.Capabilities...)
		sort.Strings(model.Capabilities)
		for capabilityIndex, capability := range model.Capabilities {
			if !canonicalModelText(capability, 1, 128) ||
				(capabilityIndex > 0 && capability == model.Capabilities[capabilityIndex-1]) {
				return nil, fmt.Errorf("%w: models[%d].capabilities is invalid", ErrInvalidResponse, index)
			}
		}
		result = append(result, model)
	}
	sort.Slice(result, func(left, right int) bool {
		return result[left].ProviderModelID < result[right].ProviderModelID
	})
	return result, nil
}

func localPage(plan providercatalog.DiscoveryPlan, models []AdapterModel) (AdapterPage, error) {
	normalized, err := normalizeModels(models)
	if err != nil {
		return AdapterPage{}, err
	}
	search := strings.ToLower(plan.Search)
	selected := make([]AdapterModel, 0, plan.PageSize+1)
	for _, model := range normalized {
		if plan.ProviderCursor != "" && model.ProviderModelID <= plan.ProviderCursor {
			continue
		}
		if search != "" && !strings.Contains(strings.ToLower(model.ProviderModelID), search) &&
			!strings.Contains(strings.ToLower(model.DisplayName), search) {
			continue
		}
		selected = append(selected, model)
		if len(selected) == plan.PageSize+1 {
			break
		}
	}
	page := AdapterPage{Models: selected}
	if len(selected) > plan.PageSize {
		page.Models = selected[:plan.PageSize]
		page.HasMore = true
		page.NextCursor = page.Models[len(page.Models)-1].ProviderModelID
	}
	return page, nil
}

func canonicalModelText(value string, minimum, maximum int) bool {
	return utf8.ValidString(value) && len(value) >= minimum && len(value) <= maximum &&
		strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n\t")
}
