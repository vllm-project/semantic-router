package managementserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func providerDiscoveryRequest(request managementapi.DiscoverModelsRequest, namespaceID string) (providercatalog.DiscoverModelsRequest, error) {
	fields := make(map[string]any, len(request.ConnectionFields))
	for name, raw := range request.ConnectionFields {
		if len(raw) == 0 || len(raw) > 4096 {
			return providercatalog.DiscoverModelsRequest{}, fmt.Errorf("connection field %q is empty or too large", name)
		}
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		var value any
		if err := decoder.Decode(&value); err != nil {
			return providercatalog.DiscoverModelsRequest{}, fmt.Errorf("decode connection field %q: %w", name, err)
		}
		if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
			return providercatalog.DiscoverModelsRequest{}, fmt.Errorf("connection field %q has trailing JSON", name)
		}
		fields[name] = value
	}
	return providercatalog.DiscoverModelsRequest{
		NamespaceID: namespaceID, CredentialID: request.CredentialID, Origin: request.BaseURL,
		ConnectionFields: fields, Search: request.Search, PageSize: request.PageSize,
		ProviderCursor: request.Cursor,
	}, nil
}

func providerCatalogPageDTO(result providercatalog.ListResult) managementapi.ProviderCatalogPage {
	items := make([]managementapi.ProviderCatalogItem, len(result.Providers))
	for index, provider := range result.Providers {
		items[index] = providerCatalogItemDTO(provider)
	}
	return managementapi.ProviderCatalogPage{
		Data: items,
		Page: managementapi.PageInfo{
			NextCursor: result.NextCursor, HasMore: result.HasMore, PageSize: result.PageSize,
		},
		CatalogRevision: result.CatalogRevision,
		Categories:      append([]string(nil), result.Categories...),
	}
}

func providerCatalogDetailDTO(result providercatalog.DetailResult) managementapi.ProviderCatalogDetail {
	return managementapi.ProviderCatalogDetail{
		Data: providerCatalogItemDTO(result.Provider), CatalogRevision: result.CatalogRevision,
	}
}

func discoveredModelsPageDTO(result providerdiscovery.Result, pageSize int) managementapi.DiscoverModelsPage {
	models := make([]managementapi.DiscoveredModel, len(result.Models))
	for index, model := range result.Models {
		models[index] = managementapi.DiscoveredModel{
			CatalogItemID: model.CatalogItemID, ProviderModelID: model.ProviderModelID,
			DisplayName: model.DisplayName, Capabilities: append([]string(nil), model.Capabilities...),
		}
	}
	return managementapi.DiscoverModelsPage{
		Data:            models,
		Page:            managementapi.PageInfo{NextCursor: result.NextCursor, HasMore: result.HasMore, PageSize: pageSize},
		CatalogRevision: result.CatalogRevision, DiscoveryRevision: result.DiscoveryRevision,
		ExpiresAt: result.ExpiresAt,
	}
}

func providerCatalogItemDTO(provider providercatalog.Definition) managementapi.ProviderCatalogItem {
	fields := make([]managementapi.ProviderConnectionField, len(provider.ConnectionFields))
	for index, field := range provider.ConnectionFields {
		options := make([]managementapi.ProviderFieldOption, len(field.Options))
		for optionIndex, option := range field.Options {
			options[optionIndex] = managementapi.ProviderFieldOption{Value: option.Value, Label: option.Label}
		}
		fields[index] = managementapi.ProviderConnectionField{
			Name: field.Name, Label: field.Label, Kind: string(field.Kind),
			Required: field.Required, Advanced: field.Advanced, Default: field.Default,
			Hint: field.Hint, Placeholder: field.Placeholder, Options: options,
		}
	}
	interfaces := make([]managementapi.ProviderInterface, len(provider.Interfaces))
	for index, providerInterface := range provider.Interfaces {
		interfaces[index] = managementapi.ProviderInterface{
			ID: providerInterface.ID, Label: providerInterface.Label, Default: providerInterface.Default,
			Capabilities: append([]string{}, providerInterface.Capabilities...),
		}
	}
	return managementapi.ProviderCatalogItem{
		ProviderID: provider.ID, Revision: provider.Revision,
		Display: managementapi.ProviderCatalogDisplay{
			Name: provider.Display.Name, Description: provider.Display.Description,
			Category: provider.Display.Category,
			Icon: managementapi.ProviderCatalogIcon{
				Source: provider.Display.Icon.Source, Value: provider.Display.Icon.Value,
				Color: provider.Display.Icon.Color,
			},
			Monogram: provider.Display.Monogram, Accent: provider.Display.Accent,
		},
		Credential: managementapi.ProviderCredentialPrompt{
			Mode: string(provider.Credential.Mode), Label: provider.Credential.Label, Hint: provider.Credential.Hint,
		},
		Origin: managementapi.ProviderOriginPrompt{
			Mode: string(provider.Origin.Mode), DefaultURL: provider.Origin.DefaultURL,
			BaseURLRequired: provider.Origin.Mode == providercatalog.OriginUserSupplied,
			Label:           provider.Origin.Label, Hint: provider.Origin.Hint,
		},
		DiscoverySupported: provider.Discovery != nil,
		Capabilities:       append([]string{}, provider.Capabilities...),
		ConnectionFields:   fields,
		Interfaces:         interfaces,
	}
}

func providerCatalogPublicationDTO(state providercatalog.PublicationState) managementapi.ProviderCatalogPublication {
	return managementapi.ProviderCatalogPublication{
		DesiredRevision: state.DesiredRevision,
		ActiveRevision:  state.ActiveRevision,
		Generation:      managementapi.WholeQuantity(strconv.FormatUint(state.Generation, 10)),
		UpdatedAt:       state.UpdatedAt,
	}
}
