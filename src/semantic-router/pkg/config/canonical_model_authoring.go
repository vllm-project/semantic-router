package config

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func compileAuthoringModels(
	ctx context.Context,
	models []AuthoringModel,
	compiler modelauthoring.ConnectionCompiler,
) ([]routingsnapshot.Model, map[string]routingsnapshot.Model, error) {
	if len(models) != 0 && compiler == nil {
		return nil, nil, fmt.Errorf("file-authored Model connections require an injected Provider Integration compiler")
	}
	compiled := make([]routingsnapshot.Model, 0, len(models))
	byName := make(map[string]routingsnapshot.Model, len(models))
	for _, source := range models {
		card := normalizedAuthoringModelCard(source.Card)
		modelID := stableRoutingResourceID("mdl", source.Name)
		backends := make([]routingsnapshot.Backend, 0, len(source.Connections))
		seenBackendIDs := make(map[string]struct{}, len(source.Connections))
		catalogRevision := ""
		for connectionIndex, connection := range source.Connections {
			backendID, backendIDErr := authoringBackendID(source.Name, connection)
			if backendIDErr != nil {
				return nil, nil, fmt.Errorf("models[%s].connections[%d]: %w", source.Name, connectionIndex, backendIDErr)
			}
			if _, duplicate := seenBackendIDs[backendID]; duplicate {
				return nil, nil, fmt.Errorf("models[%s].connections[%d] duplicates another connection", source.Name, connectionIndex)
			}
			seenBackendIDs[backendID] = struct{}{}
			result, err := compiler.CompileConnection(ctx, modelauthoring.CompileRequest{
				BackendID: backendID, Connection: connection,
			})
			if err != nil {
				return nil, nil, fmt.Errorf("models[%s].connections[%d]: %w", source.Name, connectionIndex, err)
			}
			if catalogRevision == "" {
				catalogRevision = result.CatalogRevision
			} else if catalogRevision != result.CatalogRevision {
				return nil, nil, fmt.Errorf("models[%s].connections were compiled from different Provider Integration revisions", source.Name)
			}
			backends = append(backends, result.Backend)
		}
		model := routingsnapshot.Model{
			ID: modelID, Revision: initialRoutingResourceRevision,
			CatalogRevision: catalogRevision, Name: source.Name,
			Aliases: append([]string(nil), card.Aliases...), ParamSize: card.ParamSize,
			ContextWindowSize: card.ContextWindowSize, Description: card.Description,
			Capabilities: append([]string(nil), card.Capabilities...), Reasoning: card.Reasoning,
			LoRAs: append([]string(nil), card.LoRAs...), QualityScore: card.QualityScore,
			Modality: card.Modality, Tags: append([]string(nil), card.Tags...),
			Execution: routingsnapshot.ModelExecution{
				MaxRetries: source.Execution.MaxRetries, RetryOn: append([]string(nil), source.Execution.RetryOn...),
				RequestTimeout: source.Execution.RequestTimeout,
				StreamTimeout:  source.Execution.StreamTimeout,
			},
			Pricing: routingsnapshot.ModelPricing{
				InputCostPerMillionTokens:      cloneStringPointer(source.RuntimePricing.InputCostPerMillionTokens),
				OutputCostPerMillionTokens:     cloneStringPointer(source.RuntimePricing.OutputCostPerMillionTokens),
				CacheReadCostPerMillionTokens:  cloneStringPointer(source.RuntimePricing.CacheReadCostPerMillionTokens),
				CacheWriteCostPerMillionTokens: cloneStringPointer(source.RuntimePricing.CacheWriteCostPerMillionTokens),
			},
			Backends: backends,
		}
		compiled = append(compiled, model)
		byName[source.Name] = model
	}
	return compiled, byName, nil
}

func authoringBackendID(modelName string, connection modelauthoring.Connection) (string, error) {
	identity, err := json.Marshal(struct {
		Name             string
		Provider         string
		Interface        string
		Endpoint         string
		Model            string
		Credential       string
		ConnectionFields map[string]any
		Transport        modelauthoring.TransportOverrides
	}{
		Name: connection.Name, Provider: connection.Provider, Interface: connection.Interface,
		Endpoint: connection.Endpoint, Model: connection.Model, Credential: connection.Credential,
		ConnectionFields: connection.ConnectionFields, Transport: connection.Transport,
	})
	if err != nil {
		return "", fmt.Errorf("connection identity is not serializable: %w", err)
	}
	namespaced := append([]byte("vllm-sr/model-backend/v1\x00"+modelName+"\x00"), identity...)
	return uuid.NewSHA1(uuid.NameSpaceOID, namespaced).String(), nil
}

func normalizedAuthoringModelCard(card AuthoringModelCard) AuthoringModelCard {
	card.Aliases = stableUniqueStrings(card.Aliases)
	card.Capabilities = stableUniqueStrings(card.Capabilities)
	card.LoRAs = stableUniqueStrings(card.LoRAs)
	card.Tags = stableUniqueStrings(card.Tags)
	card.Reasoning.Efforts = stableUniqueStrings(card.Reasoning.Efforts)
	return card
}

func stableUniqueStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	result := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}
