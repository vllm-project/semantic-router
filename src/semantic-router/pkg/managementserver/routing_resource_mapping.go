package managementserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func routingRecipeInput(input managementapi.RoutingRecipeWrite) routingmanagement.RecipeInput {
	return routingmanagement.RecipeInput{
		ID: input.ID, Name: input.Name, Description: input.Description,
		Document: append(json.RawMessage(nil), input.Document...),
	}
}

func routingModelInput(input managementapi.RoutingModelWrite) (routingmanagement.ModelInput, error) {
	backends := make([]routingmanagement.ModelBackendInput, len(input.Backends))
	for index, backend := range input.Backends {
		fields, err := decodeRoutingConnectionFields(backend.ConnectionFields)
		if err != nil {
			return routingmanagement.ModelInput{}, err
		}
		backends[index] = routingmanagement.ModelBackendInput{
			ProviderID: backend.ProviderID, InterfaceID: backend.InterfaceID, ProviderModelID: backend.ProviderModelID,
			CredentialID: backend.CredentialID, Origin: backend.BaseURL,
			ConnectionFields: fields, Weight: backend.Weight,
		}
	}
	return routingmanagement.ModelInput{
		ID: input.ID, Name: input.Name, Aliases: input.Aliases, Capabilities: input.Capabilities,
		ParamSize: input.ParamSize, ContextWindowSize: input.ContextWindowSize, Description: input.Description,
		Reasoning: routingReasoning(input.Reasoning), LoRAs: input.LoRAs,
		QualityScore: input.QualityScore, Modality: input.Modality, Tags: input.Tags,
		Execution: routingModelExecution(input.Control),
		Pricing:   routingsnapshot.ModelPricing(input.Pricing), Backends: backends,
	}, nil
}

func routingModelPatch(input managementapi.RoutingModelPatch) (routingmanagement.ModelPatch, error) {
	patch := routingmanagement.ModelPatch{
		Name: input.Name, Aliases: input.Aliases, Capabilities: input.Capabilities,
		ParamSize: input.ParamSize, ContextWindowSize: input.ContextWindowSize, Description: input.Description,
		LoRAs: input.LoRAs, QualityScore: input.QualityScore, Modality: input.Modality, Tags: input.Tags,
	}
	if input.Reasoning != nil {
		value := routingReasoning(*input.Reasoning)
		patch.Reasoning = &value
	}
	if input.Control != nil {
		value := routingModelExecution(*input.Control)
		patch.Execution = &value
	}
	if input.Pricing != nil {
		value := routingsnapshot.ModelPricing(*input.Pricing)
		patch.Pricing = &value
	}
	if input.Backends != nil {
		backends := make([]routingmanagement.ModelBackendInput, len(*input.Backends))
		for index, backend := range *input.Backends {
			fields, err := decodeRoutingConnectionFields(backend.ConnectionFields)
			if err != nil {
				return routingmanagement.ModelPatch{}, err
			}
			backends[index] = routingmanagement.ModelBackendInput{
				ProviderID: backend.ProviderID, InterfaceID: backend.InterfaceID, ProviderModelID: backend.ProviderModelID,
				CredentialID: backend.CredentialID, Origin: backend.BaseURL,
				ConnectionFields: fields, Weight: backend.Weight,
			}
		}
		patch.Backends = &backends
	}
	return patch, nil
}

func routingBulkImportInput(input managementapi.RoutingBulkImportRequest, namespaceID, authorityDigest string) (routingmanagement.BulkImportRequest, error) {
	fields, err := decodeRoutingConnectionFields(input.ConnectionFields)
	if err != nil {
		return routingmanagement.BulkImportRequest{}, err
	}
	selections := make([]routingmanagement.BulkModelSelection, len(input.Selections))
	for index, selection := range input.Selections {
		selections[index] = routingmanagement.BulkModelSelection{
			CatalogItemID: selection.CatalogItemID, ID: selection.ID, Name: selection.Name,
			Aliases: selection.Aliases, Capabilities: selection.Capabilities,
			ParamSize: selection.ParamSize, ContextWindowSize: selection.ContextWindowSize, Description: selection.Description,
			Reasoning: routingReasoning(selection.Reasoning), LoRAs: selection.LoRAs,
			QualityScore: selection.QualityScore, Modality: selection.Modality, Tags: selection.Tags,
			Execution: routingModelExecution(selection.Control),
			Pricing:   routingsnapshot.ModelPricing(selection.Pricing),
		}
	}
	return routingmanagement.BulkImportRequest{
		NamespaceID: namespaceID, AuthorityDigest: authorityDigest,
		CatalogRevision: input.CatalogRevision, ProviderID: input.ProviderID, InterfaceID: input.InterfaceID,
		DiscoveryClaim: input.DiscoveryClaim, CredentialID: input.CredentialID,
		Origin: input.BaseURL, ConnectionFields: fields, Weight: input.Weight, Selections: selections,
	}, nil
}

func routingEntrypointInput(input managementapi.RoutingEntrypointWrite) routingmanagement.EntrypointInput {
	rules := make([]routingmanagement.EntrypointRuleInput, len(input.Rules))
	for index, rule := range input.Rules {
		assignments := make(map[string]routingmanagement.AssignmentSetInput, len(rule.Assignments))
		for decisionID, valueSet := range rule.Assignments {
			assignmentSet := routingmanagement.AssignmentSetInput{Fallback: routingFallback(valueSet.Fallback)}
			for _, value := range valueSet.Models {
				assignmentSet.Models = append(assignmentSet.Models, routingmanagement.AssignmentInput{
					ModelID: value.ModelID, Priority: value.Priority, Weight: value.Weight, LoRAName: value.LoRAName,
					Reasoning: routingAssignmentReasoning(value.Reasoning),
				})
			}
			assignments[decisionID] = assignmentSet
		}
		rules[index] = routingmanagement.EntrypointRuleInput{
			ID: rule.ID, Name: rule.Name, Matchers: routingMatchers(rule.Matchers),
			RecipeID: rule.RecipeID, Assignments: assignments,
		}
	}
	return routingmanagement.EntrypointInput{ID: input.ID, Name: input.Name, Aliases: input.Aliases, Rules: rules}
}

func routingFallback(value *managementapi.RoutingFallbackPolicy) *routingsnapshot.FallbackPolicy {
	if value == nil {
		return nil
	}
	return &routingsnapshot.FallbackPolicy{Strategy: value.Strategy, On: append([]string(nil), value.On...)}
}

func routingReasoning(value managementapi.RoutingReasoningFamily) routingsnapshot.ReasoningFamily {
	return routingsnapshot.ReasoningFamily{Type: value.Type, Efforts: append([]string(nil), value.Efforts...)}
}

func routingAssignmentReasoning(value *managementapi.RoutingAssignmentReasoning) *routingsnapshot.AssignmentReasoning {
	if value == nil {
		return nil
	}
	return &routingsnapshot.AssignmentReasoning{Enabled: value.Enabled, Effort: value.Effort, Description: value.Description}
}

func routingMatchers(values []managementapi.RoutingMatcher) []routingsnapshot.Matcher {
	result := make([]routingsnapshot.Matcher, len(values))
	for index, value := range values {
		result[index] = routingsnapshot.Matcher{ExactPath: value.ExactPath, PathPrefix: value.PathPrefix}
		if value.Claim != nil {
			result[index].Claim = &routingsnapshot.ClaimMatcher{Name: value.Claim.Name, Value: routingsnapshot.ClaimValue{
				Kind: value.Claim.Value.Kind, String: value.Claim.Value.String,
				Boolean: value.Claim.Value.Boolean, Integer: value.Claim.Value.Integer,
			}}
		}
	}
	return result
}

func decodeRoutingConnectionFields(input map[string]json.RawMessage) (map[string]any, error) {
	fields := make(map[string]any, len(input))
	for name, raw := range input {
		if len(raw) == 0 || len(raw) > 4096 {
			return nil, fmt.Errorf("connection field %q is empty or too large", name)
		}
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		var value any
		if err := decoder.Decode(&value); err != nil {
			return nil, fmt.Errorf("decode connection field %q: %w", name, err)
		}
		if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("connection field %q has trailing JSON", name)
		}
		fields[name] = value
	}
	return fields, nil
}

func routingModelViewDTO(model routingmanagement.Model) managementapi.RoutingModelView {
	backends := make([]managementapi.RoutingModelBackendView, len(model.Current.Backends))
	for index, backend := range model.Current.Backends {
		backends[index] = managementapi.RoutingModelBackendView{
			ProviderID: backend.ProviderID, ProviderModelID: backend.ProviderModelID,
			CredentialConfigured: backend.ProviderCredentialID != "", Weight: backend.Weight,
		}
	}
	return managementapi.RoutingModelView{
		ID: model.ID, Name: model.Name, Status: string(model.Status), Revision: model.Revision,
		ModelRevision: model.Current.Revision, CatalogRevision: model.Current.CatalogRevision,
		Aliases: append([]string(nil), model.Current.Aliases...), Capabilities: append([]string(nil), model.Current.Capabilities...),
		ParamSize: model.Current.ParamSize, ContextWindowSize: model.Current.ContextWindowSize,
		Description: model.Current.Description,
		Reasoning:   routingReasoningDTO(model.Current.Reasoning), LoRAs: append([]string(nil), model.Current.LoRAs...),
		QualityScore: model.Current.QualityScore, Modality: model.Current.Modality,
		Tags:    append([]string(nil), model.Current.Tags...),
		Control: routingModelControlDTO(model.Current.Execution), Pricing: managementapi.RoutingPricing(model.Current.Pricing),
		Backends: backends, CreatedAt: model.CreatedAt, UpdatedAt: model.UpdatedAt,
	}
}

func routingModelCardViewDTO(model routingmanagement.Model) managementapi.RoutingModelCardView {
	return managementapi.RoutingModelCardView{
		ID: model.ID, Name: model.Name,
		Card: managementapi.RoutingModelCard{
			Aliases:   append([]string{}, model.Current.Aliases...),
			ParamSize: model.Current.ParamSize, ContextWindowSize: model.Current.ContextWindowSize,
			Description:  model.Current.Description,
			Capabilities: append([]string{}, model.Current.Capabilities...),
			Reasoning:    routingReasoningDTO(model.Current.Reasoning),
			LoRAs:        append([]string{}, model.Current.LoRAs...), QualityScore: model.Current.QualityScore,
			Modality: model.Current.Modality, Tags: append([]string{}, model.Current.Tags...),
		},
	}
}

func routingRecipeViewDTO(recipe routingmanagement.Recipe) managementapi.RoutingRecipeView {
	origin := recipe.Origin
	if origin == "" {
		origin = routingmanagement.RecipeOriginCustom
	}
	view := managementapi.RoutingRecipeView{
		ID: recipe.ID, Name: recipe.Name, Description: recipe.Description, Status: string(recipe.Status),
		Revision: recipe.Revision, RecipeRevision: recipe.Current.Revision,
		Origin: string(origin), Immutable: origin == routingmanagement.RecipeOriginDistribution,
		Decisions: routingDecisionsDTO(recipe.Current.Decisions),
		Document:  append(json.RawMessage(nil), recipe.Current.Document...), CreatedAt: recipe.CreatedAt, UpdatedAt: recipe.UpdatedAt,
	}
	if recipe.Provenance != nil {
		view.Provenance = &managementapi.RoutingRecipeProvenanceView{
			DistributionID:      recipe.Provenance.DistributionID,
			DistributionVersion: recipe.Provenance.DistributionVersion,
			AssetDigest:         recipe.Provenance.AssetDigest, SourceRecipeID: recipe.Provenance.SourceRecipeID,
			SourceRevision: recipe.Provenance.SourceRevision, RecipeDigest: recipe.Provenance.RecipeDigest,
			InstalledAt: recipe.Provenance.InstalledAt,
		}
	}
	return view
}

func routingEntrypointViewDTO(entrypoint routingmanagement.Entrypoint, includeTopology bool) managementapi.RoutingEntrypointView {
	view := managementapi.RoutingEntrypointView{
		ID: entrypoint.ID, Name: entrypoint.Name, Status: string(entrypoint.Status), Revision: entrypoint.Revision,
		EntrypointRevision: entrypoint.Current.Revision, Aliases: append([]string(nil), entrypoint.Current.Aliases...),
		RuleCount: entrypoint.RuleCount, AssignedModelCount: entrypoint.AssignedModelCount,
		CreatedAt: entrypoint.CreatedAt, UpdatedAt: entrypoint.UpdatedAt,
	}
	if includeTopology {
		view.Rules = routingEntrypointRulesDTO(entrypoint.Current.Rules)
	}
	return view
}

func routingModelPageDTO(page routingmanagement.Page[routingmanagement.Model], pageSize int) managementapi.RoutingModelPage {
	items := make([]managementapi.RoutingModelView, len(page.Items))
	for index := range page.Items {
		items[index] = routingModelViewDTO(page.Items[index])
	}
	return managementapi.RoutingModelPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageSize,
	}}
}

func routingModelCardPageDTO(page routingmanagement.Page[routingmanagement.Model], pageSize int) managementapi.RoutingModelCardPage {
	items := make([]managementapi.RoutingModelCardView, len(page.Items))
	for index := range page.Items {
		items[index] = routingModelCardViewDTO(page.Items[index])
	}
	return managementapi.RoutingModelCardPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageSize,
	}}
}

func routingRecipePageDTO(page routingmanagement.Page[routingmanagement.Recipe], pageSize int) managementapi.RoutingRecipePage {
	items := make([]managementapi.RoutingRecipeView, len(page.Items))
	for index := range page.Items {
		items[index] = routingRecipeViewDTO(page.Items[index])
	}
	return managementapi.RoutingRecipePage{Data: items, Page: managementapi.PageInfo{
		NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageSize,
	}}
}

func routingEntrypointPageDTO(page routingmanagement.Page[routingmanagement.Entrypoint], pageSize int) managementapi.RoutingEntrypointPage {
	items := make([]managementapi.RoutingEntrypointView, len(page.Items))
	for index := range page.Items {
		items[index] = routingEntrypointViewDTO(page.Items[index], false)
	}
	return managementapi.RoutingEntrypointPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageSize,
	}}
}

func routingResolveResponseDTO(resolution routingsnapshot.Resolution) managementapi.RoutingResolveResponse {
	response := managementapi.RoutingResolveResponse{Outcome: string(resolution.Outcome)}
	if resolution.Entrypoint != nil {
		response.Entrypoint = &managementapi.RoutingResolvedEntrypoint{
			ID: resolution.Entrypoint.ID, Revision: resolution.Entrypoint.Revision,
			Name: resolution.Entrypoint.Name, Aliases: append([]string(nil), resolution.Entrypoint.Aliases...),
		}
	}
	if resolution.Rule != nil {
		value := routingEntrypointRuleDTO(*resolution.Rule)
		response.Rule = &value
	}
	if resolution.Recipe != nil {
		response.Recipe = &managementapi.RoutingResolvedRecipe{
			ID: resolution.Recipe.ID, Revision: resolution.Recipe.Revision, Name: resolution.Recipe.Name,
			Decisions: routingDecisionsDTO(resolution.Recipe.Decisions),
			Document:  append(json.RawMessage(nil), resolution.Recipe.Document...),
		}
	}
	return response
}

func routingProbeResponseDTO(result routingmanagement.ProbeResult) managementapi.RoutingProbeResponse {
	return managementapi.RoutingProbeResponse{
		Reachable:           result.Available,
		LatencyMilliseconds: result.Latency.Milliseconds(), CheckedAt: result.CheckedAt,
	}
}

func routingReasoningDTO(value routingsnapshot.ReasoningFamily) managementapi.RoutingReasoningFamily {
	return managementapi.RoutingReasoningFamily{Type: value.Type, Efforts: append([]string(nil), value.Efforts...)}
}

func routingModelExecution(value managementapi.RoutingModelControl) routingsnapshot.ModelExecution {
	return routingsnapshot.ModelExecution{
		MaxRetries:     value.Retry.Count,
		RetryOn:        append([]string(nil), value.Retry.On...),
		RequestTimeout: value.Timeout.Request,
		StreamTimeout:  value.Timeout.Stream,
	}
}

func routingModelControlDTO(value routingsnapshot.ModelExecution) managementapi.RoutingModelControl {
	return managementapi.RoutingModelControl{
		Retry: managementapi.RoutingModelRetryControl{
			Count: value.MaxRetries,
			On:    append([]string(nil), value.RetryOn...),
		},
		Timeout: managementapi.RoutingModelTimeoutControl{
			Request: value.RequestTimeout,
			Stream:  value.StreamTimeout,
		},
	}
}

func routingDecisionsDTO(values []routingsnapshot.Decision) []managementapi.RoutingDecision {
	result := make([]managementapi.RoutingDecision, len(values))
	for index, value := range values {
		result[index] = managementapi.RoutingDecision{
			ID: value.ID, Name: value.Name, DispatchCardinality: string(value.DispatchCardinality),
		}
	}
	return result
}

func routingEntrypointRulesDTO(values []routingsnapshot.EntrypointRule) []managementapi.RoutingEntrypointRule {
	result := make([]managementapi.RoutingEntrypointRule, len(values))
	for index, value := range values {
		result[index] = routingEntrypointRuleDTO(value)
	}
	return result
}

func routingEntrypointRuleDTO(value routingsnapshot.EntrypointRule) managementapi.RoutingEntrypointRule {
	assignments := make(map[string]managementapi.RoutingAssignmentSet, len(value.Assignments))
	for decisionID, valueSet := range value.Assignments {
		assignmentSet := managementapi.RoutingAssignmentSet{}
		if valueSet.Fallback != nil {
			assignmentSet.Fallback = &managementapi.RoutingFallbackPolicy{
				Strategy: valueSet.Fallback.Strategy, On: append([]string(nil), valueSet.Fallback.On...),
			}
		}
		for _, assignment := range valueSet.Models {
			item := managementapi.RoutingAssignment{
				ModelID: assignment.ModelID, ModelRevision: assignment.ModelRevision,
				Priority: assignment.Priority, Weight: assignment.Weight, LoRAName: assignment.LoRAName,
			}
			if assignment.Reasoning != nil {
				item.Reasoning = &managementapi.RoutingAssignmentReasoning{
					Enabled: assignment.Reasoning.Enabled,
					Effort:  assignment.Reasoning.Effort, Description: assignment.Reasoning.Description,
				}
			}
			assignmentSet.Models = append(assignmentSet.Models, item)
		}
		assignments[decisionID] = assignmentSet
	}
	return managementapi.RoutingEntrypointRule{
		ID: value.ID, Name: value.Name,
		Matchers: routingMatchersDTO(value.Matchers), RecipeID: value.RecipeID,
		RecipeRevision: value.RecipeRevision, Assignments: assignments,
	}
}

func routingMatchersDTO(values []routingsnapshot.Matcher) []managementapi.RoutingMatcher {
	result := make([]managementapi.RoutingMatcher, len(values))
	for index, value := range values {
		result[index] = managementapi.RoutingMatcher{ExactPath: value.ExactPath, PathPrefix: value.PathPrefix}
		if value.Claim != nil {
			result[index].Claim = &managementapi.RoutingClaimMatcher{
				Name:  value.Claim.Name,
				Value: routingClaimValueDTO(value.Claim.Value),
			}
		}
	}
	return result
}
