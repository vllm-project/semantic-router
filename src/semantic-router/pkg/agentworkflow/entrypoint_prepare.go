package agentworkflow

import (
	"context"
	"encoding/json"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type assignmentInput struct {
	ModelID   string                               `json:"modelId"`
	Priority  int                                  `json:"priority,omitempty"`
	Weight    string                               `json:"weight,omitempty"`
	LoRAName  string                               `json:"loraName,omitempty"`
	Reasoning *routingsnapshot.AssignmentReasoning `json:"reasoning,omitempty"`
}

type assignmentSetInput struct {
	Models   []assignmentInput               `json:"models" jsonschema:"minItems=1,maxItems=32"`
	Fallback *routingsnapshot.FallbackPolicy `json:"fallback,omitempty"`
}

type entrypointRuleInput struct {
	ID          string                        `json:"id,omitempty"`
	Name        string                        `json:"name"`
	Matchers    []routingsnapshot.Matcher     `json:"matchers,omitempty"`
	RecipeID    string                        `json:"recipeId"`
	Assignments map[string]assignmentSetInput `json:"assignments"`
}

type entrypointPrepareInput struct {
	EntrypointID     string                `json:"entrypointId,omitempty"`
	ExpectedRevision int64                 `json:"expectedRevision,omitempty" jsonschema:"minimum=0"`
	Name             string                `json:"name"`
	Aliases          []string              `json:"aliases" jsonschema:"minItems=1,maxItems=64"`
	Rules            []entrypointRuleInput `json:"rules" jsonschema:"minItems=1,maxItems=64"`
}

type entrypointPrepareOutput struct {
	EntrypointID     string `json:"entrypointId"`
	Name             string `json:"name"`
	ResourceRevision int64  `json:"resourceRevision"`
	ContentRevision  int64  `json:"contentRevision"`
	RuleCount        int    `json:"ruleCount"`
	AssignedModels   int    `json:"assignedModels"`
	DesiredRevision  int64  `json:"desiredRevision"`
	OperationID      string `json:"operationId"`
	Replayed         bool   `json:"replayed"`
}

func (provider *Provider) prepareEntrypoint(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input entrypointPrepareInput
	if err := json.Unmarshal(raw, &input); err != nil || input.Name == "" ||
		len(input.Aliases) == 0 || len(input.Rules) == 0 || input.ExpectedRevision < 0 ||
		(input.EntrypointID == "" && input.ExpectedRevision != 0) {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	if input.EntrypointID == "" {
		if err := provider.authorizeNamespace(
			ctx, invocation, accesscontrol.PermissionRoutingManage,
		); err != nil {
			return agentmanagement.ToolResult{}, err
		}
	} else if input.ExpectedRevision == 0 {
		if err := provider.authorizeNamespace(
			ctx, invocation, accesscontrol.PermissionRoutingManage,
		); err != nil {
			return agentmanagement.ToolResult{}, err
		}
	} else {
		current, err := provider.routing.GetEntrypoint(ctx, invocation.NamespaceID, input.EntrypointID)
		if err != nil {
			return agentmanagement.ToolResult{}, mapRoutingError(err)
		}
		if current.Revision != input.ExpectedRevision {
			return agentmanagement.ToolResult{}, agentmanagement.ErrConflict
		}
		if err := provider.authorizeResources(
			ctx, invocation, accesscontrol.PermissionRoutingManage,
			accesscontrol.ScopeResourceEntrypoint, current.ID,
		); err != nil {
			return agentmanagement.ToolResult{}, err
		}
	}
	converted, recipeIDs, modelIDs, prepareEntrypointErr := provider.resolveEntrypointInput(
		ctx, invocation, input,
	)
	if prepareEntrypointErr != nil {
		return agentmanagement.ToolResult{}, prepareEntrypointErr
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingRead,
		accesscontrol.ScopeResourceRecipe, recipeIDs...,
	); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingRead,
		accesscontrol.ScopeResourceModel, modelIDs...,
	); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	command, prepareEntrypointErr := provider.bindCommand(invocation, agentmanagement.ToolEntrypointPrepare, raw)
	if prepareEntrypointErr != nil {
		return agentmanagement.ToolResult{}, prepareEntrypointErr
	}
	var entrypoint routingmanagement.Entrypoint
	var receipt routingmanagement.RevisionReceipt
	if input.ExpectedRevision == 0 {
		entrypoint, receipt, prepareEntrypointErr = provider.routing.CreateEntrypoint(
			ctx, invocation.NamespaceID, converted,
			routingMutation(invocation, command, "prepare Entrypoint draft from Agent Builder"),
		)
	} else {
		entrypoint, receipt, prepareEntrypointErr = provider.routing.UpdateEntrypoint(
			ctx, invocation.NamespaceID, input.EntrypointID, input.ExpectedRevision, converted,
			routingMutation(invocation, command, "update Entrypoint draft from Agent Builder"),
		)
	}
	if prepareEntrypointErr != nil {
		return agentmanagement.ToolResult{}, mapRoutingError(prepareEntrypointErr)
	}
	value, prepareEntrypointErr := json.Marshal(entrypointPrepareOutput{
		EntrypointID: entrypoint.ID, Name: entrypoint.Name,
		ResourceRevision: entrypoint.Revision, ContentRevision: entrypoint.Current.Revision,
		RuleCount: entrypoint.RuleCount, AssignedModels: entrypoint.AssignedModelCount,
		DesiredRevision: receipt.DesiredRevision, OperationID: receipt.OperationID,
		Replayed: receipt.Replayed,
	})
	if prepareEntrypointErr != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	return agentmanagement.ToolResult{Value: value}, nil
}

func (provider *Provider) resolveEntrypointInput(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	input entrypointPrepareInput,
) (routingmanagement.EntrypointInput, []string, []string, error) {
	result := routingmanagement.EntrypointInput{
		ID: input.EntrypointID, Name: input.Name,
		Aliases: append([]string(nil), input.Aliases...),
	}
	recipes := make(map[string]struct{})
	models := make(map[string]struct{})
	for _, rule := range input.Rules {
		if rule.RecipeID == "" || len(rule.Assignments) == 0 {
			return routingmanagement.EntrypointInput{}, nil, nil, agentmanagement.ErrInvalid
		}
		recipe, err := provider.routing.GetRecipe(ctx, invocation.NamespaceID, rule.RecipeID)
		if err != nil || recipe.Status == routingmanagement.StatusDisabled ||
			recipe.Status == routingmanagement.StatusDeleted {
			if err != nil {
				return routingmanagement.EntrypointInput{}, nil, nil, mapRoutingError(err)
			}
			return routingmanagement.EntrypointInput{}, nil, nil, agentmanagement.ErrNotFound
		}
		recipes[recipe.ID] = struct{}{}
		convertedRule := routingmanagement.EntrypointRuleInput{
			ID: rule.ID, Name: rule.Name,
			Matchers:    append([]routingsnapshot.Matcher(nil), rule.Matchers...),
			RecipeID:    recipe.ID,
			Assignments: make(map[string]routingmanagement.AssignmentSetInput, len(rule.Assignments)),
		}
		for decisionID, set := range rule.Assignments {
			if decisionID == "" || len(set.Models) == 0 {
				return routingmanagement.EntrypointInput{}, nil, nil, agentmanagement.ErrInvalid
			}
			convertedSet := routingmanagement.AssignmentSetInput{Fallback: set.Fallback}
			for _, assignment := range set.Models {
				model, err := provider.routing.GetModel(ctx, invocation.NamespaceID, assignment.ModelID)
				if err != nil || model.Status == routingmanagement.StatusDisabled ||
					model.Status == routingmanagement.StatusDeleted {
					if err != nil {
						return routingmanagement.EntrypointInput{}, nil, nil, mapRoutingError(err)
					}
					return routingmanagement.EntrypointInput{}, nil, nil, agentmanagement.ErrNotFound
				}
				models[model.ID] = struct{}{}
				convertedSet.Models = append(convertedSet.Models, routingmanagement.AssignmentInput{
					ModelID: model.ID, Priority: assignment.Priority, Weight: assignment.Weight,
					LoRAName: assignment.LoRAName, Reasoning: assignment.Reasoning,
				})
			}
			convertedRule.Assignments[decisionID] = convertedSet
		}
		result.Rules = append(result.Rules, convertedRule)
	}
	return result, sortedKeys(recipes), sortedKeys(models), nil
}

func sortedKeys(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}
