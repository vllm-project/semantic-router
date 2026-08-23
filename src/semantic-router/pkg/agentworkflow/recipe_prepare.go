package agentworkflow

import (
	"context"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

type recipePrepareSchema struct {
	RecipeID         string  `json:"recipeId,omitempty"`
	ExpectedRevision int64   `json:"expectedRevision" jsonschema:"minimum=0"`
	Name             *string `json:"name,omitempty"`
	Description      *string `json:"description,omitempty"`
	Document         any     `json:"document,omitempty"`
}

type recipePrepareInput struct {
	RecipeID         string          `json:"recipeId,omitempty"`
	ExpectedRevision int64           `json:"expectedRevision"`
	Name             *string         `json:"name,omitempty"`
	Description      *string         `json:"description,omitempty"`
	Document         json.RawMessage `json:"document,omitempty"`
}

type recipeMutationOutput struct {
	RecipeID         string `json:"recipeId"`
	Name             string `json:"name"`
	ResourceRevision int64  `json:"resourceRevision"`
	ContentRevision  int64  `json:"contentRevision"`
	DesiredRevision  int64  `json:"desiredRevision"`
	OperationID      string `json:"operationId"`
	Replayed         bool   `json:"replayed"`
}

func (provider *Provider) prepareRecipe(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input recipePrepareInput
	if err := json.Unmarshal(raw, &input); err != nil || input.ExpectedRevision < 0 ||
		(input.Name == nil && input.Description == nil && len(input.Document) == 0) {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	if input.ExpectedRevision == 0 {
		if input.Name == nil || *input.Name == "" || len(input.Document) == 0 {
			return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
		}
		if err := provider.authorizeNamespace(
			ctx, invocation, accesscontrol.PermissionRoutingManage,
		); err != nil {
			return agentmanagement.ToolResult{}, err
		}
		create := routingmanagement.RecipeInput{
			ID: input.RecipeID, Name: *input.Name,
			Document: append(json.RawMessage(nil), input.Document...),
		}
		if input.Description != nil {
			create.Description = *input.Description
		}
		command, err := provider.bindCommand(
			invocation, agentmanagement.ToolRecipePrepare, raw,
		)
		if err != nil {
			return agentmanagement.ToolResult{}, err
		}
		created, receipt, err := provider.routing.CreateRecipe(
			ctx, invocation.NamespaceID, create,
			routingMutation(invocation, command, "prepare Recipe draft from Agent Builder"),
		)
		if err != nil {
			return agentmanagement.ToolResult{}, mapRoutingError(err)
		}
		return recipeResult(created, receipt)
	}
	if input.RecipeID == "" {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	current, prepareRecipeErr := provider.routing.GetRecipe(ctx, invocation.NamespaceID, input.RecipeID)
	if prepareRecipeErr != nil {
		return agentmanagement.ToolResult{}, mapRoutingError(prepareRecipeErr)
	}
	if err := provider.authorizeResources(
		ctx, invocation, accesscontrol.PermissionRoutingManage,
		accesscontrol.ScopeResourceRecipe, current.ID,
	); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if current.Revision != input.ExpectedRevision {
		return agentmanagement.ToolResult{}, agentmanagement.ErrConflict
	}
	update := routingmanagement.RecipeInput{
		ID: current.ID, Name: current.Current.Name,
		Description: current.Description,
		Document:    append(json.RawMessage(nil), current.Current.Document...),
	}
	if input.Name != nil {
		update.Name = *input.Name
	}
	if input.Description != nil {
		update.Description = *input.Description
	}
	if len(input.Document) > 0 {
		update.Document = append(json.RawMessage(nil), input.Document...)
	}
	command, prepareRecipeErr := provider.bindCommand(invocation, agentmanagement.ToolRecipePrepare, raw)
	if prepareRecipeErr != nil {
		return agentmanagement.ToolResult{}, prepareRecipeErr
	}
	updated, receipt, prepareRecipeErr := provider.routing.UpdateRecipe(
		ctx, invocation.NamespaceID, current.ID, input.ExpectedRevision, update,
		routingMutation(invocation, command, "update Recipe draft from Agent Builder"),
	)
	if prepareRecipeErr != nil {
		return agentmanagement.ToolResult{}, mapRoutingError(prepareRecipeErr)
	}
	return recipeResult(updated, receipt)
}

func recipeResult(
	value routingmanagement.Recipe, receipt routingmanagement.RevisionReceipt,
) (agentmanagement.ToolResult, error) {
	encoded, err := json.Marshal(recipeMutationOutput{
		RecipeID: value.ID, Name: value.Name,
		ResourceRevision: value.Revision, ContentRevision: value.Current.Revision,
		DesiredRevision: receipt.DesiredRevision, OperationID: receipt.OperationID,
		Replayed: receipt.Replayed,
	})
	if err != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	return agentmanagement.ToolResult{Value: encoded}, nil
}
