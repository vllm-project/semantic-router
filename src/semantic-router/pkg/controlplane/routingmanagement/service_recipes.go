package routingmanagement

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (service *Service) GetRecipe(ctx context.Context, namespaceID, id string) (Recipe, error) {
	return service.store.GetRecipe(ctx, namespaceID, id)
}

func (service *Service) CreateRecipe(
	ctx context.Context, namespaceID string, input RecipeInput, mutation MutationContext,
) (Recipe, RevisionReceipt, error) {
	if input.ID == "" {
		input.ID = generatedID("rcp")
	}
	recipe, err := compileRecipe(input, 1)
	if err != nil {
		return Recipe{}, RevisionReceipt{}, err
	}
	return service.store.CreateRecipe(ctx, namespaceID, input.Description, recipe, mutation)
}

func (service *Service) UpdateRecipe(
	ctx context.Context, namespaceID, id string, expected int64, input RecipeInput, mutation MutationContext,
) (Recipe, RevisionReceipt, error) {
	current, err := service.store.GetRecipe(ctx, namespaceID, id)
	if err != nil {
		return Recipe{}, RevisionReceipt{}, err
	}
	if current.Revision != expected {
		return Recipe{}, RevisionReceipt{}, ErrConflict
	}
	input.ID = id
	recipe, err := compileRecipe(input, current.Current.Revision+1)
	if err != nil {
		return Recipe{}, RevisionReceipt{}, err
	}
	return service.store.UpdateRecipe(ctx, namespaceID, id, expected, input.Description, recipe, mutation)
}

func (service *Service) DeleteRecipe(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (RevisionReceipt, error) {
	return service.store.DeleteRecipe(ctx, namespaceID, id, expected, mutation)
}

func compileRecipe(input RecipeInput, revision int64) (routingsnapshot.Recipe, error) {
	if err := validateIdentity(input.ID, input.Name); err != nil || revision <= 0 || !canonicalText(input.Description, 0, 2048) {
		return routingsnapshot.Recipe{}, fmt.Errorf("%w: Recipe metadata is invalid", ErrInvalid)
	}
	document, decisions, err := CompileRecipeDocument(input.ID, input.Document)
	if err != nil {
		return routingsnapshot.Recipe{}, err
	}
	recipe := routingsnapshot.Recipe{
		ID: input.ID, Revision: revision, Name: input.Name, Description: input.Description,
		Decisions: decisions, Document: document,
	}
	// Empty Recipes are valid control-plane drafts. The runtime compiler remains
	// the publication gate and rejects an Entrypoint whose Recipe has no complete
	// decision-to-Model chain.
	if len(decisions) == 0 {
		return recipe, nil
	}
	if _, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: "validation", Revision: 1, Recipes: []routingsnapshot.Recipe{recipe},
	}); err != nil {
		return routingsnapshot.Recipe{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return recipe, nil
}
