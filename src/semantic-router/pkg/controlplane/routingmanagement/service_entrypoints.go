package routingmanagement

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (service *Service) GetEntrypoint(ctx context.Context, namespaceID, id string) (Entrypoint, error) {
	return service.store.GetEntrypoint(ctx, namespaceID, id)
}

func (service *Service) CreateEntrypoint(
	ctx context.Context, namespaceID string, input EntrypointInput, mutation MutationContext,
) (Entrypoint, RevisionReceipt, error) {
	if input.ID == "" {
		input.ID = generatedID("ep")
	}
	entrypoint, err := service.compileEntrypoint(ctx, namespaceID, input, 1)
	if err != nil {
		return Entrypoint{}, RevisionReceipt{}, err
	}
	return service.store.CreateEntrypoint(ctx, namespaceID, entrypoint, mutation)
}

func (service *Service) UpdateEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, input EntrypointInput, mutation MutationContext,
) (Entrypoint, RevisionReceipt, error) {
	current, err := service.store.GetEntrypoint(ctx, namespaceID, id)
	if err != nil {
		return Entrypoint{}, RevisionReceipt{}, err
	}
	if current.Revision != expected {
		return Entrypoint{}, RevisionReceipt{}, ErrConflict
	}
	input.ID = id
	entrypoint, err := service.compileEntrypoint(ctx, namespaceID, input, current.Current.Revision+1)
	if err != nil {
		return Entrypoint{}, RevisionReceipt{}, err
	}
	return service.store.UpdateEntrypoint(ctx, namespaceID, id, expected, entrypoint, mutation)
}

func (service *Service) DeleteEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (RevisionReceipt, error) {
	return service.store.DeleteEntrypoint(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) PublishEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (*routingsnapshot.Snapshot, RevisionReceipt, error) {
	return service.store.PublishEntrypoint(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) UnpublishEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (*routingsnapshot.Snapshot, RevisionReceipt, error) {
	return service.store.UnpublishEntrypoint(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) ResolveEntrypoint(
	ctx context.Context, namespaceID, id, path string, claims map[string]routingsnapshot.ClaimValue,
) (routingsnapshot.Resolution, error) {
	snapshot, err := service.store.ActiveSnapshot(ctx, namespaceID)
	if err != nil {
		return routingsnapshot.Resolution{}, err
	}
	entrypoint, exists := snapshot.Entrypoint(id)
	if !exists || len(entrypoint.Aliases) == 0 {
		return routingsnapshot.Resolution{}, ErrNotFound
	}
	return snapshot.Resolve(routingsnapshot.ResolveInput{Alias: entrypoint.Aliases[0], Path: path, Claims: claims})
}

func (service *Service) compileEntrypoint(
	ctx context.Context, namespaceID string, input EntrypointInput, revision int64,
) (routingsnapshot.Entrypoint, error) {
	if err := validateIdentity(input.ID, input.Name); err != nil || revision <= 0 || len(input.Rules) == 0 || len(input.Rules) > 64 {
		return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: Entrypoint metadata or rules are invalid", ErrInvalid)
	}
	aliases, compileEntrypointErr := uniqueCanonical(input.Aliases, 64)
	if compileEntrypointErr != nil || len(aliases) == 0 {
		return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: Entrypoint aliases are invalid", ErrInvalid)
	}
	entrypoint := routingsnapshot.Entrypoint{ID: input.ID, Revision: revision, Name: input.Name, Aliases: aliases}
	models := make(map[string]routingsnapshot.Model)
	recipes := make(map[string]routingsnapshot.Recipe)
	seenRules := make(map[string]struct{}, len(input.Rules))
	for index, ruleInput := range input.Rules {
		if ruleInput.ID == "" {
			ruleInput.ID = generatedID("rule")
		}
		if err := validateIdentity(ruleInput.ID, ruleInput.Name); err != nil {
			return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: rule %d identity is invalid", ErrInvalid, index)
		}
		if _, duplicate := seenRules[ruleInput.ID]; duplicate {
			return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: duplicate Entrypoint rule", ErrInvalid)
		}
		seenRules[ruleInput.ID] = struct{}{}
		recipeRecord, loadErr := service.store.GetRecipe(ctx, namespaceID, ruleInput.RecipeID)
		if loadErr != nil || recipeRecord.Status == StatusDeleted || recipeRecord.Status == StatusDisabled {
			return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: rule %d Recipe is unavailable", ErrInvalid, index)
		}
		recipes[recipeRecord.ID] = recipeRecord.Current
		rule := routingsnapshot.EntrypointRule{
			ID: ruleInput.ID, Name: ruleInput.Name, Matchers: ruleInput.Matchers,
			RecipeID: recipeRecord.ID, RecipeRevision: recipeRecord.Current.Revision,
			Assignments: make(map[string]routingsnapshot.AssignmentSet, len(ruleInput.Assignments)),
		}
		for decisionID, inputSet := range ruleInput.Assignments {
			if len(inputSet.Models) == 0 || len(inputSet.Models) > 32 {
				return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: assignment %q is empty or too large", ErrInvalid, decisionID)
			}
			assignmentSet := routingsnapshot.AssignmentSet{Fallback: inputSet.Fallback}
			for _, assignmentInput := range inputSet.Models {
				modelRecord, modelErr := service.store.GetModel(ctx, namespaceID, assignmentInput.ModelID)
				if modelErr != nil || modelRecord.Status == StatusDeleted || modelRecord.Status == StatusDisabled {
					return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: assigned Model %q is unavailable", ErrInvalid, assignmentInput.ModelID)
				}
				models[modelRecord.ID] = modelRecord.Current
				assignmentSet.Models = append(assignmentSet.Models, routingsnapshot.Assignment{
					ModelID: modelRecord.ID, ModelRevision: modelRecord.Current.Revision,
					Priority: assignmentInput.Priority, Weight: assignmentInput.Weight,
					LoRAName: assignmentInput.LoRAName, Reasoning: assignmentInput.Reasoning,
				})
			}
			rule.Assignments[decisionID] = assignmentSet
		}
		entrypoint.Rules = append(entrypoint.Rules, rule)
	}
	bundle := routingsnapshot.Bundle{NamespaceID: namespaceID, Revision: 1, Entrypoints: []routingsnapshot.Entrypoint{entrypoint}}
	for _, model := range models {
		bundle.Models = append(bundle.Models, model)
	}
	for _, recipe := range recipes {
		bundle.Recipes = append(bundle.Recipes, recipe)
	}
	currency, compileEntrypointErr := service.store.NamespaceCurrency(ctx, namespaceID)
	if compileEntrypointErr != nil {
		return routingsnapshot.Entrypoint{}, compileEntrypointErr
	}
	bundle.Currency = currency
	snapshot, compileEntrypointErr := routingsnapshot.Compile(bundle)
	if compileEntrypointErr != nil {
		return routingsnapshot.Entrypoint{}, fmt.Errorf("%w: %w", ErrInvalid, compileEntrypointErr)
	}
	return snapshot.Entrypoints[0], nil
}
