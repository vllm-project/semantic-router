package postgres

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type manifestPlanner struct {
	namespaceID string
	state       manifestState
	working     routingsnapshot.Bundle
	plan        manifestPlan

	modelsByName  map[string]routingmanagement.Model
	recipesByName map[string]routingmanagement.Recipe
	entriesByName map[string]routingmanagement.Entrypoint

	modelIDs        map[string]string
	modelRevisions  map[string]int64
	recipeIDs       map[string]string
	recipeRevisions map[string]int64
	decisionIDs     map[string]string
}

func buildManifestPlanPhases(
	namespaceID string,
	source *routingsnapshot.Snapshot,
	state manifestState,
) (manifestPlan, error) {
	planner, err := newManifestPlanner(namespaceID, source, state)
	if err != nil {
		return manifestPlan{}, err
	}
	planner.planModels()
	if err := planner.planRecipes(); err != nil {
		return manifestPlan{}, err
	}
	planner.planEntrypoints()
	planner.planDisabledResources()
	return planner.compile()
}

func newManifestPlanner(
	namespaceID string,
	source *routingsnapshot.Snapshot,
	state manifestState,
) (*manifestPlanner, error) {
	if source == nil || len(source.Models) == 0 || len(source.Recipes) == 0 || len(source.Entrypoints) == 0 {
		return nil, routingmanagement.ErrManifest
	}
	if source.Currency != "" && source.Currency != state.currency {
		return nil, fmt.Errorf("%w: billing currency belongs to the Namespace and cannot be changed by routing import", routingmanagement.ErrManifest)
	}
	working, err := cloneManifestBundle(source.Bundle)
	if err != nil {
		return nil, err
	}
	planner := &manifestPlanner{
		namespaceID: namespaceID,
		state:       state,
		working:     working,
		plan: manifestPlan{
			models: map[string]routingmanagement.Model{}, recipes: map[string]routingmanagement.Recipe{},
			entries: map[string]routingmanagement.Entrypoint{}, writeModels: map[string]bool{},
			writeRecipes: map[string]bool{}, writeEntries: map[string]bool{},
		},
		modelsByName:    make(map[string]routingmanagement.Model, len(state.models)),
		recipesByName:   make(map[string]routingmanagement.Recipe, len(state.recipes)),
		entriesByName:   make(map[string]routingmanagement.Entrypoint, len(state.entrypoints)),
		modelIDs:        map[string]string{},
		modelRevisions:  map[string]int64{},
		recipeIDs:       map[string]string{},
		recipeRevisions: map[string]int64{},
		decisionIDs:     map[string]string{},
	}
	planner.indexCurrentState()
	return planner, nil
}

func cloneManifestBundle(source routingsnapshot.Bundle) (routingsnapshot.Bundle, error) {
	// Planning rewrites compiler-owned identities and reference revisions. Work
	// on a deep value copy so dry runs stay transparent and safely retryable.
	payload, err := json.Marshal(source)
	if err != nil {
		return routingsnapshot.Bundle{}, fmt.Errorf("%w: clone routing manifest: %w", routingmanagement.ErrManifest, err)
	}
	var working routingsnapshot.Bundle
	if err := json.Unmarshal(payload, &working); err != nil {
		return routingsnapshot.Bundle{}, fmt.Errorf("%w: clone routing manifest: %w", routingmanagement.ErrManifest, err)
	}
	return working, nil
}

func (planner *manifestPlanner) indexCurrentState() {
	for _, value := range planner.state.models {
		planner.modelsByName[value.Name] = value
	}
	for _, value := range planner.state.recipes {
		planner.recipesByName[value.Name] = value
	}
	for _, value := range planner.state.entrypoints {
		planner.entriesByName[value.Name] = value
	}
}

func (planner *manifestPlanner) planDisabledResources() {
	for _, value := range planner.state.models {
		if _, keep := planner.plan.models[value.ID]; !keep {
			planner.plan.diff.Models.Disable = append(planner.plan.diff.Models.Disable, value.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, value.ID)
			planner.plan.disableModels = append(planner.plan.disableModels, value.ID)
		}
	}
	for _, value := range planner.state.recipes {
		if value.Origin == routingmanagement.RecipeOriginDistribution {
			continue
		}
		if _, keep := planner.plan.recipes[value.ID]; !keep {
			planner.plan.diff.Recipes.Disable = append(planner.plan.diff.Recipes.Disable, value.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, value.ID)
			planner.plan.disableRecipes = append(planner.plan.disableRecipes, value.ID)
		}
	}
	for _, value := range planner.state.entrypoints {
		if _, keep := planner.plan.entries[value.ID]; !keep {
			planner.plan.diff.Entrypoints.Disable = append(planner.plan.diff.Entrypoints.Disable, value.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, value.ID)
			planner.plan.disableEntries = append(planner.plan.disableEntries, value.ID)
		}
	}
}

func (planner *manifestPlanner) compile() (manifestPlan, error) {
	compiled, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: planner.namespaceID,
		Revision:    max64(1, planner.state.revision+1),
		Currency:    planner.state.currency,
		Models:      planner.working.Models,
		Recipes:     planner.working.Recipes,
		Entrypoints: planner.working.Entrypoints,
	})
	if err != nil {
		return manifestPlan{}, fmt.Errorf("%w: %w", routingmanagement.ErrManifest, err)
	}
	planner.plan.snapshot = compiled
	sortManifestDiff(&planner.plan.diff)
	return planner.plan, nil
}
