package postgres

import (
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (planner *manifestPlanner) planModels() {
	for index := range planner.working.Models {
		planner.planModel(&planner.working.Models[index])
	}
}

func (planner *manifestPlanner) planModel(model *routingsnapshot.Model) {
	incomingID := model.ID
	current, found := planner.modelsByName[model.Name]
	if found {
		model.ID, model.Revision = current.ID, current.Current.Revision
		alignManifestBackendIDs(model, current.Current)
		if !equalManifestValue(*model, current.Current) {
			model.Revision++
			planner.plan.writeModels[current.ID] = true
			planner.plan.diff.Models.Update = append(planner.plan.diff.Models.Update, current.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, current.ID)
		} else if current.Status != routingmanagement.StatusActive {
			planner.plan.diff.Models.Update = append(planner.plan.diff.Models.Update, current.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, current.ID)
		}
		planner.plan.models[current.ID] = current
	} else {
		model.ID, model.Revision = generatedManifestID("mdl"), 1
		planner.plan.diff.Models.Create = append(planner.plan.diff.Models.Create, model.Name)
		planner.plan.targetIDs = append(planner.plan.targetIDs, model.ID)
		for index := range model.Backends {
			model.Backends[index].ID = uuid.NewString()
		}
		planner.plan.writeModels[model.ID] = true
	}
	planner.modelIDs[incomingID] = model.ID
	planner.modelRevisions[incomingID] = model.Revision
}

func alignManifestBackendIDs(incoming *routingsnapshot.Model, current routingsnapshot.Model) {
	for index := range incoming.Backends {
		if index < len(current.Backends) {
			incoming.Backends[index].ID = current.Backends[index].ID
			continue
		}
		incoming.Backends[index].ID = uuid.NewString()
	}
}

func (planner *manifestPlanner) planRecipes() error {
	for index := range planner.working.Recipes {
		if err := planner.planRecipe(&planner.working.Recipes[index]); err != nil {
			return err
		}
	}
	return nil
}

func (planner *manifestPlanner) planRecipe(recipe *routingsnapshot.Recipe) error {
	incomingID := recipe.ID
	current, found := planner.recipesByName[recipe.Name]
	if found {
		recipe.ID, recipe.Revision = current.ID, current.Current.Revision
	} else {
		recipe.ID, recipe.Revision = generatedManifestID("rcp"), 1
	}
	planner.remapRecipeDecisionIDs(incomingID, recipe, current.Current)
	if found {
		changed := !equalManifestValue(*recipe, current.Current)
		if changed && current.Origin == routingmanagement.RecipeOriginDistribution {
			return routingmanagement.ErrImmutable
		}
		if changed {
			recipe.Revision++
			planner.plan.writeRecipes[current.ID] = true
			planner.plan.diff.Recipes.Update = append(planner.plan.diff.Recipes.Update, current.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, current.ID)
		} else if current.Status != routingmanagement.StatusActive {
			planner.plan.diff.Recipes.Update = append(planner.plan.diff.Recipes.Update, current.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, current.ID)
		}
		planner.plan.recipes[current.ID] = current
	} else {
		planner.plan.writeRecipes[recipe.ID] = true
		planner.plan.diff.Recipes.Create = append(planner.plan.diff.Recipes.Create, recipe.Name)
		planner.plan.targetIDs = append(planner.plan.targetIDs, recipe.ID)
	}
	planner.recipeIDs[incomingID] = recipe.ID
	planner.recipeRevisions[incomingID] = recipe.Revision
	return nil
}

func (planner *manifestPlanner) remapRecipeDecisionIDs(
	incomingRecipeID string,
	recipe *routingsnapshot.Recipe,
	current routingsnapshot.Recipe,
) {
	currentDecisions := make(map[string]string, len(current.Decisions))
	for _, decision := range current.Decisions {
		currentDecisions[decision.Name] = decision.ID
	}
	for index := range recipe.Decisions {
		incomingDecisionID := recipe.Decisions[index].ID
		decisionID := currentDecisions[recipe.Decisions[index].Name]
		if decisionID == "" {
			decisionID = generatedManifestID("dec")
		}
		recipe.Decisions[index].ID = decisionID
		planner.decisionIDs[incomingRecipeID+"\x00"+incomingDecisionID] = decisionID
	}
}

func (planner *manifestPlanner) planEntrypoints() {
	for index := range planner.working.Entrypoints {
		planner.planEntrypoint(&planner.working.Entrypoints[index])
	}
}

func (planner *manifestPlanner) planEntrypoint(entrypoint *routingsnapshot.Entrypoint) {
	current, found := planner.entriesByName[entrypoint.Name]
	if found {
		entrypoint.ID, entrypoint.Revision = current.ID, current.Current.Revision
	} else {
		entrypoint.ID, entrypoint.Revision = generatedManifestID("ep"), 1
	}
	planner.remapEntrypointRules(entrypoint, current.Current)
	if found {
		if !equalManifestValue(*entrypoint, current.Current) {
			entrypoint.Revision++
			planner.plan.writeEntries[current.ID] = true
			planner.plan.diff.Entrypoints.Update = append(planner.plan.diff.Entrypoints.Update, current.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, current.ID)
		} else if current.Status != routingmanagement.StatusActive {
			planner.plan.diff.Entrypoints.Update = append(planner.plan.diff.Entrypoints.Update, current.Name)
			planner.plan.targetIDs = append(planner.plan.targetIDs, current.ID)
		}
		planner.plan.entries[current.ID] = current
		return
	}
	planner.plan.writeEntries[entrypoint.ID] = true
	planner.plan.diff.Entrypoints.Create = append(planner.plan.diff.Entrypoints.Create, entrypoint.Name)
	planner.plan.targetIDs = append(planner.plan.targetIDs, entrypoint.ID)
}

func (planner *manifestPlanner) remapEntrypointRules(
	entrypoint *routingsnapshot.Entrypoint,
	current routingsnapshot.Entrypoint,
) {
	currentRules := make(map[string]string, len(current.Rules))
	for _, rule := range current.Rules {
		currentRules[rule.Name] = rule.ID
	}
	for ruleIndex := range entrypoint.Rules {
		rule := &entrypoint.Rules[ruleIndex]
		incomingRecipeID := rule.RecipeID
		rule.ID = currentRules[rule.Name]
		if rule.ID == "" {
			rule.ID = generatedManifestID("rule")
		}
		rule.RecipeID = planner.recipeIDs[incomingRecipeID]
		rule.RecipeRevision = planner.recipeRevisions[incomingRecipeID]
		rule.Assignments = planner.remapAssignmentSets(incomingRecipeID, rule.Assignments)
	}
}

func (planner *manifestPlanner) remapAssignmentSets(
	incomingRecipeID string,
	assignments map[string]routingsnapshot.AssignmentSet,
) map[string]routingsnapshot.AssignmentSet {
	mapped := make(map[string]routingsnapshot.AssignmentSet, len(assignments))
	for incomingDecisionID, set := range assignments {
		for index := range set.Models {
			incomingModelID := set.Models[index].ModelID
			set.Models[index].ModelID = planner.modelIDs[incomingModelID]
			set.Models[index].ModelRevision = planner.modelRevisions[incomingModelID]
		}
		mapped[planner.decisionIDs[incomingRecipeID+"\x00"+incomingDecisionID]] = set
	}
	return mapped
}
