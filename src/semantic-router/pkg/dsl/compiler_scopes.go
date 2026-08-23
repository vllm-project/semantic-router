package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// compileScopes lowers Recipe-local programs after the shared Model catalog.
// Every Recipe receives a new Compiler instance, which makes accidental
// cross-Recipe symbol reuse structurally impossible.
func (c *Compiler) compileScopes() {
	if err := c.config.Strategy.Validate(); err != nil {
		c.errors = append(c.errors, err)
	}
	recipes := c.compileRecipes()
	c.compileEntrypoints(recipes)
}

type compiledRecipeScope struct {
	recipeIndex     int
	decisionsByName map[string]string
}

func (c *Compiler) compileRecipes() map[string]compiledRecipeScope {
	c.config.Recipes = nil
	recipeNames := make(map[config.RecipeName]struct{}, len(c.prog.Recipes))
	compiled := make(map[string]compiledRecipeScope, len(c.prog.Recipes))
	for _, recipe := range c.prog.Recipes {
		name := config.RecipeName(recipe.Name)
		if name == "" {
			c.addError(recipe.Pos, "RECIPE name cannot be empty")
			continue
		}
		if _, exists := recipeNames[name]; exists {
			c.addError(recipe.Pos, "duplicate RECIPE %q", name)
			continue
		}
		recipeNames[name] = struct{}{}

		recipeID := config.DeterministicRoutingResourceID("rcp", recipe.Name)
		if name == config.DefaultRecipeName {
			recipeID = "rcp_default"
		}
		child := newScopedCompiler(recipe.Program)
		child.compile()
		if len(child.errors) == 0 {
			if _, err := config.MLSelectionConfigForRoutingProfile(child.config); err != nil {
				child.errors = append(child.errors, err)
			}
		}
		for index := range child.config.Decisions {
			decision := &child.config.Decisions[index]
			decision.ID = config.DeterministicRoutingResourceID("dec", recipeID, decision.Name)
		}
		for _, decision := range child.config.Decisions {
			if recipeDecisionSelectsPhysicalModels(decision) {
				child.addError(recipe.Pos, "Decision %q selects a physical Model; assign Models in ENTRYPOINT instead", decision.Name)
			}
		}
		if err := child.config.Strategy.Validate(); err != nil {
			child.errors = append(child.errors, err)
		}
		for _, err := range child.errors {
			c.errors = append(c.errors, fmt.Errorf("RECIPE %q: %w", name, err))
		}
		c.config.Recipes = append(c.config.Recipes, config.RoutingRecipe{
			ID:          recipeID,
			Revision:    1,
			Name:        name,
			Description: recipe.Description,
			Profile: config.RoutingProfile{
				Signals:     child.config.Signals,
				Projections: child.config.Projections,
				Decisions:   child.config.Decisions,
				Strategy:    child.config.Strategy,
			},
		})
		compiledRecipe := &c.config.Recipes[len(c.config.Recipes)-1]
		decisions := make(map[string]string, len(compiledRecipe.Profile.Decisions))
		for _, decision := range compiledRecipe.Profile.Decisions {
			decisions[decision.Name] = decision.ID
		}
		compiled[recipe.Name] = compiledRecipeScope{recipeIndex: len(c.config.Recipes) - 1, decisionsByName: decisions}
	}
	return compiled
}

func recipeDecisionSelectsPhysicalModels(decision config.Decision) bool {
	if len(decision.ModelRefs) > 0 {
		return true
	}
	for _, iteration := range decision.CandidateIterations {
		if len(iteration.Models) > 0 {
			return true
		}
	}
	if decision.Algorithm == nil {
		return false
	}
	if value := decision.Algorithm.Fusion; value != nil && (value.Model != "" || len(value.AnalysisModels) > 0 || len(value.AnalysisOverrides) > 0) {
		return true
	}
	if value := decision.Algorithm.Workflows; value != nil {
		if value.Planner.Model != "" || value.Final.Model != "" {
			return true
		}
		for _, role := range value.Roles {
			if len(role.Models) > 0 {
				return true
			}
		}
	}
	if value := decision.Algorithm.ReMoM; value != nil && value.SynthesisModel != "" {
		return true
	}
	return decision.Algorithm.Prompt != nil && decision.Algorithm.Prompt.Model != ""
}

func (c *Compiler) compileEntrypoints(recipes map[string]compiledRecipeScope) {
	seenModels := make(map[string]struct{})
	modelsByName := make(map[string]struct {
		id       string
		revision int64
	}, len(c.config.ModelConfig))
	for name, params := range c.config.ModelConfig {
		modelsByName[name] = struct {
			id       string
			revision int64
		}{id: params.ResourceID, revision: params.ResourceRevision}
	}
	for _, entrypoint := range c.prog.Entrypoints {
		aliases := stableEntrypointAliases(entrypoint.Name, entrypoint.Aliases)
		for _, modelName := range aliases {
			if modelName == "" {
				c.addError(entrypoint.Pos, "ENTRYPOINT name and aliases cannot contain an empty value")
				continue
			}
			if _, exists := seenModels[modelName]; exists {
				c.addError(entrypoint.Pos, "entrypoint model %q is mapped more than once", modelName)
				continue
			}
			seenModels[modelName] = struct{}{}
		}
		compiled := config.EntrypointMapping{
			ID:         config.DeterministicRoutingResourceID("ep", entrypoint.Name),
			Revision:   1,
			Name:       entrypoint.Name,
			ModelNames: aliases,
		}
		for _, rule := range authoringEntrypointRules(entrypoint) {
			if rule == nil {
				c.addError(entrypoint.Pos, "ENTRYPOINT contains an empty rule")
				continue
			}
			recipeScope, exists := recipes[rule.Recipe]
			if !exists {
				c.addError(rule.Pos, "ENTRYPOINT rule references unknown Recipe %q", rule.Recipe)
				continue
			}
			recipe := &c.config.Recipes[recipeScope.recipeIndex]
			compiledRule := config.EntrypointRule{
				ID:      config.DeterministicRoutingResourceID("rule", compiled.ID, rule.Name),
				Name:    rule.Name,
				Matches: compileEntrypointMatches(rule.Matches),
				Action: config.EntrypointRuleAction{
					RecipeID:       recipe.ID,
					RecipeRevision: recipe.Revision,
					Recipe:         recipe.Name,
					Assignments:    compileEntrypointAssignments(c, rule, recipeScope.decisionsByName, modelsByName),
				},
			}
			compiled.Rules = append(compiled.Rules, compiledRule)
		}
		c.config.Entrypoints = append(c.config.Entrypoints, compiled)
	}
	if err := c.config.PrepareEntrypointRecipes(); err != nil {
		c.errors = append(c.errors, err)
	}
}

func stableEntrypointAliases(name string, aliases []string) []string {
	result := make([]string, 0, len(aliases)+1)
	seen := make(map[string]struct{}, len(aliases)+1)
	for _, value := range append([]string{name}, aliases...) {
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func authoringEntrypointRules(entrypoint *EntrypointDecl) []*EntrypointRuleDecl {
	if len(entrypoint.Rules) > 0 {
		return entrypoint.Rules
	}
	return []*EntrypointRuleDecl{{
		Name: "default", Recipe: entrypoint.Recipe,
		Assignments: entrypoint.Assignments, Pos: entrypoint.Pos,
	}}
}

func compileEntrypointMatches(input []*EntrypointMatchDecl) []config.EntrypointMatch {
	output := make([]config.EntrypointMatch, 0, len(input))
	for _, match := range input {
		if match == nil {
			continue
		}
		compiled := config.EntrypointMatch{}
		if match.Claim != nil {
			value := config.EntrypointClaimValue{}
			switch exact := match.Claim.Exact.(type) {
			case StringValue:
				value.Kind, value.String = "string", exact.V
			case BoolValue:
				value.Kind, value.Boolean = "boolean", exact.V
			case IntValue:
				value.Kind, value.Integer = "integer", int64(exact.V)
			}
			compiled.Claim = &config.EntrypointClaimMatch{Name: match.Claim.Name, Value: value}
		}
		if match.Path != nil {
			compiled.Path = &config.EntrypointPathMatch{Exact: match.Path.Exact, Prefix: match.Path.Prefix}
		}
		output = append(output, compiled)
	}
	return output
}

func compileEntrypointAssignments(
	c *Compiler,
	rule *EntrypointRuleDecl,
	decisionsByName map[string]string,
	modelsByName map[string]struct {
		id       string
		revision int64
	},
) map[string]config.RoutingAssignmentSet {
	output := make(map[string]config.RoutingAssignmentSet, len(rule.Assignments))
	for decisionName, assignmentSet := range rule.Assignments {
		if assignmentSet == nil {
			continue
		}
		decisionID, exists := decisionsByName[decisionName]
		if !exists {
			c.addError(rule.Pos, "ENTRYPOINT assignment references unknown Decision %q", decisionName)
			continue
		}
		compiledSet := config.RoutingAssignmentSet{}
		for _, assignment := range assignmentSet.Models {
			if assignment == nil {
				continue
			}
			model, exists := modelsByName[assignment.Model]
			if !exists {
				c.addError(assignment.Pos, "ENTRYPOINT assignment references unknown Model %q", assignment.Model)
				continue
			}
			compiled := config.RoutingModelAssignment{
				ModelID:       model.id,
				ModelRevision: model.revision,
				ModelName:     assignment.Model,
				Priority:      assignment.Priority,
				Weight:        assignment.Weight,
				LoRAName:      assignment.LoRAName,
			}
			if compiled.Weight == "" {
				compiled.Weight = "1"
			}
			if assignment.Reasoning != nil {
				compiled.Reasoning = &config.RoutingAssignmentReasoning{
					Enabled:     assignment.Reasoning.Enabled,
					Effort:      assignment.Reasoning.Effort,
					Description: assignment.Reasoning.Description,
				}
			}
			compiledSet.Models = append(compiledSet.Models, compiled)
		}
		if assignmentSet.Fallback != nil {
			compiledSet.Fallback = &config.RoutingFallbackPolicy{
				Strategy: assignmentSet.Fallback.Strategy,
				On:       append([]string(nil), assignmentSet.Fallback.On...),
			}
		}
		output[decisionID] = compiledSet
	}
	return output
}

func newScopedCompiler(prog *Program) *Compiler {
	defaults := config.DefaultGlobalConfig()
	c := &Compiler{
		prog: prog,
		config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ModelSelection: defaults.ModelSelection,
			},
		},
		pluginTemplates: make(map[string]*PluginDecl),
	}
	c.config.Strategy = config.RoutingStrategy(prog.Strategy)
	return c
}
