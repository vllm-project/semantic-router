package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// checkRoutingScopes validates every recipe with an independent symbol table.
// Only the model catalog is inherited from the parent program.
func (v *Validator) checkRoutingScopes() {
	if v.prog == nil {
		return
	}
	v.checkStrategy(v.prog.Strategy, Position{})

	recipesByName := make(map[string]map[string]bool, len(v.prog.Recipes))
	seenRecipeNames := make(map[string]Position, len(v.prog.Recipes))
	for _, recipe := range v.prog.Recipes {
		if recipe.Name == "" {
			v.addDiag(DiagError, recipe.Pos, "RECIPE name cannot be empty", nil)
			continue
		}
		if first, exists := seenRecipeNames[recipe.Name]; exists {
			v.addDiag(DiagError, recipe.Pos,
				fmt.Sprintf("Recipe %q is declared more than once (first declaration at %s)", recipe.Name, first), nil)
			continue
		}
		seenRecipeNames[recipe.Name] = recipe.Pos
		recipesByName[recipe.Name] = routeNames(recipe.Program.Routes)
		v.checkRecipeScope(recipe)
	}
	v.checkEntrypointScopes(recipesByName)
}

func routeNames(routes []*RouteDecl) map[string]bool {
	names := make(map[string]bool, len(routes))
	for _, route := range routes {
		if route != nil {
			names[route.Name] = true
		}
	}
	return names
}

func (v *Validator) checkRecipeScope(recipe *RecipeDecl) {
	// Recipe authoring is deliberately Model-free. Compile just the isolated
	// document so every physical selection surface (including algorithms) is
	// checked through the same typed contract as Compile.
	compiled := newScopedCompiler(recipe.Program)
	compiled.compile()
	for _, decision := range compiled.config.Decisions {
		if recipeDecisionSelectsPhysicalModels(decision) {
			v.addDiag(
				DiagConstraint,
				recipe.Pos,
				fmt.Sprintf("RECIPE %q: Decision %q selects a physical Model; assign Models in ENTRYPOINT instead", recipe.Name, decision.Name),
				nil,
			)
		}
	}

	scoped := recipeProgramWithSharedModels(v.prog, recipe.Program)
	child := newValidator(scoped)
	child.buildSymbolTable()
	child.checkReferences()
	child.checkConstraints()
	child.checkConflicts()
	child.checkStrategy(scoped.Strategy, recipe.Pos)
	for _, diag := range child.diagnostics {
		diag.Message = fmt.Sprintf("RECIPE %q: %s", recipe.Name, diag.Message)
		v.diagnostics = append(v.diagnostics, diag)
	}
}

func (v *Validator) checkEntrypointScopes(recipesByName map[string]map[string]bool) {
	seenModels := make(map[string]Position)
	seenEntrypointNames := make(map[string]Position)
	modelsByName := make(map[string]bool, len(v.prog.Models))
	for _, model := range v.prog.Models {
		modelsByName[model.Name] = true
	}
	for _, entrypoint := range v.prog.Entrypoints {
		if first, exists := seenEntrypointNames[entrypoint.Name]; exists {
			v.addDiag(DiagError, entrypoint.Pos, fmt.Sprintf("Entrypoint %q is already declared at %s", entrypoint.Name, first), nil)
		} else {
			seenEntrypointNames[entrypoint.Name] = entrypoint.Pos
		}
		for _, modelName := range stableEntrypointAliases(entrypoint.Name, entrypoint.Aliases) {
			if first, exists := seenModels[modelName]; exists {
				v.addDiag(DiagWarning, entrypoint.Pos,
					fmt.Sprintf("Entrypoint model %q is already mapped at %s", modelName, first), nil)
				continue
			}
			seenModels[modelName] = entrypoint.Pos
		}
		seenRuleNames := make(map[string]bool, len(entrypoint.Rules))
		for _, rule := range authoringEntrypointRules(entrypoint) {
			if rule == nil {
				continue
			}
			if seenRuleNames[rule.Name] {
				v.addDiag(DiagError, rule.Pos, fmt.Sprintf("Entrypoint rule %q is declared more than once", rule.Name), nil)
			}
			seenRuleNames[rule.Name] = true
			decisionNames, recipeExists := recipesByName[rule.Recipe]
			if !recipeExists {
				v.addDiag(DiagWarning, rule.Pos, fmt.Sprintf("Entrypoint rule references unknown Recipe %q", rule.Recipe), nil)
				continue
			}
			v.checkEntrypointAssignments(rule, decisionNames, modelsByName)
		}
	}
}

func (v *Validator) checkEntrypointAssignments(rule *EntrypointRuleDecl, decisionNames map[string]bool, modelsByName map[string]bool) {
	if len(rule.Assignments) != len(decisionNames) {
		v.addDiag(DiagConstraint, rule.Pos, "Entrypoint must assign every Decision in its Recipe", nil)
	}
	for decisionName, assignmentSet := range rule.Assignments {
		if !decisionNames[decisionName] {
			v.addDiag(DiagWarning, rule.Pos, fmt.Sprintf("Entrypoint assignment references unknown Decision %q", decisionName), nil)
		}
		if assignmentSet == nil {
			v.addDiag(DiagConstraint, rule.Pos, fmt.Sprintf("Entrypoint assignment for Decision %q requires at least one Model", decisionName), nil)
			continue
		}
		refs := assignmentSet.Models
		if len(refs) == 0 {
			v.addDiag(DiagConstraint, rule.Pos, fmt.Sprintf("Entrypoint assignment for Decision %q requires at least one Model", decisionName), nil)
			continue
		}
		seen := make(map[string]bool, len(refs))
		for _, ref := range refs {
			if ref == nil || ref.Model == "" {
				v.addDiag(DiagError, rule.Pos, fmt.Sprintf("Entrypoint assignment for Decision %q contains an empty Model name", decisionName), nil)
				continue
			}
			if seen[ref.Model] {
				v.addDiag(DiagConstraint, ref.Pos, fmt.Sprintf("Entrypoint assignment for Decision %q contains duplicate Model %q", decisionName, ref.Model), nil)
			}
			seen[ref.Model] = true
			_, exists := modelsByName[ref.Model]
			if !exists {
				v.addDiag(DiagWarning, ref.Pos, fmt.Sprintf("Entrypoint assignment references undefined Model %q", ref.Model), nil)
				continue
			}
			if ref.LoRAName != "" && !v.modelLoRAs[ref.Model][ref.LoRAName] {
				v.addDiag(DiagWarning, ref.Pos, fmt.Sprintf("Entrypoint assignment LoRA %q is not declared for Model %q", ref.LoRAName, ref.Model), nil)
			}
		}
	}
}

func recipeProgramWithSharedModels(parent, recipe *Program) *Program {
	if recipe == nil {
		return &Program{Models: parent.Models}
	}
	return &Program{
		Strategy:             recipe.Strategy,
		Signals:              recipe.Signals,
		ProjectionPartitions: recipe.ProjectionPartitions,
		ProjectionScores:     recipe.ProjectionScores,
		ProjectionMappings:   recipe.ProjectionMappings,
		Routes:               recipe.Routes,
		Models:               parent.Models,
		Plugins:              recipe.Plugins,
		TestBlocks:           recipe.TestBlocks,
	}
}

func (v *Validator) checkStrategy(strategy string, pos Position) {
	if err := config.RoutingStrategy(strategy).Validate(); err != nil {
		v.addDiag(DiagConstraint, pos, err.Error(), nil)
	}
}
