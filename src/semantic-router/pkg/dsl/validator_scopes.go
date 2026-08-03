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

	recipeNames := map[string]Position{"default": {}}
	for _, recipe := range v.prog.Recipes {
		if recipe.Name == "" {
			v.addDiag(DiagError, recipe.Pos, "RECIPE name cannot be empty", nil)
			continue
		}
		if first, exists := recipeNames[recipe.Name]; exists {
			v.addDiag(DiagError, recipe.Pos,
				fmt.Sprintf("Recipe %q is declared more than once (first declaration at %s)", recipe.Name, first), nil)
			continue
		}
		recipeNames[recipe.Name] = recipe.Pos
		v.checkRecipeScope(recipe)
	}
	v.checkEntrypointScopes(recipeNames)
}

func (v *Validator) checkRecipeScope(recipe *RecipeDecl) {
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

func (v *Validator) checkEntrypointScopes(recipeNames map[string]Position) {
	seenModels := make(map[string]Position)
	for _, entrypoint := range v.prog.Entrypoints {
		if _, exists := recipeNames[entrypoint.Recipe]; !exists {
			v.addDiag(DiagWarning, entrypoint.Pos,
				fmt.Sprintf("Entrypoint references unknown recipe %q", entrypoint.Recipe), nil)
		}
		for _, modelName := range entrypoint.ModelNames {
			if first, exists := seenModels[modelName]; exists {
				v.addDiag(DiagWarning, entrypoint.Pos,
					fmt.Sprintf("Entrypoint model %q is already mapped at %s", modelName, first), nil)
				continue
			}
			seenModels[modelName] = entrypoint.Pos
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
