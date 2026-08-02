package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// compileScopes lowers recipe-local programs after the shared model catalog
// and default routing profile have been compiled. Every recipe receives a new
// Compiler instance, which makes accidental cross-recipe symbol reuse
// structurally impossible.
func (c *Compiler) compileScopes() {
	if err := c.config.Strategy.Validate(); err != nil {
		c.errors = append(c.errors, err)
	}
	recipeNames := c.compileRecipes()
	c.compileEntrypoints(recipeNames)
}

func (c *Compiler) compileRecipes() map[config.RecipeName]struct{} {
	c.config.Recipes = []config.RoutingRecipe{{
		Name: config.DefaultRecipeName,
		Profile: config.RoutingProfile{
			Signals:     c.config.Signals,
			Projections: c.config.Projections,
			Decisions:   c.config.Decisions,
			Strategy:    c.config.Strategy,
		},
	}}

	recipeNames := map[config.RecipeName]struct{}{config.DefaultRecipeName: {}}
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

		child := newScopedCompiler(recipe.Program)
		child.compile()
		if err := child.config.Strategy.Validate(); err != nil {
			child.errors = append(child.errors, err)
		}
		for _, err := range child.errors {
			c.errors = append(c.errors, fmt.Errorf("RECIPE %q: %w", name, err))
		}
		c.config.Recipes = append(c.config.Recipes, config.RoutingRecipe{
			Name:        name,
			Description: recipe.Description,
			Profile: config.RoutingProfile{
				Signals:     child.config.Signals,
				Projections: child.config.Projections,
				Decisions:   child.config.Decisions,
				Strategy:    child.config.Strategy,
			},
		})
	}
	return recipeNames
}

func (c *Compiler) compileEntrypoints(recipeNames map[config.RecipeName]struct{}) {
	seenModels := make(map[string]struct{})
	for _, entrypoint := range c.prog.Entrypoints {
		recipeName := config.RecipeName(entrypoint.Recipe)
		if _, exists := recipeNames[recipeName]; !exists {
			c.addError(entrypoint.Pos, "ENTRYPOINT references unknown recipe %q", entrypoint.Recipe)
			continue
		}
		for _, modelName := range entrypoint.ModelNames {
			if modelName == "" {
				c.addError(entrypoint.Pos, "ENTRYPOINT model_names cannot contain an empty value")
				continue
			}
			if _, exists := seenModels[modelName]; exists {
				c.addError(entrypoint.Pos, "entrypoint model %q is mapped more than once", modelName)
				continue
			}
			seenModels[modelName] = struct{}{}
		}
		c.config.Entrypoints = append(c.config.Entrypoints, config.EntrypointMapping{
			ModelNames: append([]string(nil), entrypoint.ModelNames...),
			Recipe:     recipeName,
		})
	}
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
