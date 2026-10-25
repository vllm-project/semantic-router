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
	if err := c.config.CandidateRequirements.Validate(); err != nil {
		c.errors = append(c.errors, err)
	}
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
			ModelBindings:         cloneModelBindings(c.config.ModelBindings),
			CandidateRequirements: c.config.CandidateRequirements.Clone(),
			Signals:               c.config.Signals,
			Projections:           c.config.Projections,
			Decisions:             c.config.Decisions,
			Strategy:              c.config.Strategy,
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
		if err := child.config.CandidateRequirements.Validate(); err != nil {
			child.errors = append(child.errors, err)
		}
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
				ModelBindings:         cloneModelBindings(child.config.ModelBindings),
				CandidateRequirements: child.config.CandidateRequirements.Clone(),
				Signals:               child.config.Signals,
				Projections:           child.config.Projections,
				Decisions:             child.config.Decisions,
				Strategy:              child.config.Strategy,
			},
		})
	}
	return recipeNames
}

func (c *Compiler) compileEntrypoints(recipeNames map[config.RecipeName]struct{}) {
	seenModels := make(map[config.InferenceAPI]map[string]struct{})
	for _, entrypoint := range c.prog.Entrypoints {
		recipeName := config.RecipeName(entrypoint.Recipe)
		if _, exists := recipeNames[recipeName]; !exists {
			c.addError(entrypoint.Pos, "ENTRYPOINT references unknown recipe %q", entrypoint.Recipe)
			continue
		}
		api := entrypointAPI(entrypoint)
		if api != config.ChatAPI && api != config.SystemOneAPI {
			c.addError(entrypoint.Pos, "ENTRYPOINT api must be chat or systemone, got %q", entrypoint.API)
			continue
		}
		if seenModels[api] == nil {
			seenModels[api] = make(map[string]struct{})
		}
		for _, modelName := range entrypoint.ModelNames {
			if modelName == "" {
				c.addError(entrypoint.Pos, "ENTRYPOINT model_names cannot contain an empty value")
				continue
			}
			if _, exists := seenModels[api][modelName]; exists {
				c.addError(entrypoint.Pos, "entrypoint model %q is mapped more than once", modelName)
				continue
			}
			seenModels[api][modelName] = struct{}{}
		}
		c.config.Entrypoints = append(c.config.Entrypoints, config.EntrypointMapping{
			API:        config.InferenceAPI(entrypoint.API),
			ModelNames: append([]string(nil), entrypoint.ModelNames...),
			Recipe:     recipeName,
		})
	}
}

// An omitted api is Chat; Chat and System One keep separate model-name namespaces.
func entrypointAPI(entrypoint *EntrypointDecl) config.InferenceAPI {
	if entrypoint.API == "" {
		return config.ChatAPI
	}
	return config.InferenceAPI(entrypoint.API)
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
	c.config.ModelBindings = cloneModelBindings(prog.ModelBindings)
	c.config.CandidateRequirements = prog.CandidateRequirements.Clone()
	return c
}
