package dsl

import (
	"fmt"
	"reflect"

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
	if err := c.config.LayerPlugins(); err != nil {
		c.errors = append(c.errors, err)
	}
}

// compileRecipes compiles the top-level program as the recipe named default,
// and each RECIPE program. A program of RECIPE programs whose top level
// declares no routing and that no entrypoint maps has no recipe named default,
// unless it has several and no ENTRYPOINT is the default: the empty top level
// then serves the default entrypoint.
func (c *Compiler) compileRecipes() map[config.RecipeName]struct{} {
	top := config.RoutingRecipe{
		Name: config.DefaultRecipeName,
		Profile: config.RoutingProfile{
			ModelBindings:         cloneModelBindings(c.config.ModelBindings),
			CandidateRequirements: c.config.CandidateRequirements.Clone(),
			Plugins:               c.config.Plugins,
			Signals:               c.config.Signals,
			Projections:           c.config.Projections,
			Decisions:             c.config.Decisions,
			Strategy:              c.config.Strategy,
		},
	}
	c.config.Recipes = nil
	if len(c.prog.Recipes) == 0 || !emptyProfile(top.Profile) || c.entrypointMaps(config.DefaultRecipeName) ||
		(len(c.prog.Recipes) > 1 && !c.declaresDefaultEntrypoint()) {
		c.config.Recipes = append(c.config.Recipes, top)
	}

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
				Plugins:               child.config.Plugins,
				Signals:               child.config.Signals,
				Projections:           child.config.Projections,
				Decisions:             child.config.Decisions,
				Strategy:              child.config.Strategy,
			},
		})
	}
	return recipeNames
}

func (c *Compiler) declaresDefaultEntrypoint() bool {
	for _, entrypoint := range c.prog.Entrypoints {
		if entrypoint.Default {
			return true
		}
	}
	return false
}

func (c *Compiler) entrypointMaps(recipe config.RecipeName) bool {
	for _, entrypoint := range c.prog.Entrypoints {
		if config.RecipeName(entrypoint.Recipe) == recipe {
			return true
		}
	}
	return false
}

// emptyProfile reports whether a profile declares no routing.
func emptyProfile(profile config.RoutingProfile) bool {
	if profile.Strategy != "" || len(profile.ModelBindings) > 0 || profile.CandidateRequirements != nil ||
		len(profile.Plugins) > 0 || len(profile.Decisions) > 0 {
		return false
	}
	return reflect.ValueOf(profile.Signals).IsZero() && reflect.ValueOf(profile.Projections).IsZero()
}

func (c *Compiler) compileEntrypoints(recipeNames map[config.RecipeName]struct{}) {
	seenModels := make(map[config.InferenceAPI]map[string]struct{})
	var marked *EntrypointDecl
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
		if entrypoint.Default {
			switch {
			case api != config.ChatAPI:
				c.addError(entrypoint.Pos, "only a chat ENTRYPOINT can be the default")
			case marked != nil:
				c.addError(entrypoint.Pos, "ENTRYPOINT %s is already the default; mark exactly one", marked.Pos)
			default:
				marked = entrypoint
			}
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
			Default:    entrypoint.Default,
		})
	}
	switch {
	case marked != nil:
		c.config.UseDefaultRecipe(config.RecipeName(marked.Recipe))
	case len(c.config.Recipes) == 0:
	case len(c.config.Recipes) == 1:
		c.config.UseDefaultRecipe(c.config.Recipes[0].Name)
	case c.config.Recipes[0].Name == config.DefaultRecipeName:
		c.markTopLevelDefault()
	default:
		c.errors = append(c.errors, fmt.Errorf("%d recipes need exactly one ENTRYPOINT with default: true", len(c.config.Recipes)))
	}
}

// markTopLevelDefault makes the top-level program the default recipe beside
// RECIPE programs, as it was before ENTRYPOINT took default: its first chat
// ENTRYPOINT is the default one, else one serving vllm-sr/auto leads the table.
func (c *Compiler) markTopLevelDefault() {
	for i := range c.config.Entrypoints {
		entrypoint := &c.config.Entrypoints[i]
		if entrypoint.Recipe == config.DefaultRecipeName && (entrypoint.API == "" || entrypoint.API == config.ChatAPI) {
			entrypoint.Default = true
			return
		}
	}
	c.config.Entrypoints = append([]config.EntrypointMapping{{
		ModelNames: []string{config.DefaultEntrypointModel}, Recipe: config.DefaultRecipeName, Default: true,
	}}, c.config.Entrypoints...)
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
	c.config.Plugins = append([]config.DecisionPlugin(nil), prog.RecipePlugins...)
	return c
}
