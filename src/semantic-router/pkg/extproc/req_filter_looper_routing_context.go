package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// hydrateLooperRoutingContext restores the parent request's recipe boundary on
// router-generated looper calls. A routing decision made earlier in the same
// request lifecycle remains authoritative.
func (r *OpenAIRouter) hydrateLooperRoutingContext(ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil || !ctx.LooperRequest || ctx.Routing.IsResolved() {
		return
	}

	recipeName := config.RecipeName(strings.TrimSpace(headerValueCI(ctx, headers.VSRSelectedRecipe)))
	if recipeName == "" {
		return
	}
	recipe, ok := r.Config.RecipeByRuntimeScope(recipeName)
	if !ok {
		return
	}
	ctx.Routing.SelectRecipe(recipe)
}

func (r *OpenAIRouter) looperDecisionForRoutingContext(
	ctx *RequestContext,
	decisionName string,
) *config.Decision {
	if ctx == nil || ctx.Routing.SelectedRecipe() == nil {
		return nil
	}
	return routingRecipeDecisionByName(ctx.Routing.SelectedRecipe(), decisionName)
}

func routingRecipeDecisionByName(recipe *config.RoutingRecipe, decisionName string) *config.Decision {
	if recipe == nil {
		return nil
	}
	for index := range recipe.Profile.Decisions {
		if recipe.Profile.Decisions[index].Name == decisionName {
			return &recipe.Profile.Decisions[index]
		}
	}
	return nil
}
