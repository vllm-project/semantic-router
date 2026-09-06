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
	recipe, ok := r.looperRecipeByName(recipeName)
	if !ok {
		return
	}
	ctx.Routing.SelectRecipe(recipe)
}

func (r *OpenAIRouter) looperRecipeByName(name config.RecipeName) (*config.RoutingRecipe, bool) {
	if r == nil || r.Config == nil {
		return nil, false
	}
	if name == config.DefaultRecipeName {
		recipe := r.Config.DefaultRecipe()
		return recipe, recipe != nil
	}
	return r.Config.RecipeByName(name)
}

func (r *OpenAIRouter) looperDecisionForRoutingContext(
	ctx *RequestContext,
	decisionName string,
) *config.Decision {
	if ctx == nil {
		return nil
	}
	if recipe := ctx.Routing.SelectedRecipe(); recipe != nil {
		return routingRecipeDecisionByName(recipe, decisionName)
	}
	if ctx.Routing.IsResolved() {
		return nil
	}

	// A present but unknown recipe header must not fall through to another
	// recipe's same-named decision.
	if strings.TrimSpace(headerValueCI(ctx, headers.VSRSelectedRecipe)) != "" {
		return nil
	}

	// Internal calls from older clients did not carry recipe scope. Preserve
	// their default-profile behavior while establishing an explicit routing
	// context for replay recorder selection.
	if r != nil && r.Config != nil {
		if recipe := r.Config.DefaultRecipe(); recipe != nil {
			if decision := routingRecipeDecisionByName(recipe, decisionName); decision != nil {
				ctx.Routing.SelectRecipe(recipe)
				return decision
			}
		}
	}

	// Retain support for focused programmatic callers that supplied an already
	// scoped decision object before RequestRoutingContext existed.
	if ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.Name == decisionName {
		return ctx.VSRSelectedDecision
	}
	return nil
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
