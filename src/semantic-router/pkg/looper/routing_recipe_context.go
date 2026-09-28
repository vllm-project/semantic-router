package looper

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type routingRecipeContextKey struct{}

func contextWithRoutingRecipe(ctx context.Context, recipe config.RecipeName) context.Context {
	normalized := config.RecipeName(strings.TrimSpace(string(recipe)))
	if normalized == "" {
		return ctx
	}
	return context.WithValue(ctx, routingRecipeContextKey{}, normalized)
}

func routingRecipeFromContext(ctx context.Context) config.RecipeName {
	if ctx == nil {
		return ""
	}
	recipe, _ := ctx.Value(routingRecipeContextKey{}).(config.RecipeName)
	return recipe
}
