package handlers

import (
	"net/http"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Recipe is a Dashboard scope assertion, not an alternate routing input. The
// Router still resolves the request-facing model through its own entrypoints.
func validateTopologyScope(req TestQueryRequest, configPath string) *TestQueryResult {
	if req.Recipe == "" {
		return nil // Keep the existing text/model API usable without a scope selector.
	}
	if req.Model == "" {
		return failedTestQueryResult(req, "This recipe has no selected entrypoint. Select a recipe with a Chat entrypoint to run a preview.", http.StatusBadRequest)
	}
	cfg, err := routerconfig.Parse(configPath)
	if err != nil {
		return failedTestQueryResult(req, "Cannot verify the selected recipe. Reload the configuration before retrying.", http.StatusServiceUnavailable)
	}
	recipe, ok := cfg.RecipeForRoutingModel(req.Model)
	if !ok || string(recipe.Name) != req.Recipe {
		return failedTestQueryResult(req, "The selected entrypoint does not route to this recipe. Reload the configuration before retrying.", http.StatusConflict)
	}
	return nil
}

func verifyTopologyScope(req TestQueryRequest, response *RouterEvalResponse) *TestQueryResult {
	if req.Recipe == "" {
		return nil
	}
	if response.RequestedModel == req.Model && response.Recipe == req.Recipe {
		return nil
	}
	result := failedTestQueryResult(req, "The Router preview returned a different or missing entrypoint/recipe. Reload the configuration before retrying.", http.StatusConflict)
	result.RequestedModel = response.RequestedModel
	result.Recipe = response.Recipe
	return result
}
