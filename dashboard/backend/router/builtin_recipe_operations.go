package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

var listBuiltinRecipesOperation = apicontract.Operation{
	ID:      "listBuiltinRecipes",
	Summary: "List the Recipes installed with vllm-sr",
	Description: "Reads the bundles packaged with the Dashboard. Each decision states the distinct models it needs; " +
		"a decision that responds without calling a backend has a null algorithm and needs none.",
	Responses: map[int]apicontract.Response{
		http.StatusOK:                 apicontract.JSONResponse[handlers.BuiltinRecipeCatalog]("Installed bundles, their recipes, and each decision's candidate pool"),
		http.StatusBadGateway:         apicontract.JSONResponse[handlers.BuiltinRecipeErrorResponse]("The installed vllm-sr package returned an invalid listing"),
		http.StatusServiceUnavailable: apicontract.JSONResponse[handlers.BuiltinRecipeErrorResponse]("The installed vllm-sr package could not be run"),
	},
}

var composeBuiltinRecipeOperation = apicontract.Operation{
	ID:      "composeBuiltinRecipe",
	Summary: "Compose a built-in Recipe into the current config",
	Description: "Returns the current config with the recipe added, each decision that calls a backend bound to its models, " +
		"and an entrypoint for model_names. Nothing is written: save the result with POST /api/router/config/update. " +
		"The installed vllm-sr CLI applies the binding rules and validates the result.",
	Request: apicontract.JSONRequest[handlers.BuiltinRecipeComposeRequest]("Installed recipe, model refs for each decision that calls a backend, and public model names"),
	Responses: map[int]apicontract.Response{
		http.StatusOK: apicontract.JSONResponse[map[string]any]("Composed canonical config"),
		http.StatusBadRequest: apicontract.JSONResponse[handlers.BuiltinRecipeErrorResponse](
			"error invalid_request for a malformed body, or recipe_rejected with the CLI's reason, such as an unknown model, " +
				"too few candidates, a recipe the config already contains, or a public name in use",
		),
		http.StatusForbidden: apicontract.EitherResponse(
			"JSON when the Dashboard is read-only (error readonly_mode); text when the caller lacks the permission or the origin or CSRF check failed",
			apicontract.JSONResponse[handlers.BuiltinRecipeErrorResponse](""),
			apicontract.TextResponse(""),
		),
		http.StatusRequestEntityTooLarge: apicontract.TextResponse("The request exceeds 64 KiB"),
		http.StatusServiceUnavailable:    apicontract.JSONResponse[handlers.BuiltinRecipeErrorResponse]("The installed vllm-sr package could not run the composition"),
	},
}
