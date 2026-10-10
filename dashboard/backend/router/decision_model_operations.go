package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

var decisionRoutesOperation = apicontract.Operation{
	ID:      "listDecisionRoutes",
	Summary: "List active System One routes",
	Description: "Reads the configured frontend without probing models. available describes routing availability, not backend health. " +
		"Each route includes model, recipe, algorithms, question_types, and execution_timeout_ms. " +
		"The timeout is the largest algorithm execution deadline in the recipe; signals have separate timeouts.",
	Responses: decisionRouteResponses("Frontend route inventory with available and routes"),
}

var decisionRouteRunOperation = apicontract.Operation{
	ID:      "runDecisionRoute",
	Summary: "Run an active System One route",
	Description: "Runs an operator diagnostic using configured management credentials. Browser credentials are not forwarded; " +
		"no public listener API key is needed. The frontend validates the native request and owns route selection. " +
		"The selected algorithm controls its execution deadline and physical call budget; signals use their own timeouts.",
	Request:   decisionRouteRequestBody(),
	Responses: decisionRouteResponses("Selected native response with routing outcome: recipe, decision, algorithm, stage, selected_model, quality, and model_calls"),
}

func decisionRouteRequestBody() *apicontract.RequestBody {
	body := apicontract.JSONRequest[handlers.DecisionModelRouteRequest]("An active route model alias and the original native request")
	schema := body.Content["application/json"].Schema
	schema.AdditionalProperties = false
	// The Dashboard checks only this envelope. Keep the native object open so
	// it does not invent a second copy of the frontend's versioned protocol.
	schema.Properties["request"] = apicontract.Schema{
		Type:                 "object",
		AdditionalProperties: true,
		Description:          "System One request containing state and named questions; validated by the configured frontend",
	}
	return body
}

func decisionRouteResponses(success string) map[int]apicontract.Response {
	responses := map[int]apicontract.Response{
		http.StatusOK:                    apicontract.JSONResponse[map[string]any](success),
		http.StatusBadRequest:            apicontract.JSONResponse[map[string]any]("Invalid request wrapper or native request"),
		http.StatusNotFound:              apicontract.JSONResponse[map[string]any]("The frontend endpoint or selected route is not available"),
		http.StatusRequestEntityTooLarge: apicontract.JSONResponse[map[string]any]("The request exceeds 2 MiB"),
		http.StatusBadGateway:            apicontract.JSONResponse[map[string]any]("The frontend is unreachable or returned an invalid response"),
		http.StatusServiceUnavailable:    apicontract.JSONResponse[map[string]any]("The route is unavailable, the request was canceled, or the cascade could not resolve the request"),
		http.StatusGatewayTimeout:        apicontract.JSONResponse[map[string]any]("The frontend or selected algorithm deadline elapsed"),
	}
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		responses[status] = apicontract.EitherResponse(
			"Text for Dashboard authentication or permission failures; JSON for a frontend management authorization failure",
			apicontract.TextResponse(""), apicontract.JSONResponse[map[string]any](""),
		)
	}
	return responses
}
