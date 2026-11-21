package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

var modelHealthOperation = apicontract.Operation{
	ID:      "getModelHealth",
	Summary: "Health of the configured backend models",
	Description: "Checks each backend with a cheap read, never inference: its reliability.health_check_path when set, " +
		"else the provider's model list with the backend's credentials, else its base URL. Repeat ?model= to check " +
		"only those models. Checks younger than 20 s answer every viewer; ?refresh=1 checks again unless the last " +
		"check of a model is under 5 s old. Backends are named as the configuration names them, without URLs or credentials.",
	Responses: map[int]apicontract.Response{
		http.StatusOK:                 apicontract.JSONResponse[handlers.ModelHealthResponse]("The latest health of each requested model"),
		http.StatusBadRequest:         apicontract.JSONResponse[handlers.ModelHealthErrorResponse]("error invalid_request for an empty or oversized model name, or more than 64 models"),
		http.StatusNotFound:           apicontract.JSONResponse[handlers.ModelHealthErrorResponse]("error model_not_found for a model the active configuration does not declare"),
		http.StatusServiceUnavailable: apicontract.JSONResponse[handlers.ModelHealthErrorResponse]("error config_unavailable when the active configuration cannot be read"),
	},
}
