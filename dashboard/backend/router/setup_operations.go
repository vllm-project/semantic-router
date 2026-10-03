package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

// OpenAPI operations for the first-run surface: health, settings, and setup.
// Each schema comes from the type the handler encodes or decodes.

var healthzOperation = apicontract.Operation{
	ID:      "getHealthz",
	Summary: "Dashboard liveness",
	Responses: map[int]apicontract.Response{
		http.StatusOK: apicontract.JSONResponse[handlers.HealthResponse]("The Dashboard process is serving requests"),
	},
}

var settingsOperation = apicontract.Operation{
	ID:          "getSettings",
	Summary:     "Dashboard settings",
	Description: "readonlyMode also reflects whether the caller lacks config.write.",
	Responses: map[int]apicontract.Response{
		http.StatusOK: apicontract.JSONResponse[handlers.SettingsResponse]("Settings for the current caller"),
	},
}

var setupStateOperation = apicontract.Operation{
	ID:          "getSetupState",
	Summary:     "Setup mode state",
	Description: "Public so the first-run screen can render before an account exists. reason is set when the config could not be read.",
	Responses: map[int]apicontract.Response{
		http.StatusOK: apicontract.JSONResponse[handlers.SetupStateResponse]("Resolved setup state and config summary"),
	},
}

var setupImportRemoteOperation = apicontract.Operation{
	ID:      "importRemoteSetupConfig",
	Summary: "Import a setup config from a remote URL",
	Request: apicontract.JSONRequest[handlers.SetupImportRemoteRequest]("Remote config location"),
	Responses: map[int]apicontract.Response{
		http.StatusOK:                    apicontract.JSONResponse[handlers.SetupImportRemoteResponse]("Fetched and summarized config"),
		http.StatusBadRequest:            apicontract.TextResponse("The setup config is unavailable, the body is invalid, or the destination is not permitted"),
		http.StatusRequestEntityTooLarge: apicontract.TextResponse("The remote config exceeds the size limit"),
		http.StatusBadGateway:            apicontract.TextResponse("The remote config could not be fetched"),
	},
}

var setupValidateOperation = apicontract.Operation{
	ID:      "validateSetupConfig",
	Summary: "Validate a setup config candidate",
	Request: apicontract.JSONRequest[handlers.SetupConfigRequest]("Candidate canonical config"),
	Responses: map[int]apicontract.Response{
		http.StatusOK:                  apicontract.JSONResponse[handlers.SetupValidateResponse]("The candidate is valid"),
		http.StatusBadRequest:          apicontract.TextResponse("The setup config is unavailable or the candidate is invalid"),
		http.StatusInternalServerError: apicontract.TextResponse("The validated config could not be encoded"),
	},
}

// setupReadonlyError mirrors the map SetupActivateHandler encodes when the
// Dashboard is read-only. The handler is the source of truth;
// TestSetupActivateReadonlyBodyMatchesOpenAPI catches drift.
type setupReadonlyError struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// setupActivationFailure mirrors the map failSetupActivation encodes once the
// candidate config was written but the runtime could not start.
type setupActivationFailure struct {
	Error                 string `json:"error"`
	Stage                 string `json:"stage"`
	ConfigurationRestored bool   `json:"configuration_restored"`
	SetupMode             bool   `json:"setupMode"`
	Message               string `json:"message"`
}

var setupActivateOperation = apicontract.Operation{
	ID:      "activateSetupConfig",
	Summary: "Activate the setup config and leave setup mode",
	Description: "Writes the candidate config and starts the Router and Envoy. Error bodies are plain text except two JSON cases: " +
		"a read-only Dashboard (403, error readonly_mode) and a runtime start failure after the write (500, error setup_activation_failed). " +
		"Choose the parser by Content-Type.",
	Request: apicontract.JSONRequest[handlers.SetupConfigRequest]("Candidate canonical config"),
	Responses: map[int]apicontract.Response{
		http.StatusOK:         apicontract.JSONResponse[handlers.SetupActivateResponse]("The config was activated and services are starting"),
		http.StatusAccepted:   apicontract.JSONResponse[handlers.SetupActivateResponse]("The config was saved to the Kubernetes ConfigMap and awaits a rollout"),
		http.StatusBadRequest: apicontract.TextResponse("The setup config is unavailable or the candidate is invalid"),
		http.StatusForbidden: apicontract.EitherResponse(
			"JSON when the Dashboard is read-only; text when the caller lacks the permission, the origin or CSRF check failed, or the ConfigMap is controller-owned",
			apicontract.JSONResponse[setupReadonlyError](""),
			apicontract.TextResponse(""),
		),
		http.StatusConflict: apicontract.TextResponse("Setup is no longer active, or the config changed during the request"),
		http.StatusInternalServerError: apicontract.EitherResponse(
			"JSON when the runtime failed to start after the write: stage is config_validation, runtime_config_sync, or runtime_start, "+
				"and configuration_restored says whether a retry is safe. Text when the config could not be read, converted, backed up, or persisted",
			apicontract.JSONResponse[setupActivationFailure](""),
			apicontract.TextResponse(""),
		),
	},
}

var setupPresetsOperation = apicontract.Operation{
	ID:      "listSetupPresets",
	Summary: "List built-in setup presets",
	Responses: map[int]apicontract.Response{
		http.StatusOK: apicontract.JSONResponse[[]handlers.PresetInfo]("Built-in presets"),
	},
}

var setupPresetDeltaOperation = apicontract.Operation{
	ID:      "computeSetupPresetDelta",
	Summary: "Compare configured models with a preset",
	Request: apicontract.JSONRequest[handlers.PresetDeltaRequest]("Preset and configured model names"),
	Responses: map[int]apicontract.Response{
		http.StatusOK:         apicontract.JSONResponse[handlers.PresetDeltaResponse]("Configured and missing preset models"),
		http.StatusBadRequest: apicontract.TextResponse("The body is invalid"),
		http.StatusNotFound:   apicontract.TextResponse("The preset id is unknown"),
	},
}
