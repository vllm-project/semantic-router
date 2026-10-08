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

// setupErrorBody mirrors the map SetupActivateHandler encodes when the
// Dashboard is read-only, and writeRuntimeConfigMutationError's body.
// TestSetupActivateReadonlyBodyMatchesOpenAPI catches drift.
type setupErrorBody struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// setupActivationFailure mirrors the map failSetupActivation encodes after
// the candidate config was written.
type setupActivationFailure struct {
	Error                 string `json:"error"`
	Stage                 string `json:"stage"`
	ConfigurationRestored bool   `json:"configuration_restored"`
	SetupMode             bool   `json:"setupMode"`
	Message               string `json:"message"`
}

// setupActivationFailureStages are the stage values failSetupActivation reports.
var setupActivationFailureStages = []string{"config_validation", "runtime_config_sync", "activation_record"}

func setupActivationFailureSchema() apicontract.Schema {
	schema := apicontract.SchemaFor[setupActivationFailure]()
	stage := schema.Properties["stage"]
	stage.Enum = setupActivationFailureStages
	schema.Properties["stage"] = stage
	return schema
}

var setupActivateOperation = apicontract.Operation{
	ID:      "activateSetupConfig",
	Summary: "Activate the setup config and leave setup mode",
	Description: "Writes the candidate config and records a pending activation for the `vllm-sr serve` that owns the stack; " +
		"the Dashboard does not start containers. Error bodies are JSON or plain text; choose the parser by Content-Type.",
	Request: apicontract.JSONRequest[handlers.SetupConfigRequest]("Candidate canonical config"),
	Responses: map[int]apicontract.Response{
		http.StatusOK: apicontract.JSONResponse[handlers.SetupActivateResponse](
			"Config saved and activation recorded. message says whether an attached `vllm-sr serve` is starting the stack " +
				"or the user must run `vllm-sr serve`; Envoy starts only in stacks that run it",
		),
		http.StatusAccepted:   apicontract.JSONResponse[handlers.SetupActivateResponse]("Config saved to the Kubernetes ConfigMap; a rollout applies it"),
		http.StatusBadRequest: apicontract.TextResponse("The setup config is unavailable or the candidate is invalid"),
		http.StatusForbidden: apicontract.EitherResponse(
			"JSON when the Dashboard is read-only (error readonly_mode); text when the caller lacks the permission, "+
				"the origin or CSRF check failed, the session was revoked, or the ConfigMap is controller-owned",
			apicontract.JSONResponse[setupErrorBody](""),
			apicontract.TextResponse(""),
		),
		http.StatusConflict: apicontract.EitherResponse(
			"JSON when another config change holds the lock (error deploy_in_progress or managed_recipe_active); "+
				"text when setup is no longer active or the ConfigMap changed or awaits a rollout",
			apicontract.JSONResponse[setupErrorBody](""),
			apicontract.TextResponse(""),
		),
		http.StatusInternalServerError: apicontract.EitherResponse(
			"JSON error setup_activation_failed after the config was written: stage names the failed step and "+
				"configuration_restored says whether a retry is safe. JSON error config_coordination_failed when the config lock "+
				"is unavailable. Text when the config could not be read, converted, backed up, or persisted",
			apicontract.JSONOneOfResponse("", setupActivationFailureSchema(), apicontract.SchemaFor[setupErrorBody]()),
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
