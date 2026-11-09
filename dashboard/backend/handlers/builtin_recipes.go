package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

const (
	builtinRecipesPath       = "/api/recipes/builtin"
	builtinRecipeComposePath = "/api/recipes/builtin/compose"
	builtinRecipeModule      = "cli.builtin_recipes_json"
	builtinRecipeTimeout     = 60 * time.Second
	// builtinRecipeRejectedExitCode is how the module reports a request it
	// refused; its standard output is then the reason.
	builtinRecipeRejectedExitCode = 65
)

var errInvalidBuiltinRecipeCatalog = errors.New("invalid built-in recipe catalog")

// BuiltinRecipeRunner runs the built-in Recipe operations of the packaged
// vllm-sr CLI. Keeping this seam injectable lets the Dashboard apply the CLI's
// binding rules without copying them into Go.
type BuiltinRecipeRunner interface {
	List(context.Context) ([]byte, error)
	// Compose returns the config at configPath with the request applied, or a
	// *BuiltinRecipeRejection when the CLI refuses the request.
	Compose(ctx context.Context, configPath string, request BuiltinRecipeComposeRequest) ([]byte, error)
}

// BuiltinRecipeRejection carries the CLI's reason for refusing a composition.
type BuiltinRecipeRejection struct {
	Message string
}

func (rejection *BuiltinRecipeRejection) Error() string { return rejection.Message }

// BuiltinRecipeCatalog lists the Recipes in the bundles installed with vllm-sr.
type BuiltinRecipeCatalog struct {
	CatalogVersion string                `json:"catalog_version"`
	Bundles        []BuiltinRecipeBundle `json:"bundles"`
}

type BuiltinRecipeBundle struct {
	Name    string          `json:"name"`
	SHA256  string          `json:"sha256"`
	Recipes []BuiltinRecipe `json:"recipes"`
}

type BuiltinRecipe struct {
	Name        string                  `json:"name"`
	Description string                  `json:"description"`
	Decisions   []BuiltinRecipeDecision `json:"decisions"`
}

// BuiltinRecipeDecision is the candidate pool one decision needs. A decision
// that responds without calling a backend has no algorithm and a minimum of 0.
type BuiltinRecipeDecision struct {
	Name              string  `json:"name"`
	Description       string  `json:"description"`
	Algorithm         *string `json:"algorithm"`
	MinimumCandidates int     `json:"minimum_candidates"`
}

// BuiltinRecipeComposeRequest binds the decisions of one installed Recipe to
// configured models and names the entrypoint; the first model name is primary.
type BuiltinRecipeComposeRequest struct {
	Bundle     string                      `json:"bundle"`
	Recipe     string                      `json:"recipe"`
	Bindings   map[string][]map[string]any `json:"bindings"`
	ModelNames []string                    `json:"model_names"`
}

// BuiltinRecipeErrorResponse is the body of every built-in Recipe API error.
type BuiltinRecipeErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

type packagedBuiltinRecipeRunner struct {
	cli packagedCLI
}

// NewPackagedBuiltinRecipeRunner runs the CLI packaged in the same runtime
// image as the Dashboard.
func NewPackagedBuiltinRecipeRunner(pythonPath string) BuiltinRecipeRunner {
	return &packagedBuiltinRecipeRunner{cli: newPackagedCLI(pythonPath)}
}

func (runner *packagedBuiltinRecipeRunner) List(ctx context.Context) ([]byte, error) {
	return runner.cli.run(ctx, builtinRecipeTimeout, nil, builtinRecipeModule, "list")
}

func (runner *packagedBuiltinRecipeRunner) Compose(ctx context.Context, configPath string, request BuiltinRecipeComposeRequest) ([]byte, error) {
	input, err := json.Marshal(struct {
		ConfigPath string `json:"config_path"`
		BuiltinRecipeComposeRequest
	}{ConfigPath: configPath, BuiltinRecipeComposeRequest: request})
	if err != nil {
		return nil, err
	}
	output, err := runner.cli.run(ctx, builtinRecipeTimeout, input, builtinRecipeModule, "compose")
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) && exit.ExitCode() == builtinRecipeRejectedExitCode {
			return nil, &BuiltinRecipeRejection{Message: strings.TrimSpace(string(output))}
		}
		return nil, err
	}
	return output, nil
}

func normalizeBuiltinRecipeCatalog(raw []byte) ([]byte, error) {
	var catalog BuiltinRecipeCatalog
	if err := json.Unmarshal(raw, &catalog); err != nil || !validBuiltinRecipeCatalog(catalog) {
		return nil, errInvalidBuiltinRecipeCatalog
	}
	return json.Marshal(catalog)
}

func validBuiltinRecipeCatalog(catalog BuiltinRecipeCatalog) bool {
	if catalog.CatalogVersion == "" || catalog.Bundles == nil {
		return false
	}
	for _, bundle := range catalog.Bundles {
		if bundle.Name == "" || bundle.Recipes == nil {
			return false
		}
		for _, recipe := range bundle.Recipes {
			if recipe.Name == "" || recipe.Decisions == nil {
				return false
			}
			for _, decision := range recipe.Decisions {
				if decision.Name == "" || decision.MinimumCandidates < 0 {
					return false
				}
			}
		}
	}
	return true
}

// BuiltinRecipesHandler lists the Recipes installed with vllm-sr and the
// models each decision needs.
func BuiltinRecipesHandler(runner BuiltinRecipeRunner) http.HandlerFunc {
	catalog := &packagedDocument{load: func(ctx context.Context) ([]byte, error) {
		raw, err := runner.List(ctx)
		if err != nil {
			return nil, err
		}
		return normalizeBuiltinRecipeCatalog(raw)
	}}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != builtinRecipesPath {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			writeBuiltinRecipeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET and HEAD are supported.")
			return
		}

		payload, err := catalog.get(r.Context())
		if errors.Is(err, errInvalidBuiltinRecipeCatalog) {
			writeBuiltinRecipeError(w, http.StatusBadGateway, "builtin_recipes_contract_invalid", "The installed vllm-sr package returned an invalid built-in recipe listing.")
			return
		}
		if err != nil {
			writeBuiltinRecipeError(w, http.StatusServiceUnavailable, "builtin_recipes_unavailable", "Built-in recipes could not be loaded from the installed vllm-sr package.")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "private, max-age=60")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}
}

// BuiltinRecipeComposeHandler returns the config at configPath with one
// installed Recipe bound and published, without writing it. The config update
// endpoint validates, versions, and applies a composed config.
func BuiltinRecipeComposeHandler(runner BuiltinRecipeRunner, configPath string, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != builtinRecipeComposePath {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeBuiltinRecipeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is supported.")
			return
		}
		if readonlyMode {
			writeBuiltinRecipeError(w, http.StatusForbidden, "readonly_mode", "Dashboard is in read-only mode. Configuration editing is disabled.")
			return
		}

		var request BuiltinRecipeComposeRequest
		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil || request.Bindings == nil || request.ModelNames == nil {
			writeBuiltinRecipeError(w, http.StatusBadRequest, "invalid_request", "Send one JSON object with bundle, recipe, bindings, and model_names.")
			return
		}

		config, err := runner.Compose(r.Context(), configPath, request)
		var rejection *BuiltinRecipeRejection
		if errors.As(err, &rejection) {
			writeBuiltinRecipeError(w, http.StatusBadRequest, "recipe_rejected", rejection.Message)
			return
		}
		if err != nil {
			writeBuiltinRecipeError(w, http.StatusServiceUnavailable, "builtin_recipes_unavailable", "The built-in recipe could not be composed by the installed vllm-sr package.")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(config)
	}
}

func writeBuiltinRecipeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(BuiltinRecipeErrorResponse{Error: code, Message: message})
}
