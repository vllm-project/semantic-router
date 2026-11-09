package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

const builtinRecipeListing = `{
  "bundles": [{
    "files": ["README.md", "config.yaml", "metadata.yaml", "probes.yaml", "recipe.dsl"],
    "name": "mom-v1",
    "recipes": [{
      "decisions": [
        {"algorithm": null, "description": "Contain prompt attacks.", "minimum_candidates": 0, "name": "guard"},
        {"algorithm": "multi_factor", "description": "Serve private requests.", "minimum_candidates": 1, "name": "private"}
      ],
      "description": "Keep private traffic inside the deployment boundary.",
      "name": "vault"
    }],
    "sha256": "sha256:aaaa"
  }],
  "catalog_version": "latest"
}`

type fakeBuiltinRecipeRunner struct {
	listing    []byte
	listErr    error
	listCalls  int
	composed   []byte
	composeErr error
	configPath string
	request    *BuiltinRecipeComposeRequest
}

func (runner *fakeBuiltinRecipeRunner) List(context.Context) ([]byte, error) {
	runner.listCalls++
	return runner.listing, runner.listErr
}

func (runner *fakeBuiltinRecipeRunner) Compose(_ context.Context, configPath string, request BuiltinRecipeComposeRequest) ([]byte, error) {
	runner.configPath, runner.request = configPath, &request
	return runner.composed, runner.composeErr
}

func TestBuiltinRecipesHandlerReturnsTheListingAndCachesSuccess(t *testing.T) {
	t.Parallel()

	runner := &fakeBuiltinRecipeRunner{listErr: errors.New("python unavailable")}
	handler := BuiltinRecipesHandler(runner)
	get := func() *httptest.ResponseRecorder {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/recipes/builtin", nil))
		return response
	}
	if response := get(); response.Code != http.StatusServiceUnavailable {
		t.Fatalf("failed load status=%d body=%s", response.Code, response.Body.String())
	}

	runner.listing, runner.listErr = []byte(builtinRecipeListing), nil
	for attempt := 0; attempt < 2; attempt++ {
		response := get()
		if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "application/json" {
			t.Fatalf("attempt %d status=%d body=%s", attempt, response.Code, response.Body.String())
		}
		var catalog BuiltinRecipeCatalog
		if err := json.Unmarshal(response.Body.Bytes(), &catalog); err != nil {
			t.Fatal(err)
		}
		guard := catalog.Bundles[0].Recipes[0].Decisions[0]
		if catalog.CatalogVersion != "latest" || catalog.Bundles[0].Name != "mom-v1" ||
			guard.Name != "guard" || guard.Algorithm != nil || guard.MinimumCandidates != 0 || guard.Description != "Contain prompt attacks." {
			t.Fatalf("listing = %+v", catalog)
		}
		if strings.Contains(response.Body.String(), "files") {
			t.Fatalf("listing exposed bundle files: %s", response.Body.String())
		}
	}
	if runner.listCalls != 2 {
		t.Fatalf("list calls=%d want=2 (one failure, then one cached success)", runner.listCalls)
	}
}

func TestBuiltinRecipesHandlerFailsClosedWithoutLeakingDetails(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		runner *fakeBuiltinRecipeRunner
		status int
		code   string
	}{
		{"command failure", &fakeBuiltinRecipeRunner{listErr: errors.New("/private/install secret-token")}, http.StatusServiceUnavailable, "builtin_recipes_unavailable"},
		{"invalid json", &fakeBuiltinRecipeRunner{listing: []byte("/private/install")}, http.StatusBadGateway, "builtin_recipes_contract_invalid"},
		{"missing version", &fakeBuiltinRecipeRunner{listing: []byte(`{"bundles":[]}`)}, http.StatusBadGateway, "builtin_recipes_contract_invalid"},
		{"negative minimum", &fakeBuiltinRecipeRunner{listing: []byte(strings.Replace(builtinRecipeListing, `"minimum_candidates": 1`, `"minimum_candidates": -1`, 1))}, http.StatusBadGateway, "builtin_recipes_contract_invalid"},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			BuiltinRecipesHandler(test.runner).ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/recipes/builtin", nil))
			var body BuiltinRecipeErrorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if response.Code != test.status || body.Error != test.code {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			if strings.Contains(response.Body.String(), "private") || strings.Contains(response.Body.String(), "secret") {
				t.Fatalf("error leaked runner details: %s", response.Body.String())
			}
		})
	}
}

func TestBuiltinRecipeComposeHandlerReturnsTheComposedConfig(t *testing.T) {
	t.Parallel()

	composed := []byte(`{"version":"v0.3","entrypoints":[{"model_names":["my/vault","vault"],"recipe":"vault"}]}`)
	runner := &fakeBuiltinRecipeRunner{composed: composed}
	body := `{"bundle":"mom-v1","recipe":"vault","bindings":{"private":[{"model":"model-a","use_reasoning":false}]},"model_names":["my/vault","vault"]}`
	response := httptest.NewRecorder()
	BuiltinRecipeComposeHandler(runner, "/state/config.yaml", false).ServeHTTP(response,
		httptest.NewRequest(http.MethodPost, "/api/recipes/builtin/compose", strings.NewReader(body)))

	if response.Code != http.StatusOK || response.Body.String() != string(composed) ||
		response.Header().Get("Content-Type") != "application/json" || response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("status=%d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}
	want := BuiltinRecipeComposeRequest{
		Bundle:     "mom-v1",
		Recipe:     "vault",
		Bindings:   map[string][]map[string]any{"private": {{"model": "model-a", "use_reasoning": false}}},
		ModelNames: []string{"my/vault", "vault"},
	}
	if runner.configPath != "/state/config.yaml" || !reflect.DeepEqual(*runner.request, want) {
		t.Fatalf("runner received %q %+v", runner.configPath, runner.request)
	}
}

func TestBuiltinRecipeComposeHandlerSeparatesRejectionsFromFailures(t *testing.T) {
	t.Parallel()

	const valid = `{"bundle":"mom-v1","recipe":"vault","bindings":{},"model_names":["my/vault"]}`
	for _, test := range []struct {
		name     string
		method   string
		body     string
		readonly bool
		err      error
		status   int
		code     string
		message  string
		composes bool
	}{
		{name: "rejection", body: valid, err: &BuiltinRecipeRejection{Message: "Config already contains recipe 'vault'"}, status: http.StatusBadRequest, code: "recipe_rejected", message: "Config already contains recipe 'vault'", composes: true},
		{name: "failure", body: valid, err: errors.New("cli.builtin_recipes_json failed: /private/install"), status: http.StatusServiceUnavailable, code: "builtin_recipes_unavailable", composes: true},
		{name: "unknown field", body: `{"config_path":"/etc/passwd","bundle":"mom-v1","recipe":"vault","bindings":{},"model_names":["x"]}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "missing model names", body: `{"bundle":"mom-v1","recipe":"vault","bindings":{}}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "malformed", body: `{"bundle":`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "readonly", body: valid, readonly: true, status: http.StatusForbidden, code: "readonly_mode"},
		{name: "method", method: http.MethodGet, status: http.StatusMethodNotAllowed, code: "method_not_allowed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			runner := &fakeBuiltinRecipeRunner{composeErr: test.err}
			method := test.method
			if method == "" {
				method = http.MethodPost
			}
			response := httptest.NewRecorder()
			BuiltinRecipeComposeHandler(runner, "config.yaml", test.readonly).ServeHTTP(response,
				httptest.NewRequest(method, "/api/recipes/builtin/compose", strings.NewReader(test.body)))
			var body BuiltinRecipeErrorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if response.Code != test.status || body.Error != test.code || (test.message != "" && body.Message != test.message) {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			if strings.Contains(body.Message, "/private/") {
				t.Fatalf("error leaked runner details: %s", response.Body.String())
			}
			if (runner.request != nil) != test.composes {
				t.Fatalf("runner called=%v, want %v", runner.request != nil, test.composes)
			}
		})
	}
}

func TestPackagedBuiltinRecipeRunnerUsesIsolatedModule(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake executable contract uses a POSIX shell")
	}

	directory := t.TempDir()
	requestPath := filepath.Join(directory, "request.json")
	executable := filepath.Join(directory, "python3")
	script := `#!/bin/sh
set -eu
[ "$#" -eq 3 ]
[ "$1" = "-m" ]
[ "$2" = "cli.builtin_recipes_json" ]
[ ! -e config.yaml ]
case "$3" in
list) printf '{"catalog_version":"latest","bundles":[]}' ;;
compose)
  cat > "$BUILTIN_RECIPE_TEST_REQUEST_PATH"
  case "$BUILTIN_RECIPE_TEST_OUTCOME" in
  rejected) echo "Decision 'private' references unconfigured provider models: ['invented']"; exit 65 ;;
  crashed) echo "Traceback: /private/install" >&2; exit 1 ;;
  *) printf '{"version":"v0.3"}' ;;
  esac ;;
*) exit 2 ;;
esac
`
	if err := os.WriteFile(executable, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake CLI: %v", err)
	}
	t.Setenv("BUILTIN_RECIPE_TEST_REQUEST_PATH", requestPath)
	runner := NewPackagedBuiltinRecipeRunner(executable)

	listing, err := runner.List(context.Background())
	if err != nil || string(listing) != `{"catalog_version":"latest","bundles":[]}` {
		t.Fatalf("list = %s, %v", listing, err)
	}

	request := BuiltinRecipeComposeRequest{Bundle: "mom-v1", Recipe: "vault", Bindings: map[string][]map[string]any{}, ModelNames: []string{"my/vault"}}
	t.Setenv("BUILTIN_RECIPE_TEST_OUTCOME", "composed")
	composed, err := runner.Compose(context.Background(), "/state/config.yaml", request)
	if err != nil || string(composed) != `{"version":"v0.3"}` {
		t.Fatalf("compose = %s, %v", composed, err)
	}
	sent, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(sent) != `{"config_path":"/state/config.yaml","bundle":"mom-v1","recipe":"vault","bindings":{},"model_names":["my/vault"]}` {
		t.Fatalf("module request = %s", sent)
	}

	t.Setenv("BUILTIN_RECIPE_TEST_OUTCOME", "rejected")
	_, err = runner.Compose(context.Background(), "/state/config.yaml", request)
	var rejection *BuiltinRecipeRejection
	if !errors.As(err, &rejection) || rejection.Message != "Decision 'private' references unconfigured provider models: ['invented']" {
		t.Fatalf("rejection = %v", err)
	}

	t.Setenv("BUILTIN_RECIPE_TEST_OUTCOME", "crashed")
	_, err = runner.Compose(context.Background(), "/state/config.yaml", request)
	if err == nil || errors.As(err, &rejection) {
		t.Fatalf("crash = %v, want a failure that is not a rejection", err)
	}
}
