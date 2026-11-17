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

	composed := []byte(`{"version":"v0.5","entrypoints":[{"model_names":["my/vault","vault"],"recipe":"vault"}]}`)
	runner := &fakeBuiltinRecipeRunner{composed: composed}
	body := `{"bundle":"mom-v1","recipe":"vault","bindings":{"private":[{"model":"model-a","use_reasoning":false}]},"model_names":["my/vault","vault"]}`
	response := httptest.NewRecorder()
	BuiltinRecipeComposeHandler(runner, "/state/config.yaml", false).ServeHTTP(response,
		httptest.NewRequest(http.MethodPost, "/api/recipes/builtin/compose", strings.NewReader(body)))

	var actual, expected any
	if err := json.Unmarshal(response.Body.Bytes(), &actual); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(composed, &expected); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || !reflect.DeepEqual(actual, expected) ||
		response.Header().Get("Content-Type") != "application/json" || !strings.Contains(response.Header().Get("Cache-Control"), "no-store") {
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
  *) printf '{"version":"v0.5"}' ;;
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
	if err != nil || string(composed) != `{"version":"v0.5"}` {
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

func TestBuiltinRecipeComposeRedactsConfigSecrets(t *testing.T) {
	composed := []byte(`{"version":"v0.5","listeners":[{"api_keys":["listener-fixture"]}],"providers":{"models":[{"backend_refs":[{"auth":{"api_key":"backend-fixture"},"headers":{"Authorization":"header-fixture"},"endpoint":"https://user:url-fixture@example.test/v1"}]}]},"global":{"services":{"response_api":{"redis":{"password":"redis-fixture"}}}}}`)
	for _, privileged := range []bool{false, true} {
		t.Run(map[bool]string{false: "editor", true: "secret viewer"}[privileged], func(t *testing.T) {
			runner := &fakeBuiltinRecipeRunner{composed: composed}
			permissions := []string{"config.write"}
			if privileged {
				permissions = append(permissions, configSecretPermission)
			}
			request := httptest.NewRequest(http.MethodPost, builtinRecipeComposePath, strings.NewReader(`{"bundle":"mom-v1","recipe":"vault","bindings":{},"model_names":["my/vault"]}`))
			request = request.WithContext(requestAs("/", permissions...).Context())
			response := httptest.NewRecorder()
			BuiltinRecipeComposeHandler(runner, "config.yaml", false)(response, request)
			if response.Code != http.StatusOK || !json.Valid(response.Body.Bytes()) {
				t.Fatalf("compose status=%d body=%s", response.Code, response.Body.String())
			}
			for _, secret := range []string{"listener-fixture", "backend-fixture", "header-fixture", "url-fixture", "redis-fixture"} {
				if visible := strings.Contains(response.Body.String(), secret); visible != privileged {
					t.Errorf("credential %s visible=%v, want %v", secret, visible, privileged)
				}
			}
			if !strings.Contains(response.Header().Get("Cache-Control"), "no-store") {
				t.Fatal("composed config can be cached")
			}
		})
	}
}

func TestBuiltinRecipeComposeRedactedConfigCanBeSaved(t *testing.T) {
	for _, retarget := range []bool{false, true} {
		t.Run(map[bool]string{false: "unchanged targets", true: "changed target"}[retarget], func(t *testing.T) {
			path := writeSecretConfig(t, configSecretFixture)
			original, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			node, err := configSecretDocument(original)
			if err != nil {
				t.Fatal(err)
			}
			recipes := mappingValueNode(node, "recipes")
			added := cloneYAMLNode(recipes.Content[0])
			mappingValueNode(added, "name").Value = "composed-recipe"
			recipes.Content = append(recipes.Content, added)
			published, err := configSecretDocument([]byte("entrypoints:\n  - model_names: [default]\n    recipe: default\n    default: true\n  - model_names: [composed]\n    recipe: composed-recipe\n"))
			if err != nil {
				t.Fatal(err)
			}
			setMappingValueNode(node, "entrypoints", mappingValueNode(published, "entrypoints"))
			composed, err := marshalYAMLTaggedJSON(node)
			if err != nil {
				t.Fatal(err)
			}
			runner := &fakeBuiltinRecipeRunner{composed: composed}
			request := httptest.NewRequest(http.MethodPost, builtinRecipeComposePath, strings.NewReader(`{"bundle":"mom-v1","recipe":"composed-recipe","bindings":{},"model_names":["composed"]}`))
			request = request.WithContext(requestAs("/", "config.write").Context())
			view := httptest.NewRecorder()
			BuiltinRecipeComposeHandler(runner, path, false)(view, request)
			if view.Code != http.StatusOK || strings.Contains(view.Body.String(), configSecretFixture) {
				t.Fatalf("compose did not return a redacted config: status=%d", view.Code)
			}
			body := view.Body.String()
			if retarget {
				body = strings.ReplaceAll(body, "127.0.0.1:6379", "different.example.test:6379")
			}
			previous := propagateConfig
			calls := 0
			propagateConfig = func(string, string) error { calls++; return nil }
			t.Cleanup(func() { propagateConfig = previous })
			request = httptest.NewRequest(http.MethodPut, "/", strings.NewReader(body))
			request = request.WithContext(requestAs("/", "config.write").Context())
			response := httptest.NewRecorder()
			UpdateConfigHandler(path, false, filepath.Dir(path))(response, request)
			saved, readErr := os.ReadFile(path)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if retarget {
				if response.Code != http.StatusBadRequest || string(saved) != string(original) || calls != 0 {
					t.Fatalf("retargeted composition was saved: status=%d calls=%d", response.Code, calls)
				}
				return
			}
			if response.Code != http.StatusOK {
				t.Fatalf("composed save status=%d: %s", response.Code, response.Body.String())
			}
			if !strings.Contains(string(saved), "composed-recipe") || !strings.Contains(string(saved), configSecretFixture) || strings.Contains(string(saved), redactedConfigSecret) {
				t.Fatal("composed save lost the recipe or original credential")
			}
		})
	}
}

func TestBuiltinRecipeComposeRedactionFailureHidesDetails(t *testing.T) {
	// Duplicate keys fail redaction; parser errors may quote their contents.
	runner := &fakeBuiltinRecipeRunner{composed: []byte("password: first\npassword: " + configSecretFixture + "\n")}
	request := httptest.NewRequest(http.MethodPost, builtinRecipeComposePath, strings.NewReader(`{"bundle":"mom-v1","recipe":"vault","bindings":{},"model_names":["my/vault"]}`))
	request = request.WithContext(requestAs("/", "config.write").Context())
	response := httptest.NewRecorder()
	BuiltinRecipeComposeHandler(runner, "/private/config.yaml", false)(response, request)
	var body BuiltinRecipeErrorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusServiceUnavailable || body.Error != "builtin_recipes_unavailable" || body.Message != "The composed recipe configuration could not be prepared." {
		t.Fatalf("redaction failure exposed unexpected response: status=%d", response.Code)
	}
	if strings.Contains(response.Body.String(), configSecretFixture) || strings.Contains(response.Body.String(), "/private/") {
		t.Fatal("redaction failure exposed secret or filesystem details")
	}
}
