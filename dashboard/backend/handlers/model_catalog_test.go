package handlers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

type fakeModelCatalogSource struct {
	mu      sync.Mutex
	payload []byte
	err     error
	calls   int
}

func (source *fakeModelCatalogSource) Load(context.Context) ([]byte, error) {
	source.mu.Lock()
	defer source.mu.Unlock()
	source.calls++
	return source.payload, source.err
}

func (source *fakeModelCatalogSource) callCount() int {
	source.mu.Lock()
	defer source.mu.Unlock()
	return source.calls
}

func TestModelCatalogHandlerReturnsCLIContractAndCachesSuccess(t *testing.T) {
	t.Parallel()

	source := &fakeModelCatalogSource{payload: []byte(validModelCatalogPayload(`,
    "configured": {"path":"/private/config.yaml","api_key":"must-not-render"}`))}
	handler := ModelCatalogHandler(source)

	for attempt := 0; attempt < 2; attempt++ {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil))
		if response.Code != http.StatusOK {
			t.Fatalf("attempt %d status=%d body=%s", attempt, response.Code, response.Body.String())
		}
		body := response.Body.String()
		for _, expected := range []string{
			`"catalog_version":"latest"`,
			`"channel":"latest"`,
			`"id":"vllm-sr/chorus-v1"`,
			`"verification":{"status":"verified","authority":"vllm-sr-maintainers"`,
			`"recommended_pool":["local/example"]`,
		} {
			if !strings.Contains(body, expected) {
				t.Fatalf("response omitted %s: %s", expected, body)
			}
		}
		if strings.Contains(body, "configured") || strings.Contains(body, "must-not-render") || strings.Contains(body, "/private/") {
			t.Fatalf("response leaked configured runtime data: %s", body)
		}
	}
	if source.callCount() != 1 {
		t.Fatalf("source calls=%d want=1", source.callCount())
	}
}

func TestModelCatalogHandlerFailsClosedWithoutLeakingSourceErrors(t *testing.T) {
	t.Parallel()

	source := &fakeModelCatalogSource{err: errors.New("private-command-detail secret-token")}
	response := httptest.NewRecorder()
	ModelCatalogHandler(source).ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil))

	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `"error":"catalog_unavailable"`) {
		t.Fatalf("missing stable error code: %s", response.Body.String())
	}
	if strings.Contains(response.Body.String(), "private-command-detail") || strings.Contains(response.Body.String(), "secret-token") {
		t.Fatalf("source error leaked: %s", response.Body.String())
	}
}

func TestModelCatalogHandlerRejectsUnknownNestedFieldsWithoutLeakingThem(t *testing.T) {
	t.Parallel()

	const canary = "nested-catalog-secret-canary"
	payload := strings.Replace(
		validModelCatalogPayload(","),
		`"description":"Balanced routing."`,
		`"description":"Balanced routing.","api_key":"`+canary+`"`,
		1,
	)
	response := httptest.NewRecorder()
	ModelCatalogHandler(&fakeModelCatalogSource{payload: []byte(payload)}).ServeHTTP(
		response,
		httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil),
	)

	if response.Code != http.StatusBadGateway {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), canary) || strings.Contains(response.Body.String(), "api_key") {
		t.Fatalf("response leaked an unknown nested field: %s", response.Body.String())
	}
}

func TestModelCatalogHandlerRejectsMalformedCLIContract(t *testing.T) {
	t.Parallel()

	for name, payload := range map[string]string{
		"invalid json":         `{`,
		"empty inventory":      `{"catalogs":[],"models":[]}`,
		"missing protocols":    validModelCatalogPayload(","),
		"missing roles":        validModelCatalogPayload(","),
		"missing authority":    validModelCatalogPayload(","),
		"invalid asset digest": validModelCatalogPayload(","),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if name == "missing protocols" {
				payload = strings.Replace(payload, `"protocols":["openai_chat"]`, `"protocols":[]`, 1)
			}
			if name == "missing roles" {
				payload = strings.Replace(payload, `"roles":[{"name":"balanced","required":true,"minimum_candidates":1,"traits":["chat"],"recommended_pool":["local/example"]}]`, `"roles":[]`, 1)
			}
			if name == "missing authority" {
				payload = strings.Replace(payload, `"authority":"vllm-sr-maintainers"`, `"authority":""`, 1)
			}
			if name == "invalid asset digest" {
				payload = strings.Replace(payload, `"asset_sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"`, `"asset_sha256":"sha256:not-a-digest"`, 1)
			}
			response := httptest.NewRecorder()
			ModelCatalogHandler(&fakeModelCatalogSource{payload: []byte(payload)}).ServeHTTP(
				response,
				httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil),
			)
			if response.Code != http.StatusBadGateway {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestModelCatalogHandlerEnforcesCanonicalReadOnlyRoute(t *testing.T) {
	t.Parallel()

	handler := ModelCatalogHandler(&fakeModelCatalogSource{payload: []byte(validModelCatalogPayload(","))})

	methodResponse := httptest.NewRecorder()
	handler.ServeHTTP(methodResponse, httptest.NewRequest(http.MethodPost, "/api/models/catalog", nil))
	if methodResponse.Code != http.StatusMethodNotAllowed || methodResponse.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("POST status=%d allow=%q", methodResponse.Code, methodResponse.Header().Get("Allow"))
	}

	pathResponse := httptest.NewRecorder()
	handler.ServeHTTP(pathResponse, httptest.NewRequest(http.MethodGet, "/api/models/catalog/extra", nil))
	if pathResponse.Code != http.StatusNotFound {
		t.Fatalf("nested path status=%d", pathResponse.Code)
	}
}

func TestCLIModelCatalogSourceUsesCatalogOnlyCommandFromIsolatedDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake executable contract uses a POSIX shell")
	}

	executable := filepath.Join(t.TempDir(), "python3")
	script := `#!/bin/sh
set -eu
[ "$#" -eq 8 ]
[ "$1" = "-m" ]
[ "$2" = "cli.main" ]
[ "$3" = "model" ]
[ "$4" = "list" ]
[ "$5" = "--all-versions" ]
[ "$6" = "--all" ]
[ "$7" = "--output" ]
[ "$8" = "json" ]
[ ! -e config.yaml ]
printf '%s' "$MODEL_CATALOG_TEST_PAYLOAD"
`
	if err := os.WriteFile(executable, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake CLI: %v", err)
	}
	t.Setenv("MODEL_CATALOG_TEST_PAYLOAD", validModelCatalogPayload(""))

	payload, err := NewCLIModelCatalogSource(executable).Load(context.Background())
	if err != nil {
		t.Fatalf("load catalog through real command seam: %v", err)
	}
	if _, err := normalizeModelCatalogDocument(payload); err != nil {
		t.Fatalf("normalize command payload: %v", err)
	}
}

func validModelCatalogPayload(extra string) string {
	return `{
  "catalogs":[{"catalog_version":"latest","channel":"latest","default_model":"vllm-sr/chorus-v1","enabled_models":["vllm-sr/chorus-v1"]}],
  "models":[{
    "id":"vllm-sr/chorus-v1",
    "display_name":"Chorus V1",
    "description":"Balanced routing.",
    "kind":"virtual",
    "family":"chorus",
    "generation":1,
    "policy_version":"1.0.0",
    "entrypoint":"vllm-sr/chorus-v1",
    "recipe":"balance",
    "protocols":["openai_chat"],
    "traits":["balanced","chat"],
    "roles":[{"name":"balanced","required":true,"minimum_candidates":1,"traits":["chat"],"recommended_pool":["local/example"]}],
    "catalog_version":"latest",
    "channel":"latest",
    "compatible":true,
    "compatibility_reason":"compatible",
    "enabled_by_default":true,
    "default":true,
    "verification":{"status":"verified","authority":"vllm-sr-maintainers","asset_sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
  }]` + extra + `}`
}
