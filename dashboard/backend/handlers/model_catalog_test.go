package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
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

	payload := []byte(validModelCatalogPayload(`,
		"configured": {"path":"/private/config.yaml","api_key":"must-not-render"}`))
	if _, err := normalizeModelCatalogDocument(payload); err != nil {
		t.Fatalf("valid fixture rejected: %v", err)
	}
	source := &fakeModelCatalogSource{payload: payload}
	handler := ModelCatalogHandler(source)

	for attempt := 0; attempt < 2; attempt++ {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil))
		if response.Code != http.StatusOK {
			t.Fatalf("attempt %d status=%d body=%s", attempt, response.Code, response.Body.String())
		}
		body := response.Body.String()
		for _, expected := range []string{
			`"schema_version":"vllm-sr/model-catalog/v2"`,
			`"catalog_version":"latest"`,
			`"channel":"latest"`,
			`"id":"vllm-sr/mom-v1-blend"`,
			`"verification":{"authority":"vllm-sr-maintainers","status":"reproduced"`,
			`"id":"openai"`,
			`"id":"openai/chat-completions@1"`,
			`"id":"example/benchmark@1.0.0"`,
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
		"invalid json":              `{`,
		"empty inventory":           `{"schema_version":"vllm-sr/model-catalog/v2","catalogs":[],"protocols":[],"providers":[],"reasoning_families":[],"models":[],"benchmarks":[],"evaluations":[],"evaluation_coverage":[],"indices":[],"index_results":[]}`,
		"missing protocols":         validModelCatalogPayload(","),
		"missing default base path": validModelCatalogPayload(","),
		"missing roles":             validModelCatalogPayload(","),
		"missing authority":         validModelCatalogPayload(","),
		"invalid asset digest":      validModelCatalogPayload(","),
		"orphan physical model":     validModelCatalogPayload(","),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if name == "missing protocols" {
				payload = strings.Replace(payload, `"protocols":["openai/chat-completions@1"]`, `"protocols":[]`, 1)
			}
			if name == "missing default base path" {
				payload = strings.Replace(payload, `    "default_base_path":"/v1",
`, "", 1)
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
			if name == "orphan physical model" {
				payload = strings.Replace(payload, `"models":[{`, `"models":[{
    "id":"example/physical",
    "display_name":"Example Physical",
    "description":"Physical model without a provider model.",
    "kind":"physical",
    "publisher":"Example",
    "presentation":{"logo":"package:example","monogram":"E","monochrome":true},
    "distribution":{"type":"open_weights","source":"https://models.example/model","license":"Apache-2.0"},
    "family":"example",
    "lifecycle":"active",
    "capabilities":["chat"],
    "modalities":{"input":["text"],"output":["text"]},
    "verification":{"status":"claimed","authority":"Example","verified_at":"2026-09-05","source":"https://models.example/model"}
  },{`, 1)
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

func TestCatalogProviderBindingRelationshipIsClosed(t *testing.T) {
	t.Parallel()

	provider := modelcatalog.ProviderDefinition{
		ID:        "example",
		Protocols: []string{"openai/chat-completions@1"},
		Models: []modelcatalog.CatalogModelBinding{{
			Catalog:      "example/model",
			Relationship: modelcatalog.CatalogModelRelationshipGateway,
			ID:           "example/model",
			Protocols:    []string{"openai/chat-completions@1"},
			Lifecycle:    "active",
			Verification: modelcatalog.CatalogBindingVerification{Status: "claimed"},
		}},
	}
	models := map[string]struct{}{"example/model": {}}
	protocols := map[string]struct{}{"openai/chat-completions@1": {}}
	definitions := []modelcatalog.ModelCard{{ID: "example/model", Kind: "physical", Lifecycle: "active"}}
	if err := validateCatalogProviderBindings(
		[]modelcatalog.ProviderDefinition{provider}, models, protocols, definitions, nil,
	); err != nil {
		t.Fatalf("valid relationship rejected: %v", err)
	}

	provider.Models[0].Relationship = "brokered"
	if err := validateCatalogProviderBindings(
		[]modelcatalog.ProviderDefinition{provider}, models, protocols, definitions, nil,
	); err == nil || !strings.Contains(err.Error(), "malformed provider catalog model") {
		t.Fatalf("unsupported relationship accepted: %v", err)
	}
}

func TestCatalogProviderBindingMustProjectCompleteReasoningContract(t *testing.T) {
	t.Parallel()

	provider := modelcatalog.ProviderDefinition{
		ID:                 "example",
		Protocols:          []string{"openai/chat-completions@1"},
		ReasoningTransport: modelcatalog.ReasoningTransportThinkingObject,
		Models: []modelcatalog.CatalogModelBinding{{
			Catalog:      "example/model",
			Relationship: modelcatalog.CatalogModelRelationshipFirstParty,
			ID:           "example-model",
			Protocols:    []string{"openai/chat-completions@1"},
			Lifecycle:    "active",
			Verification: modelcatalog.CatalogBindingVerification{Status: "claimed"},
		}},
	}
	models := map[string]struct{}{"example/model": {}}
	protocols := map[string]struct{}{"openai/chat-completions@1": {}}
	definitions := []modelcatalog.ModelCard{{
		ID: "example/model", Kind: "physical", Lifecycle: "active", ReasoningFamily: "effort",
	}}
	reasoning := map[string]modelcatalog.ReasoningFamilyDefinition{
		"effort": {
			ID: "effort", Type: "reasoning_effort", Parameter: "reasoning_effort",
			Levels: []string{"low", "high"}, Default: "high",
			Modes: []string{"enabled"}, DefaultMode: "enabled",
		},
	}

	if err := validateCatalogProviderBindings(
		[]modelcatalog.ProviderDefinition{provider}, models, protocols, definitions, reasoning,
	); err == nil || !strings.Contains(err.Error(), "malformed provider catalog model") {
		t.Fatalf("switch-only transport accepted effort contract: %v", err)
	}

	provider.Models[0].ReasoningTransport = modelcatalog.ReasoningTransportThinkingEffort
	if err := validateCatalogProviderBindings(
		[]modelcatalog.ProviderDefinition{provider}, models, protocols, definitions, reasoning,
	); err != nil {
		t.Fatalf("complete reasoning transport rejected: %v", err)
	}
}

func TestCatalogHTTPSURLsAllowDocumentFragments(t *testing.T) {
	t.Parallel()

	for _, value := range []string{
		"https://huggingface.co/example/model#evaluation-results",
		"https://example.test/model-card?revision=1#benchmarks",
	} {
		if !validHTTPSURL(value) {
			t.Fatalf("valid source URL with document fragment rejected: %s", value)
		}
	}
	for _, value := range []string{
		"http://example.test/model-card#benchmarks",
		"https://user:secret@example.test/model-card#benchmarks",
		"#benchmarks",
	} {
		if validHTTPSURL(value) {
			t.Fatalf("unsafe or non-absolute source URL accepted: %s", value)
		}
	}
	if validHTTPSTransportURL("https://api.example.test/v1#documentation") {
		t.Fatal("provider transport URL with a fragment was accepted")
	}
	if !validHTTPSTransportURL("https://api.example.test/v1") {
		t.Fatal("valid provider transport URL was rejected")
	}
	for _, value := range []string{
		"https://:443/v1",
		"https://example.test:bad/v1",
		"https://example.test:65536/v1",
		"https://example.test:0/v1",
		"\nhttps://example.test/v1",
		"https://example.test/v1 ",
	} {
		if validHTTPSTransportURL(value) {
			t.Fatalf("malformed provider transport URL accepted: %q", value)
		}
	}
}

func TestGeneratedPublicModelCatalogSatisfiesDashboardContract(t *testing.T) {
	t.Parallel()

	repositoryRoot := filepath.Clean(filepath.Join(packageWorkingDirectory(t), "..", "..", ".."))
	payload, err := os.ReadFile(filepath.Join(repositoryRoot, "website", "static", "model-catalog", "catalog.json"))
	if err != nil {
		t.Fatalf("read generated public catalog: %v", err)
	}
	normalized, err := normalizeModelCatalogDocument(payload)
	if err != nil {
		t.Fatalf("generated public catalog violates Dashboard API contract: %v", err)
	}
	var document modelCatalogEnvelope
	if unmarshalErr := json.Unmarshal(normalized, &document); unmarshalErr != nil {
		t.Fatalf("decode normalized public catalog: %v", unmarshalErr)
	}
	if len(document.Models) != 88 || len(document.Providers) != 60 || len(document.Evaluations) != 1360 {
		t.Fatalf(
			"unexpected generated inventory: models=%d providers=%d evaluations=%d",
			len(document.Models),
			len(document.Providers),
			len(document.Evaluations),
		)
	}
}

func TestDashboardRejectsAvailableEvaluationWithoutValidCalendarAnchor(t *testing.T) {
	t.Parallel()

	repositoryRoot := filepath.Clean(filepath.Join(packageWorkingDirectory(t), "..", "..", ".."))
	payload, err := os.ReadFile(filepath.Join(repositoryRoot, "website", "static", "model-catalog", "catalog.json"))
	if err != nil {
		t.Fatalf("read generated public catalog: %v", err)
	}
	var document map[string]any
	if unmarshalErr := json.Unmarshal(payload, &document); unmarshalErr != nil {
		t.Fatalf("decode generated public catalog: %v", unmarshalErr)
	}
	evaluations := document["evaluations"].([]any)
	first := evaluations[0].(map[string]any)
	delete(first, "measured_at")
	delete(first, "observed_at")
	withoutAnchor, err := json.Marshal(document)
	if err != nil {
		t.Fatalf("encode catalog without anchor: %v", err)
	}
	if _, normalizeErr := normalizeModelCatalogDocument(withoutAnchor); normalizeErr == nil {
		t.Fatal("available evaluation without a calendar anchor was accepted")
	}

	first["observed_at"] = "2026-09-31"
	invalidAnchor, err := json.Marshal(document)
	if err != nil {
		t.Fatalf("encode catalog with invalid anchor: %v", err)
	}
	if _, normalizeErr := normalizeModelCatalogDocument(invalidAnchor); normalizeErr == nil {
		t.Fatal("available evaluation with an invalid calendar date was accepted")
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

func TestPackagedModelCatalogSourceUsesIsolatedExporter(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake executable contract uses a POSIX shell")
	}

	const formerCatalogLimit = 4 << 20
	largeDescription := strings.Repeat("x", formerCatalogLimit)
	payloadPath := filepath.Join(t.TempDir(), "catalog.json")
	largePayload := strings.Replace(
		validModelCatalogPayload(""),
		"Balanced routing.",
		largeDescription,
		1,
	)
	if len(largePayload) <= formerCatalogLimit {
		t.Fatalf("large fixture is %d bytes, want more than %d", len(largePayload), formerCatalogLimit)
	}
	if err := os.WriteFile(payloadPath, []byte(largePayload), 0o600); err != nil {
		t.Fatalf("write large catalog fixture: %v", err)
	}

	executable := filepath.Join(t.TempDir(), "python3")
	script := `#!/bin/sh
set -eu
[ "$#" -eq 2 ]
[ "$1" = "-m" ]
[ "$2" = "cli.model_catalog_export" ]
[ ! -e config.yaml ]
cat "$MODEL_CATALOG_TEST_PAYLOAD_PATH"
`
	if err := os.WriteFile(executable, []byte(script), 0o700); err != nil {
		t.Fatalf("write fake CLI: %v", err)
	}
	t.Setenv("MODEL_CATALOG_TEST_PAYLOAD_PATH", payloadPath)

	payload, err := NewPackagedModelCatalogSource(executable).Load(context.Background())
	if err != nil {
		t.Fatalf("load catalog through real command seam: %v", err)
	}
	if len(payload) <= formerCatalogLimit {
		t.Fatalf("loaded payload is %d bytes, want more than %d", len(payload), formerCatalogLimit)
	}
	if _, err := normalizeModelCatalogDocument(payload); err != nil {
		t.Fatalf("normalize command payload: %v", err)
	}
}

func validModelCatalogPayload(extra string) string {
	return `{
  "schema_version":"vllm-sr/model-catalog/v2",
  "catalogs":[{"catalog_version":"latest","channel":"latest","default_model":"vllm-sr/mom-v1-blend","enabled_models":["vllm-sr/mom-v1-blend"],"default_intelligence_index":"example/index@1.0.0"}],
  "protocols":[{
    "id":"openai/chat-completions@1",
    "display_name":"OpenAI Chat Completions",
    "wire_format":"openai.chat.v1",
    "default_base_path":"/v1",
    "operations":[{"id":"create","method":"POST","path":"/v1/chat/completions"}],
    "capabilities":["chat"]
  }],
  "providers":[{
    "id":"openai",
    "display_name":"OpenAI",
    "description":"OpenAI API.",
    "category":"model_api",
    "support_tier":"native",
    "default_base_url":"https://api.openai.com/v1",
    "protocols":["openai/chat-completions@1"],
    "default_protocol":"openai/chat-completions@1",
    "supported_operations":["openai/chat-completions@1#create"],
    "reasoning_transport":"output_config_effort",
    "auth":{"strategy":"bearer","header":"Authorization","prefix":"Bearer"},
    "presentation":{"logo":"package:openai","monogram":"O","monochrome":true},
    "conformance":{"status":"fixture_verified","verified_at":"2026-09-04"}
  }],
  "reasoning_families":[],
  "models":[{
    "id":"vllm-sr/mom-v1-blend",
    "display_name":"MoM V1 Blend",
    "description":"Balanced routing.",
    "kind":"virtual",
    "publisher":"vllm-sr.ai",
    "presentation":{"logo":"package:vllm","monogram":"V","monochrome":true},
    "distribution":{"type":"router_recipe","source":"https://vllm-sr.ai/models"},
    "family":"mom",
    "generation":1,
    "policy_version":"1.0.0",
    "asset":"mom-v1",
    "entrypoint":"vllm-sr/mom-v1-blend",
    "recipe":"balance",
    "lifecycle":"active",
    "capabilities":["chat"],
    "modalities":{"input":["text"],"output":["text"]},
    "traits":["balanced","chat"],
    "roles":[{"name":"balanced","required":true,"minimum_candidates":1,"traits":["chat"],"recommended_pool":["local/example"]}],
    "verification":{"status":"reproduced","authority":"vllm-sr-maintainers","verified_at":"2026-09-04","asset_sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
  }],
  "evaluation_coverage":[],
  "benchmarks":[{
    "id":"example/benchmark@1.0.0",
    "display_name":"Example Benchmark",
    "domain":"general",
    "default_profile":"published-standard",
    "profiles":[{"id":"published-standard","display_name":"Published standard","description":"Test profile."}],
    "metrics":[{"id":"score","unit":"proportion","direction":"higher_is_better","range":[0,1]}]
  }],
  "evaluations":[],
  "indices":[{
    "id":"example/index@1.0.0",
    "display_name":"Example Index",
    "description":"Test index.",
    "aggregation":"weighted_mean",
    "scale":[0,100],
    "missing":{"policy":"require_all"},
    "domains":{"general":1},
    "components":[{"benchmark":"example/benchmark@1.0.0","metric":"score","benchmark_profile":"published-standard","weight":1,"normalization":{"type":"identity"}}]
  }],
  "index_results":[{
    "model":"vllm-sr/mom-v1-blend",
    "reasoning_effort":"default",
    "index":"example/index@1.0.0",
    "status":"not_applicable",
    "score":null,
    "coverage":0,
    "components":[],
    "provenance":[]
  }]` + extra + `}`
}
