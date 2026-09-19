//go:build !windows && cgo

package apiserver

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestScrubSecretsInErrorMessage(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "yaml api_key",
			in:   `failed to parse: api_key: "plain-secret-value" near line 12`,
			want: `failed to parse: api_key: [REDACTED] near line 12`,
		},
		{
			name: "json password",
			in:   `invalid config: {"password":"db-pass-canary"}`,
			want: `invalid config: {"password":[REDACTED]}`,
		},
		{
			name: "keeps api_key_env",
			in:   `missing env for api_key_env: OPENAI_API_KEY`,
			want: `missing env for api_key_env: OPENAI_API_KEY`,
		},
		{
			name: "unquoted password",
			in:   `Backup config is invalid: password: s3cretValue`,
			want: `Backup config is invalid: password: [REDACTED]`,
		},
		{
			name: "auth token",
			in:   `invalid config: auth_token: token-canary`,
			want: `invalid config: auth_token: [REDACTED]`,
		},
		{
			name: "authorization header",
			in:   `invalid header: Authorization: Bearer bearer-canary`,
			want: `invalid header: Authorization: [REDACTED]`,
		},
		{
			name: "x api key header",
			in:   `invalid header: x-api-key=header-canary`,
			want: `invalid header: x-api-key=[REDACTED]`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := scrubSecretsInErrorMessage(tc.in)
			if got != tc.want {
				t.Fatalf("scrubSecretsInErrorMessage() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestWriteErrorResponseScrubsSecrets(t *testing.T) {
	server := &ClassificationAPIServer{}
	rr := httptest.NewRecorder()
	server.writeErrorResponse(
		rr,
		http.StatusBadRequest,
		"PARSE_ERROR",
		`Failed to parse config: api_key: "redact-me-canary-value-0001"`,
	)
	body := rr.Body.String()
	if strings.Contains(body, "redact-me-canary-value-0001") {
		t.Fatalf("error response leaked canary: %s", body)
	}
	if !strings.Contains(body, redactedConfigValue) {
		t.Fatalf("expected redacted placeholder in error body, got %s", body)
	}
}

func TestRedactSensitiveConfigValue(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"providers": map[string]interface{}{
			"models": []interface{}{
				map[string]interface{}{
					"name": "m1",
					"backend_refs": []interface{}{
						map[string]interface{}{
							"api_key":     "plain-secret",
							"api_key_env": "OPENAI_API_KEY",
							"password":    "db-pass",
						},
						map[string]interface{}{
							"api_key":       "${MODEL_API_KEY}",
							"password":      "$DB_PASSWORD",
							"client_secret": "${CLIENT_SECRET:-literal-fallback}",
						},
					},
				},
			},
		},
		"tokens_per_unit": 100,
		"token_filter":    "keep",
		"auth_token":      "auth-token",
		"Authorization":   "Bearer authorization-token",
		"x-api-key":       "header-token",
	}

	redacted, ok := redactSensitiveConfigValue(input).(map[string]interface{})
	if !ok {
		t.Fatalf("expected map result")
	}

	providers := redacted["providers"].(map[string]interface{})
	models := providers["models"].([]interface{})
	model := models[0].(map[string]interface{})
	refs := model["backend_refs"].([]interface{})
	ref := refs[0].(map[string]interface{})

	if ref["api_key"] != redactedConfigValue {
		t.Fatalf("api_key = %v, want %q", ref["api_key"], redactedConfigValue)
	}
	if ref["password"] != redactedConfigValue {
		t.Fatalf("password = %v, want %q", ref["password"], redactedConfigValue)
	}
	if ref["api_key_env"] != "OPENAI_API_KEY" {
		t.Fatalf("api_key_env should remain visible, got %v", ref["api_key_env"])
	}
	referenceRef := refs[1].(map[string]interface{})
	if referenceRef["api_key"] != "${MODEL_API_KEY}" {
		t.Fatalf("environment api_key reference should remain reusable, got %v", referenceRef["api_key"])
	}
	if referenceRef["password"] != "$DB_PASSWORD" {
		t.Fatalf("environment password reference should remain reusable, got %v", referenceRef["password"])
	}
	if referenceRef["client_secret"] != redactedConfigValue {
		t.Fatalf("environment reference with literal fallback must be redacted, got %v", referenceRef["client_secret"])
	}
	if redacted["tokens_per_unit"] != 100 {
		t.Fatalf("tokens_per_unit should not be redacted")
	}
	if redacted["token_filter"] != "keep" {
		t.Fatalf("token_filter should not be redacted")
	}
	for _, key := range []string{"auth_token", "Authorization", "x-api-key"} {
		if redacted[key] != redactedConfigValue {
			t.Fatalf("%s = %v, want %q", key, redacted[key], redactedConfigValue)
		}
	}
}

func TestConfigGetRedactsSecretsForViewerAndOperator(t *testing.T) {
	const canary = "redact-me-canary-value-0001"
	configPath := writeConfigWithPlaintextSecret(t, canary)

	cases := []struct {
		name       string
		role       string
		wantLeak   bool
		wantRedact bool
	}{
		{name: "viewer", role: "viewer", wantLeak: false, wantRedact: true},
		{name: "operator", role: "operator", wantLeak: false, wantRedact: true},
		{name: "admin", role: "admin", wantLeak: true, wantRedact: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			envName := "VSR_MGMT_TOKEN_" + strings.ToUpper(tc.role)
			token := tc.role + "-token"
			t.Setenv(envName, token)

			server := &ClassificationAPIServer{
				classificationSvc: services.NewPlaceholderClassificationService(),
				configPath:        configPath,
				config: &config.RouterConfig{
					ManagementAPI: config.ManagementAPIConfig{
						Auth: config.ManagementAPIAuthConfig{
							Mode: config.ManagementAuthModeBearer,
							Tokens: []config.ManagementAPITokenRef{
								{Env: envName, Role: tc.role},
							},
							Roles: config.DefaultManagementAPIRoles(),
						},
					},
				},
			}
			mux := server.setupRoutes()

			req := httptest.NewRequest(http.MethodGet, "/api/v1/config", nil)
			req.Header.Set("Authorization", "Bearer "+token)
			rr := httptest.NewRecorder()
			mux.ServeHTTP(rr, req)
			if rr.Code != http.StatusOK {
				t.Fatalf("expected 200, got %d body=%s", rr.Code, rr.Body.String())
			}

			body := rr.Body.String()
			leaked := strings.Contains(body, canary)
			if leaked != tc.wantLeak {
				t.Fatalf("canary leak=%v want=%v body=%s", leaked, tc.wantLeak, body)
			}
			if tc.wantRedact && !strings.Contains(body, redactedConfigValue) {
				t.Fatalf("expected redacted placeholder in body, got %s", body)
			}
		})
	}
}

func TestClassifierInfoRedactsSecretsWithoutSecretView(t *testing.T) {
	const canary = "redact-me-canary-value-0001"
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{Address: "10.0.0.1", Port: 8000, APIKey: canary},
			},
			ModelConfig: map[string]config.ModelParams{
				"m1": {AccessKey: canary},
			},
		},
		ManagementAPI: config.ManagementAPIConfig{
			Auth: config.ManagementAPIAuthConfig{
				Mode: config.ManagementAuthModeBearer,
				Tokens: []config.ManagementAPITokenRef{
					{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
				},
				Roles: config.DefaultManagementAPIRoles(),
			},
		},
	}
	// Plant a json-visible password so redaction (not only json:"-") is required.
	cfg.SemanticCache.Redis = &config.RedisConfig{}
	cfg.SemanticCache.Redis.Connection.Password = canary

	t.Setenv("VSR_MGMT_TOKEN", "viewer-token")

	server := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
	}
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/api/v1/inventory/classifier", nil)
	req.Header.Set("Authorization", "Bearer viewer-token")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rr.Code, rr.Body.String())
	}
	if strings.Contains(rr.Body.String(), canary) {
		t.Fatalf("/api/v1/inventory/classifier leaked credential for viewer: %s", rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), redactedConfigValue) {
		t.Fatalf("expected redacted placeholder for password field, got %s", rr.Body.String())
	}
}

func TestClassifierInfoOmitsParsedSourceDocument(t *testing.T) {
	const canary = "redact-me-canary-value-0001"
	cfg, err := config.ParseYAMLBytesWithoutEnvExpansion([]byte(validateTestYAML("source-doc-card", canary)))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if len(cfg.SourceDocument) == 0 {
		t.Fatal("parsed config must retain SourceDocument for this regression")
	}
	encodedSource := base64.StdEncoding.EncodeToString(cfg.SourceDocument)

	cfg.ManagementAPI = config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	}
	t.Setenv("VSR_MGMT_TOKEN", "viewer-token")

	server := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
	}
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/api/v1/inventory/classifier", nil)
	req.Header.Set("Authorization", "Bearer viewer-token")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rr.Code, rr.Body.String())
	}

	body := rr.Body.String()
	if strings.Contains(body, canary) {
		t.Fatalf("/api/v1/inventory/classifier leaked parsed source secret: %s", body)
	}
	if strings.Contains(body, encodedSource) {
		t.Fatalf("/api/v1/inventory/classifier leaked encoded SourceDocument: %s", body)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("inventory body must remain valid JSON: %v", err)
	}
	configView, _ := parsed["config"].(map[string]interface{})
	if configView == nil {
		t.Fatalf("expected config object, got %s", body)
	}
	for _, key := range []string{"SourceDocument", "sourceDocument", "DocumentHash", "documentHash", "ConfigBaseDir", "configBaseDir"} {
		if _, ok := configView[key]; ok {
			t.Fatalf("inventory config included runtime-only field %q: %s", key, body)
		}
	}
}

func TestAdminCanViewSecretsInClassifierInfo(t *testing.T) {
	const canary = "redact-me-canary-value-0001"
	cfg := &config.RouterConfig{
		ManagementAPI: config.ManagementAPIConfig{
			Auth: config.ManagementAPIAuthConfig{
				Mode: config.ManagementAuthModeBearer,
				Tokens: []config.ManagementAPITokenRef{
					{Env: "VSR_MGMT_TOKEN", Role: "admin"},
				},
				Roles: config.DefaultManagementAPIRoles(),
			},
		},
	}
	cfg.SemanticCache.Redis = &config.RedisConfig{}
	cfg.SemanticCache.Redis.Connection.Password = canary

	t.Setenv("VSR_MGMT_TOKEN", "admin-token")
	server := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
	}
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/api/v1/inventory/classifier", nil)
	req.Header.Set("Authorization", "Bearer admin-token")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), canary) {
		t.Fatalf("admin should see plaintext password, got %s", rr.Body.String())
	}
}

func TestDefaultAdminRoleIncludesSecretView(t *testing.T) {
	roles := config.DefaultManagementAPIRoles()
	admin := managementPrincipal{Role: "admin", AuthEnabled: true}
	if !admin.hasPermission(PermSecretView, roles) {
		t.Fatal("admin wildcard should grant secret_view")
	}
	viewer := managementPrincipal{Role: "viewer", AuthEnabled: true}
	if viewer.hasPermission(PermSecretView, roles) {
		t.Fatal("viewer must not have secret_view")
	}
	operator := managementPrincipal{Role: "operator", AuthEnabled: true}
	if operator.hasPermission(PermSecretView, roles) {
		t.Fatal("operator must not have secret_view")
	}
}

func writeConfigWithPlaintextSecret(t *testing.T, canary string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	// Minimal YAML map — handleConfigGet unmarshals generically, no full schema needed.
	content := "providers:\n  models:\n    - name: demo\n      backend_refs:\n        - api_key: " + canary + "\n          api_key_env: OPENAI_API_KEY\n" +
		"global:\n  stores:\n    vector_store:\n      llama_stack:\n        endpoint: http://llama-stack:8321\n        auth_token: " + canary + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

func TestSecretViewSensitivityRoutesStayOnConfigRead(t *testing.T) {
	for _, route := range apiRoutes() {
		if route.Sensitivity != SensitivitySecretView {
			continue
		}
		if route.Permission != PermConfigRead {
			t.Fatalf("secret_view sensitivity route %s should keep config.read for access; got %s",
				route.pattern(), route.Permission)
		}
	}
}

// Ensure response JSON shape still parses after redaction.
func TestConfigGetRedactedBodyIsJSON(t *testing.T) {
	const canary = "redact-me-canary-value-0001"
	configPath := writeConfigWithPlaintextSecret(t, canary)
	t.Setenv("VSR_MGMT_TOKEN", "viewer-token")

	server := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		configPath:        configPath,
		config: &config.RouterConfig{
			ManagementAPI: config.ManagementAPIConfig{
				Auth: config.ManagementAPIAuthConfig{
					Mode: config.ManagementAuthModeBearer,
					Tokens: []config.ManagementAPITokenRef{
						{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
					},
					Roles: config.DefaultManagementAPIRoles(),
				},
			},
		},
	}
	mux := server.setupRoutes()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/config", nil)
	req.Header.Set("Authorization", "Bearer viewer-token")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	var parsed map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("redacted body must remain valid JSON: %v", err)
	}
}
