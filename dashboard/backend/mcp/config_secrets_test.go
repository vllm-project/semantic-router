package mcp

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

const (
	argumentTokenCanary      = "argument-token-canary"
	argumentPasswordCanary   = "argument-password-canary"
	argumentAPIKeyCanary     = "argument-api-key-canary"
	argumentHeaderCanary     = "argument-header-canary"
	argumentEnvCanary        = "argument-env-canary"
	argumentUnknownCanary    = "argument-unknown-canary"
	environmentCanary        = "environment-secret-canary"
	headerCanary             = "header-secret-canary"
	oauthCanary              = "oauth-secret-canary"
	urlUserCanary            = "url-user-canary"
	urlPasswordCanary        = "url-password-canary"
	urlPathCanary            = "url-path-capability-canary"
	urlQueryTokenCanary      = "url-query-token-canary"   //nolint:gosec // Deliberate non-secret test canary.
	urlQueryAPIKeyCanary     = "url-query-api-key-canary" //nolint:gosec // Deliberate non-secret test canary.
	urlQuerySignatureCanary  = "url-query-signature-canary"
	urlQueryCredentialCanary = "url-query-credential-canary" //nolint:gosec // Deliberate non-secret test canary.
	urlQueryAuthCanary       = "url-query-auth-canary"
	urlQueryCodeCanary       = "url-query-code-canary"
	urlQueryKeyCanary        = "url-query-key-canary"
	urlFragmentCanary        = "url-fragment-token-canary"
	urlOpaqueFragmentCanary  = "url-opaque-fragment-canary"
	oauthTokenURLCanary      = "oauth-token-url-canary" //nolint:gosec // Deliberate non-secret test canary.
	malformedURLCanary       = "malformed-url-canary"
	schemelessURLCanary      = "schemeless-url-canary"
)

var secretCanaries = []string{
	argumentTokenCanary,
	argumentPasswordCanary,
	argumentAPIKeyCanary,
	argumentHeaderCanary,
	argumentEnvCanary,
	argumentUnknownCanary,
	environmentCanary,
	headerCanary,
	oauthCanary,
	urlUserCanary,
	urlPasswordCanary,
	urlPathCanary,
	urlQueryTokenCanary,
	urlQueryAPIKeyCanary,
	urlQuerySignatureCanary,
	urlQueryCredentialCanary,
	urlQueryAuthCanary,
	urlQueryCodeCanary,
	urlQueryKeyCanary,
	urlFragmentCanary,
	urlOpaqueFragmentCanary,
	oauthTokenURLCanary,
}

func serverConfigWithSecrets() *ServerConfig {
	return &ServerConfig{
		ID:          "secret-server",
		Name:        "Secret server",
		Description: "Exercises API redaction",
		Transport:   TransportStdio,
		Connection: ConnectionConfig{
			Command: "secret-server",
			Args: []string{
				"--token", argumentTokenCanary,
				"--password=" + argumentPasswordCanary,
				"--API_KEY", argumentAPIKeyCanary,
				"--header", "Authorization: Bearer " + argumentHeaderCanary,
				"-e", "GITHUB_TOKEN=" + argumentEnvCanary,
				"--github-token", argumentUnknownCanary,
				"--safe=value",
			},
			Env: map[string]string{
				"API_TOKEN": environmentCanary,
				"DELETE_ME": "environment-delete-canary",
			},
			URL: "https://" + urlUserCanary + ":" + urlPasswordCanary + "@mcp.example.test/rpc/" + urlPathCanary +
				"?token=" + urlQueryTokenCanary +
				"&apiKey=" + urlQueryAPIKeyCanary +
				"&X-Amz-Signature=" + urlQuerySignatureCanary +
				"&credential=" + urlQueryCredentialCanary +
				"&auth=" + urlQueryAuthCanary +
				"&code=" + urlQueryCodeCanary +
				"&view=ordinary&" + urlQueryKeyCanary +
				"#access_token=" + urlFragmentCanary + "&opaque=" + urlOpaqueFragmentCanary,
			Headers: map[string]string{
				"Authorization": headerCanary,
				"X-Delete":      "header-delete-canary",
				"X-Rotate":      "header-rotate-canary",
			},
		},
		Enabled: true,
		Security: &SecurityConfig{OAuth: &OAuthConfig{
			ClientID:         "client-id",
			ClientSecret:     oauthCanary,
			AuthorizationURL: "https://auth.example.test/authorize",
			TokenURL:         "https://auth.example.test/token?apiKey=" + oauthTokenURLCanary + "&view=ordinary",
		}},
	}
}

func assertNoSecretCanary(t *testing.T, value []byte) {
	t.Helper()
	for _, canary := range append(secretCanaries, "environment-delete-canary", "header-delete-canary", "header-rotate-canary") {
		if strings.Contains(string(value), canary) {
			t.Fatalf("redacted value leaked %q: %s", canary, value)
		}
	}
}

func TestRedactedServerStateHidesSecretsWithoutMutatingRuntimeConfig(t *testing.T) {
	t.Parallel()
	config := serverConfigWithSecrets()
	state := &ServerState{
		Config: config,
		Status: StatusError,
		Error:  "upstream returned " + oauthCanary,
	}

	redacted := RedactedServerState(state)
	encoded, err := json.Marshal(redacted)
	if err != nil {
		t.Fatal(err)
	}
	assertNoSecretCanary(t, encoded)
	if !strings.Contains(string(encoded), RedactedValue) {
		t.Fatalf("redacted response does not contain placeholder: %s", encoded)
	}
	if !strings.Contains(redacted.Config.Connection.URL, "?"+RedactedValue) ||
		strings.Contains(redacted.Config.Connection.URL, "ordinary") {
		t.Fatalf("redaction did not hide the complete URL query: %q", redacted.Config.Connection.URL)
	}
	if !strings.Contains(redacted.Config.Security.OAuth.TokenURL, "?"+RedactedValue) ||
		strings.Contains(redacted.Config.Security.OAuth.TokenURL, "ordinary") {
		t.Fatalf("redaction did not hide the complete OAuth URL query: %q", redacted.Config.Security.OAuth.TokenURL)
	}
	if redacted.Error != "Connection unavailable" {
		t.Fatalf("redacted error = %q", redacted.Error)
	}
	if len(redacted.Config.Connection.Args) != 1 || redacted.Config.Connection.Args[0] != RedactedValue {
		t.Fatalf("redacted arguments did not use one opaque marker: %#v", redacted.Config.Connection.Args)
	}
	if config.Connection.Env["API_TOKEN"] != environmentCanary ||
		config.Connection.Headers["Authorization"] != headerCanary ||
		config.Security.OAuth.ClientSecret != oauthCanary ||
		!strings.Contains(config.Connection.URL, urlPasswordCanary) {
		t.Fatalf("redaction mutated manager-owned config: %#v", config)
	}
}

func TestManagerUpdateMergesRedactedSecretsAndSupportsDeleteReplace(t *testing.T) {
	t.Parallel()
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	original := serverConfigWithSecrets()
	if addErr := manager.AddServer(original); addErr != nil {
		t.Fatal(addErr)
	}

	masked := RedactedServerConfig(original)
	masked.Name = "Updated server"
	masked.Connection.Env = nil // omitted fields preserve all stored values
	delete(masked.Connection.Headers, "X-Delete")
	masked.Connection.Headers["X-Rotate"] = "rotated-header-secret"
	if updateErr := manager.UpdateServer(masked); updateErr != nil {
		t.Fatal(updateErr)
	}

	stored, ok := manager.GetServer(original.ID)
	if !ok {
		t.Fatal("updated server missing")
	}
	if stored.Connection.Env["API_TOKEN"] != environmentCanary ||
		stored.Connection.Env["DELETE_ME"] != "environment-delete-canary" {
		t.Fatalf("omitted environment did not preserve stored values: %#v", stored.Connection.Env)
	}
	if stored.Connection.Headers["Authorization"] != headerCanary {
		t.Fatalf("masked header was not preserved: %#v", stored.Connection.Headers)
	}
	if _, exists := stored.Connection.Headers["X-Delete"]; exists {
		t.Fatalf("omitted header key was not deleted: %#v", stored.Connection.Headers)
	}
	if stored.Connection.Headers["X-Rotate"] != "rotated-header-secret" {
		t.Fatalf("header replacement was not stored: %#v", stored.Connection.Headers)
	}
	if stored.Connection.URL != original.Connection.URL ||
		stored.Security.OAuth.ClientSecret != oauthCanary ||
		stored.Security.OAuth.TokenURL != original.Security.OAuth.TokenURL {
		t.Fatalf("URL or OAuth placeholder did not preserve stored values: %#v", stored)
	}
	if strings.Join(stored.Connection.Args, "\x00") != strings.Join(original.Connection.Args, "\x00") {
		t.Fatalf("credential arguments were not preserved: %#v", stored.Connection.Args)
	}

	runtimeClient, err := NewClient(stored)
	if err != nil {
		t.Fatal(err)
	}
	runtimeConfig := runtimeClient.GetConfig()
	if runtimeConfig.Connection.Headers["Authorization"] != headerCanary ||
		runtimeConfig.Connection.Env["API_TOKEN"] != environmentCanary ||
		runtimeConfig.Security.OAuth.ClientSecret != oauthCanary {
		t.Fatalf("runtime client did not receive original secrets: %#v", runtimeConfig)
	}

	replacement := RedactedServerConfig(stored)
	replacement.Connection.Headers = map[string]string{} // explicit empty map clears all keys
	replacement.Connection.Env = map[string]string{"API_TOKEN": "new-environment-secret"}
	replacement.Connection.Args = []string{"--token", "new-token-secret", "--api-key=new-api-key-secret"}
	replacement.Connection.URL = "https://new-user:new-password@mcp.example.test/rpc"
	replacement.Security.OAuth.ClientSecret = "new-oauth-secret"
	replacement.Security.OAuth.AuthorizationURL = "https://auth.example.test/authorize"
	replacement.Security.OAuth.TokenURL = "https://auth.example.test/token"
	if updateErr := manager.UpdateServer(replacement); updateErr != nil {
		t.Fatal(updateErr)
	}

	replaced, ok := manager.GetServer(original.ID)
	if !ok {
		t.Fatal("replaced server missing")
	}
	if len(replaced.Connection.Headers) != 0 {
		t.Fatalf("explicit empty headers did not clear the map: %#v", replaced.Connection.Headers)
	}
	if len(replaced.Connection.Env) != 1 || replaced.Connection.Env["API_TOKEN"] != "new-environment-secret" {
		t.Fatalf("environment replacement = %#v", replaced.Connection.Env)
	}
	if got := replaced.Connection.Args; len(got) != 3 || got[1] != "new-token-secret" || got[2] != "--api-key=new-api-key-secret" {
		t.Fatalf("argument replacement = %#v", got)
	}
	if replaced.Connection.URL != replacement.Connection.URL || replaced.Security.OAuth.ClientSecret != "new-oauth-secret" {
		t.Fatalf("explicit URL or OAuth replacement was not stored: %#v", replaced)
	}
}

func TestManagerRejectsSecretReuseAcrossEndpointChanges(t *testing.T) {
	t.Parallel()

	httpCases := []struct {
		name   string
		mutate func(*ServerConfig, *ServerConfig)
	}{
		{name: "header placeholder", mutate: func(masked, original *ServerConfig) {
			masked.Connection.Headers = RedactedServerConfig(original).Connection.Headers
		}},
		{name: "omitted headers", mutate: func(masked, _ *ServerConfig) {
			masked.Connection.Headers = nil
		}},
		{name: "OAuth placeholder", mutate: func(masked, original *ServerConfig) {
			masked.Security = RedactedServerConfig(original).Security
		}},
		{name: "omitted OAuth", mutate: func(masked, _ *ServerConfig) {
			masked.Security = nil
		}},
		{name: "URL placeholder", mutate: func(masked, _ *ServerConfig) {
			masked.Connection.URL = "https://attacker.example.test/" + RedactedValue
		}},
	}
	for _, test := range httpCases {
		t.Run("HTTP "+test.name, func(t *testing.T) {
			manager, original := managerWithSecretServer(t, TransportStreamableHTTP)
			masked := safeHTTPReplacement(original)
			test.mutate(masked, original)
			assertEndpointSecretReuseRejected(t, manager, original, masked)
		})
	}

	stdioCases := []struct {
		name   string
		mutate func(*ServerConfig)
	}{
		{name: "omitted arguments", mutate: func(masked *ServerConfig) { masked.Connection.Args = nil }},
		{name: "argument placeholder", mutate: func(masked *ServerConfig) {
			masked.Connection.Args = []string{RedactedValue}
		}},
		{name: "omitted environment", mutate: func(masked *ServerConfig) { masked.Connection.Env = nil }},
		{name: "environment placeholder", mutate: func(masked *ServerConfig) {
			masked.Connection.Env = map[string]string{"API_TOKEN": RedactedValue}
		}},
	}
	for _, test := range stdioCases {
		t.Run("stdio "+test.name, func(t *testing.T) {
			manager, original := managerWithSecretServer(t, TransportStdio)
			masked := safeStdioReplacement(original)
			test.mutate(masked)
			assertEndpointSecretReuseRejected(t, manager, original, masked)
		})
	}

	t.Run("explicit replacements may change endpoints", func(t *testing.T) {
		manager, original := managerWithSecretServer(t, TransportStreamableHTTP)
		if updateErr := manager.UpdateServer(safeHTTPReplacement(original)); updateErr != nil {
			t.Fatalf("explicit HTTP replacement failed: %v", updateErr)
		}
		manager, original = managerWithSecretServer(t, TransportStdio)
		if updateErr := manager.UpdateServer(safeStdioReplacement(original)); updateErr != nil {
			t.Fatalf("explicit stdio replacement failed: %v", updateErr)
		}
	})
}

func TestManagerTreatsExplicitStdioArgumentChangesAsEndpointChanges(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name string
		env  map[string]string
	}{
		{name: "omitted environment", env: nil},
		{name: "environment placeholder", env: map[string]string{"API_TOKEN": RedactedValue}},
	} {
		t.Run(test.name, func(t *testing.T) {
			manager, original := managerWithSecretServer(t, TransportStdio)
			updated := explicitStdioArgumentReplacement(original, test.env)
			assertEndpointSecretReuseRejected(t, manager, original, updated)
		})
	}

	for _, test := range []struct {
		name string
		env  map[string]string
		want map[string]string
	}{
		{name: "explicit environment clear", env: map[string]string{}, want: map[string]string{}},
		{
			name: "explicit environment replacement",
			env:  map[string]string{"API_TOKEN": "replacement-environment-secret"},
			want: map[string]string{"API_TOKEN": "replacement-environment-secret"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			manager, original := managerWithSecretServer(t, TransportStdio)
			updated := explicitStdioArgumentReplacement(original, test.env)

			resolved, err := manager.resolveConnectionTestConfig(updated)
			if err != nil {
				t.Fatalf("resolve connection test config: %v", err)
			}
			if !equalStringMap(resolved.Connection.Env, test.want) {
				t.Fatalf("resolved environment = %#v, want %#v", resolved.Connection.Env, test.want)
			}
			if strings.Join(resolved.Connection.Args, "\x00") != strings.Join(updated.Connection.Args, "\x00") {
				t.Fatalf("resolved arguments = %#v, want %#v", resolved.Connection.Args, updated.Connection.Args)
			}

			if err := manager.UpdateServer(updated); err != nil {
				t.Fatalf("update server: %v", err)
			}
			stored, ok := manager.GetServer(original.ID)
			if !ok {
				t.Fatal("updated server missing")
			}
			if !equalStringMap(stored.Connection.Env, test.want) {
				t.Fatalf("stored environment = %#v, want %#v", stored.Connection.Env, test.want)
			}
			if strings.Join(stored.Connection.Args, "\x00") != strings.Join(updated.Connection.Args, "\x00") {
				t.Fatalf("stored arguments = %#v, want %#v", stored.Connection.Args, updated.Connection.Args)
			}
		})
	}
}

func TestManagerPreservesStdioArgumentsForOmittedOrOpaqueUpdates(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name string
		args []string
	}{
		{name: "omitted arguments", args: nil},
		{name: "opaque argument marker", args: []string{RedactedValue}},
	} {
		t.Run(test.name, func(t *testing.T) {
			manager, original := managerWithSecretServer(t, TransportStdio)
			updated := RedactedServerConfig(original)
			updated.Connection.Args = test.args

			resolved, err := manager.resolveConnectionTestConfig(updated)
			if err != nil {
				t.Fatalf("resolve connection test config: %v", err)
			}
			if strings.Join(resolved.Connection.Args, "\x00") != strings.Join(original.Connection.Args, "\x00") {
				t.Fatalf("resolved arguments = %#v, want %#v", resolved.Connection.Args, original.Connection.Args)
			}

			if err := manager.UpdateServer(updated); err != nil {
				t.Fatalf("update server: %v", err)
			}
			stored, ok := manager.GetServer(original.ID)
			if !ok {
				t.Fatal("updated server missing")
			}
			if strings.Join(stored.Connection.Args, "\x00") != strings.Join(original.Connection.Args, "\x00") {
				t.Fatalf("stored arguments = %#v, want %#v", stored.Connection.Args, original.Connection.Args)
			}
		})
	}
}

func explicitStdioArgumentReplacement(original *ServerConfig, environment map[string]string) *ServerConfig {
	updated := safeStdioReplacement(original)
	updated.Connection.Command = original.Connection.Command
	updated.Connection.Cwd = original.Connection.Cwd
	updated.Connection.Args = []string{"-y", "untrusted-mcp-server"}
	updated.Connection.Env = environment
	return updated
}

func equalStringMap(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		if right[key] != value {
			return false
		}
	}
	return true
}

func TestManagerBindsRedactedOAuthSecretToClientAndTokenEndpoint(t *testing.T) {
	t.Parallel()
	manager, original := managerWithSecretServer(t, TransportStreamableHTTP)
	masked := RedactedServerConfig(original)
	masked.Security.OAuth.TokenURL = "https://attacker.example.test/token"
	if updateErr := manager.UpdateServer(masked); updateErr == nil {
		t.Fatal("OAuth client secret was reused for a different token endpoint")
	}
	if _, resolveErr := manager.resolveConnectionTestConfig(masked); resolveErr == nil {
		t.Fatal("connection test reused OAuth client secret for a different token endpoint")
	}

	masked = RedactedServerConfig(original)
	masked.Security.OAuth.ClientID = "different-client"
	if updateErr := manager.UpdateServer(masked); updateErr == nil {
		t.Fatal("OAuth client secret was reused for a different client")
	}
}

func managerWithSecretServer(t *testing.T, transport TransportType) (*Manager, *ServerConfig) {
	t.Helper()
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	original := serverConfigWithSecrets()
	original.Transport = transport
	if transport == TransportStreamableHTTP {
		original.Connection.Command = ""
		original.Connection.Args = nil
		original.Connection.Env = nil
	}
	if addErr := manager.AddServer(original); addErr != nil {
		t.Fatal(addErr)
	}
	return manager, original
}

func safeHTTPReplacement(original *ServerConfig) *ServerConfig {
	masked := RedactedServerConfig(original)
	masked.Connection.URL = "https://replacement.example.test/mcp"
	masked.Connection.Headers = map[string]string{}
	masked.Security = &SecurityConfig{}
	return masked
}

func safeStdioReplacement(original *ServerConfig) *ServerConfig {
	masked := RedactedServerConfig(original)
	masked.Connection.Command = "replacement-command"
	masked.Connection.Args = []string{}
	masked.Connection.Env = map[string]string{}
	masked.Connection.Headers = map[string]string{}
	masked.Connection.URL = ""
	masked.Security = &SecurityConfig{}
	return masked
}

func assertEndpointSecretReuseRejected(
	t *testing.T,
	manager *Manager,
	original, updated *ServerConfig,
) {
	t.Helper()
	if updateErr := manager.UpdateServer(updated); updateErr == nil {
		t.Fatal("endpoint change reused stored credentials")
	}
	if _, resolveErr := manager.resolveConnectionTestConfig(updated); resolveErr == nil {
		t.Fatal("connection test reused stored credentials")
	}
	stored, ok := manager.GetServer(original.ID)
	if !ok || stored.Connection.URL != original.Connection.URL || stored.Connection.Command != original.Connection.Command {
		t.Fatalf("rejected endpoint change mutated the stored config: %#v", stored)
	}
	if strings.Join(stored.Connection.Args, "\x00") != strings.Join(original.Connection.Args, "\x00") ||
		!equalStringMap(stored.Connection.Env, original.Connection.Env) {
		t.Fatalf("rejected endpoint change mutated stored arguments or environment: %#v", stored.Connection)
	}
}

func TestRedactedURLCredentialsFailsClosedForMalformedAndSchemelessUserInfo(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		url  string
	}{
		{name: "malformed", url: "https://%zz.example.test/rpc?token=" + malformedURLCanary},
		{name: "schemeless userinfo", url: schemelessURLCanary + "@mcp.example.test/v1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := redactURLCredentials(test.url); got != RedactedValue {
				t.Fatalf("redacted URL = %q, want placeholder", got)
			}
			merged, err := mergeRedactedURL(test.url, RedactedValue)
			if err != nil {
				t.Fatal(err)
			}
			if merged != test.url {
				t.Fatalf("merged URL = %q, want original %q", merged, test.url)
			}
		})
	}
}

func TestManagerUpdateRejectsUnmatchedRedactedValue(t *testing.T) {
	t.Parallel()
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := serverConfigWithSecrets()
	if addErr := manager.AddServer(config); addErr != nil {
		t.Fatal(addErr)
	}

	updated := RedactedServerConfig(config)
	updated.Connection.Headers["X-New"] = RedactedValue
	if updateErr := manager.UpdateServer(updated); updateErr == nil {
		t.Fatal("expected unmatched redacted header to be rejected")
	}
	stored, _ := manager.GetServer(config.ID)
	if _, exists := stored.Connection.Headers["X-New"]; exists {
		t.Fatalf("rejected update changed runtime config: %#v", stored.Connection.Headers)
	}
}

func TestManagerRejectsOpaqueArgumentMarkerMixedWithNewArguments(t *testing.T) {
	t.Parallel()
	cases := [][]string{
		{RedactedValue, "--callback", "https://attacker.example.test"},
		{"--token", RedactedValue, "--callback", "https://attacker.example.test"},
		{"--token=" + RedactedValue},
	}
	for index, arguments := range cases {
		t.Run(fmt.Sprintf("case-%d", index), func(t *testing.T) {
			manager, original := managerWithSecretServer(t, TransportStdio)
			updated := RedactedServerConfig(original)
			updated.Connection.Args = arguments
			if updateErr := manager.UpdateServer(updated); updateErr == nil {
				t.Fatal("mixed argument marker was accepted by update")
			}
			if _, resolveErr := manager.resolveConnectionTestConfig(updated); resolveErr == nil {
				t.Fatal("mixed argument marker was accepted by connection test")
			}
			stored, _ := manager.GetServer(original.ID)
			if strings.Join(stored.Connection.Args, "\x00") != strings.Join(original.Connection.Args, "\x00") {
				t.Fatalf("rejected marker injection changed stored arguments: %#v", stored.Connection.Args)
			}
		})
	}
}

func TestConnectionTestConfigResolvesStoredSecretsWithoutPersisting(t *testing.T) {
	t.Parallel()
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := serverConfigWithSecrets()
	if addErr := manager.AddServer(config); addErr != nil {
		t.Fatal(addErr)
	}

	masked := RedactedServerConfig(config)
	masked.Name = "Temporary test name"
	resolved, resolveErr := manager.resolveConnectionTestConfig(masked)
	if resolveErr != nil {
		t.Fatal(resolveErr)
	}
	if resolved.Connection.Env["API_TOKEN"] != environmentCanary ||
		resolved.Connection.Headers["Authorization"] != headerCanary ||
		resolved.Security.OAuth.ClientSecret != oauthCanary ||
		resolved.Connection.Args[1] != argumentTokenCanary ||
		!strings.Contains(resolved.Connection.URL, urlPasswordCanary) {
		t.Fatalf("connection test config did not resolve stored secrets: %#v", resolved)
	}

	stored, _ := manager.GetServer(config.ID)
	if stored.Name != config.Name {
		t.Fatalf("connection test mutated persisted config name: %q", stored.Name)
	}
	unknown := RedactedServerConfig(config)
	unknown.ID = "unknown-server"
	if _, resolveErr := manager.resolveConnectionTestConfig(unknown); resolveErr == nil {
		t.Fatal("expected an unknown server's redacted placeholders to be rejected")
	}
}
