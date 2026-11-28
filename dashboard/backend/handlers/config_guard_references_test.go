package handlers

import (
	"net/http"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func guardReferenceBody(address, password, signing string, tls bool) map[string]interface{} {
	body := backendBody("127.0.0.1:8000", nil)
	body["global"] = map[string]interface{}{"stores": map[string]interface{}{
		"guard_verdicts": map[string]interface{}{
			"backend": "redis", "hmac_key_env": signing,
			"redis": map[string]interface{}{"address": address, "password_env": password, "tls": tls},
		},
	}}
	return body
}

func TestGuardStoreReferencesRequireSecretPermission(t *testing.T) {
	path := createValidTestConfig(t, t.TempDir())
	saved := guardReferenceBody("redis.example:6379", "GUARD_PASSWORD", "GUARD_HMAC", true)
	if response := postConfig(t, path, saved); response.Code != http.StatusForbidden {
		t.Fatalf("writer introduced references: %d %s", response.Code, response.Body.String())
	}
	if response := postConfig(t, path, saved, auth.PermConfigSecretsRead); response.Code != http.StatusOK {
		t.Fatalf("admin save: %d %s", response.Code, response.Body.String())
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]map[string]interface{}{
		"redirect":      guardReferenceBody("attacker.example:6379", "GUARD_PASSWORD", "GUARD_HMAC", true),
		"password":      guardReferenceBody("redis.example:6379", "OPENAI_API_KEY", "GUARD_HMAC", true),
		"signing key":   guardReferenceBody("redis.example:6379", "GUARD_PASSWORD", "OTHER_HMAC", true),
		"TLS downgrade": guardReferenceBody("redis.example:6379", "GUARD_PASSWORD", "GUARD_HMAC", false),
	} {
		t.Run(name, func(t *testing.T) {
			if response := postConfig(t, path, body); response.Code != http.StatusForbidden {
				t.Fatalf("writer changed secret destination: %d %s", response.Code, response.Body.String())
			}
			requireUnchangedFile(t, path, before)
		})
	}
	if response := postConfig(t, path, saved); response.Code != http.StatusOK {
		t.Fatalf("writer preserved references: %d %s", response.Code, response.Body.String())
	}
}

func TestNamedSecretFieldsCannotBeRedirected(t *testing.T) {
	for _, field := range []string{"api_key_env", "key_env", "hmac_key_env", "password_env"} {
		saved := []byte("store: {address: redis.example:6379, " + field + ": TEST_KEY}\n")
		moved := []byte("store: {address: attacker.example:6379, " + field + ": TEST_KEY}\n")
		if !keyReferencesKept(saved, saved) || keyReferencesKept(saved, moved) {
			t.Fatalf("reference destination not protected for %s", field)
		}
	}
}

func TestGuardStoreDefaultTLSDoesNotRequireSecretPermission(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		t.Run(map[bool]string{false: "omitted_to_false", true: "false_to_omitted"}[explicit], func(t *testing.T) {
			path := createValidTestConfig(t, t.TempDir())
			body := guardReferenceBody("redis.example:6379", "GUARD_PASSWORD", "GUARD_HMAC", false)
			redis := body["global"].(map[string]interface{})["stores"].(map[string]interface{})["guard_verdicts"].(map[string]interface{})["redis"].(map[string]interface{})
			if !explicit {
				delete(redis, "tls")
			}
			if response := postConfig(t, path, body, auth.PermConfigSecretsRead); response.Code != http.StatusOK {
				t.Fatalf("admin save: %d %s", response.Code, response.Body.String())
			}
			if explicit {
				delete(redis, "tls")
			} else {
				redis["tls"] = false
			}
			if response := postConfig(t, path, body); response.Code != http.StatusOK {
				t.Fatalf("equivalent TLS default rejected: %d %s", response.Code, response.Body.String())
			}
		})
	}
}
