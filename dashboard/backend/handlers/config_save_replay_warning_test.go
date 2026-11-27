package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// A save that starts Router Replay's memory store empty says so: a changed
// ttl_seconds, or a store that no longer keeps the records in memory. Any
// other save keeps the store and warns of nothing.
func TestASaveThatEmptiesTheMemoryReplayStoreWarns(t *testing.T) {
	applyWithoutRuntime(t)
	config := canonicalConfigBody("127.0.0.1:8000")
	config["global"] = map[string]any{"services": map[string]any{
		"router_replay": map[string]any{"store_backend": "memory", "ttl_seconds": 3600},
	}}
	previous, err := yaml.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	for _, save := range []struct {
		name, patch, field string
	}{
		{"ttl_seconds", `{"services":{"router_replay":{"ttl_seconds":600}}}`, "global.services.router_replay.ttl_seconds"},
		{"store_backend", `{"services":{"router_replay":{"store_backend":"file"}}}`, "global.services.router_replay.store_backend"},
		{"another setting", `{"router":{"strategy":"priority"}}`, ""},
	} {
		t.Run(save.name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "config.yaml")
			if err := os.WriteFile(path, previous, 0o600); err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			UpdateRouterDefaultsHandler(path, false, root)(response,
				httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewBufferString(save.patch)))
			var fields []string
			for _, warning := range decodeSavedConfig(t, response).Warnings {
				if warning.Code == replayStoreResetCode {
					fields = append(fields, warning.Field)
				}
			}
			if (save.field == "" && len(fields) > 0) || (save.field != "" && (len(fields) != 1 || fields[0] != save.field)) {
				t.Fatalf("replay store warnings on fields %v, want %q", fields, save.field)
			}
		})
	}
}

// A decision with a Router Replay plugin of its own keeps its records in a
// memory store named after it, so renaming it loses them.
func TestRenamingADecisionWithItsOwnReplayStoreWarns(t *testing.T) {
	applyWithoutRuntime(t)
	withDecision := func(name string) map[string]any {
		config := canonicalConfigBody("127.0.0.1:8000")
		config["global"] = map[string]any{"services": map[string]any{
			"router_replay": map[string]any{"store_backend": "memory"},
		}}
		decision := config["recipes"].([]map[string]interface{})[0]["decisions"].([]map[string]interface{})[0]
		decision["name"] = name
		decision["plugins"] = []map[string]any{{"type": "router_replay", "configuration": map[string]any{"enabled": true, "max_records": 50}}}
		return config
	}
	previous, err := yaml.Marshal(withDecision("business"))
	if err != nil {
		t.Fatal(err)
	}
	for _, save := range []struct {
		name, decision string
		warns          bool
	}{{"renamed", "business-renamed", true}, {"kept", "business", false}} {
		t.Run(save.name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "config.yaml")
			if err := os.WriteFile(path, previous, 0o600); err != nil {
				t.Fatal(err)
			}
			body, err := json.Marshal(withDecision(save.decision))
			if err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			UpdateConfigHandler(path, false, root)(response,
				httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(body)))
			var messages []string
			for _, warning := range decodeSavedConfig(t, response).Warnings {
				if warning.Code == replayStoreResetCode {
					messages = append(messages, warning.Message)
				}
			}
			if warned := len(messages) == 1 && strings.Contains(messages[0], "records of business in"); warned != save.warns || (!save.warns && len(messages) > 0) {
				t.Fatalf("replay store warnings = %q, want a warning naming the old decision: %v", messages, save.warns)
			}
		})
	}
}
