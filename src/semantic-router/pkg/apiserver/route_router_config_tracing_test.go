//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestConfigMutationRejectsTracingChangesWithoutPersistence(t *testing.T) {
	for _, method := range []string{http.MethodPost, http.MethodPut, http.MethodPatch} {
		t.Run(method, func(t *testing.T) {
			configPath := writeDeployTestBaseConfig(t)
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			candidate, err := config.ParseYAMLBytes(before)
			if err != nil {
				t.Fatal(err)
			}
			candidate.Observability.Tracing.Enabled = true
			candidate.Observability.Tracing.Exporter.Endpoint = "replacement-collector:4317"
			payload := map[string]string{"yaml": string(mustMarshalCanonicalConfigYAML(t, candidate))}
			path := apiConfigPath
			if method == http.MethodPost {
				payload["mode"] = string(routerConfigMutationReplace)
				path = apiConfigPlanPath
			}
			body, err := json.Marshal(payload)
			if err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(method, path, bytes.NewReader(body))
			request.Header.Set("If-Match", configDocumentETag(before))
			response := httptest.NewRecorder()
			server := &ClassificationAPIServer{configPath: configPath}
			switch method {
			case http.MethodPost:
				server.handleConfigPlan(response, request)
			case http.MethodPut:
				server.handleConfigPut(response, request)
			case http.MethodPatch:
				server.handleConfigPatch(response, request)
			}
			if response.Code != http.StatusConflict ||
				!strings.Contains(response.Body.String(), "RESTART_REQUIRED") ||
				!strings.Contains(response.Body.String(), "tracing configuration changed") {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
			after, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) {
				t.Fatal("restart-required update changed the source config and its ETag")
			}
			entries, err := os.ReadDir(filepath.Dir(configPath))
			if err != nil || len(entries) != 1 {
				t.Fatalf("restart-required update created persistent state: entries=%v, error=%v", entries, err)
			}
		})
	}
}

func TestTracingReloadCompatibility(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*config.TracingConfig)
	}{
		{"enabled", func(c *config.TracingConfig) { c.Enabled = !c.Enabled }},
		{"provider", func(c *config.TracingConfig) { c.Provider = "other" }},
		{"exporter type", func(c *config.TracingConfig) { c.Exporter.Type = "stdout" }},
		{"endpoint", func(c *config.TracingConfig) { c.Exporter.Endpoint = "replacement:4317" }},
		{"transport", func(c *config.TracingConfig) { c.Exporter.Insecure = !c.Exporter.Insecure }},
		{"sampling type", func(c *config.TracingConfig) { c.Sampling.Type = "always_off" }},
		{"sampling rate", func(c *config.TracingConfig) { c.Sampling.Rate = 0.5 }},
		{"service name", func(c *config.TracingConfig) { c.Resource.ServiceName = "replacement" }},
		{"service version", func(c *config.TracingConfig) { c.Resource.ServiceVersion = "v2" }},
		{"environment", func(c *config.TracingConfig) { c.Resource.DeploymentEnvironment = "production" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			current := minimalDeployTestConfig("old_route")
			next := minimalDeployTestConfig("new_route")
			if err := validateParsedHotReloadCompatibility(current, next); err != nil {
				t.Fatalf("routing change with unchanged tracing rejected: %v", err)
			}
			tc.mutate(&next.Observability.Tracing)
			if err := validateParsedHotReloadCompatibility(current, next); err == nil ||
				!strings.Contains(err.Error(), "tracing configuration changed") {
				t.Fatalf("tracing change did not require restart: %v", err)
			}
		})
	}
}
