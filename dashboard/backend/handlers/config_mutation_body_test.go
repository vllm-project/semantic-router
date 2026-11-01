package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

const mutationDeployYAML = `routing:
  decisions:
    - name: deployed-default
      description: Deployed route
      priority: 5
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`

type configMutationCase struct {
	name       string
	path       string
	body       []byte
	parsedCode int
	handler    func(configPath, configDir string) http.HandlerFunc
}

func configMutationCases(t *testing.T) []configMutationCase {
	t.Helper()
	updateBody, err := json.Marshal(canonicalConfigBody("127.0.0.1:8000"))
	if err != nil {
		t.Fatalf("marshal config update: %v", err)
	}
	deployBody, err := json.Marshal(DeployRequest{YAML: mutationDeployYAML})
	if err != nil {
		t.Fatalf("marshal deploy request: %v", err)
	}
	return []configMutationCase{
		{
			name: "full config update", path: "/api/router/config/update",
			body: updateBody, parsedCode: http.StatusOK,
			handler: func(configPath, configDir string) http.HandlerFunc {
				return UpdateConfigHandler(configPath, false, configDir)
			},
		},
		{
			name: "router defaults", path: "/api/router/config/global/update",
			body: []byte(`{}`), parsedCode: http.StatusOK,
			handler: func(configPath, configDir string) http.HandlerFunc {
				return UpdateRouterDefaultsHandler(configPath, false, configDir)
			},
		},
		{
			name: "raw global YAML", path: "/api/router/config/global/raw/update",
			body: []byte("router: {}\n"), parsedCode: http.StatusOK,
			handler: func(configPath, configDir string) http.HandlerFunc {
				return UpdateGlobalConfigYAMLHandler(configPath, false, configDir)
			},
		},
		{
			name: "deploy preview", path: "/api/router/config/deploy/preview",
			body: deployBody, parsedCode: http.StatusOK,
			handler: func(configPath, _ string) http.HandlerFunc {
				return DeployPreviewHandler(configPath, false)
			},
		},
		{
			name: "deploy", path: "/api/router/config/deploy",
			body: deployBody, parsedCode: http.StatusOK,
			handler: func(configPath, configDir string) http.HandlerFunc {
				return DeployHandler(configPath, false, configDir)
			},
		},
		{
			name: "rollback", path: "/api/router/config/rollback",
			body: []byte(`{"version":"does-not-matter"}`), parsedCode: http.StatusNotFound,
			handler: func(configPath, configDir string) http.HandlerFunc {
				return RollbackHandler(configPath, false, configDir)
			},
		},
	}
}

func deployFamilyCases(t *testing.T) []configMutationCase {
	t.Helper()
	var family []configMutationCase
	for _, mutation := range configMutationCases(t) {
		switch mutation.name {
		case "deploy preview", "deploy", "rollback":
			family = append(family, mutation)
		}
	}
	return family
}

type lengthReader struct {
	io.Reader
}

func mutationRequest(method, path string, body []byte, streamed bool) *http.Request {
	var reader io.Reader = bytes.NewReader(body)
	if streamed {
		reader = lengthReader{bytes.NewReader(body)}
	}
	request := httptest.NewRequest(method, path, reader)
	if streamed {
		request.ContentLength = -1
	}
	return request
}

func sizedBody(t *testing.T, payload []byte, size int) []byte {
	t.Helper()
	if len(payload) >= size {
		t.Fatalf("payload length %d is not smaller than target %d", len(payload), size)
	}
	body := bytes.Repeat([]byte{' '}, size)
	copy(body, payload)
	return body
}

func assertConfigFileUnchanged(t *testing.T, configPath string, before []byte) {
	t.Helper()
	after, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("persisted config changed")
	}
}

func TestConfigMutationBodyLimit(t *testing.T) {
	cases := configMutationCases(t)
	overLimit := int(ConfigMutationBodyLimit) + 64
	for _, transport := range []struct {
		name     string
		streamed bool
	}{
		{name: "declared length", streamed: false},
		{name: "streamed", streamed: true},
	} {
		for _, mutation := range cases {
			t.Run(transport.name+"/"+mutation.name, func(t *testing.T) {
				root := t.TempDir()
				configPath := createValidTestConfig(t, root)
				before, err := os.ReadFile(configPath)
				if err != nil {
					t.Fatal(err)
				}
				body := sizedBody(t, mutation.body, overLimit)
				request := mutationRequest(http.MethodPost, mutation.path, body, transport.streamed)
				if transport.streamed && request.ContentLength != -1 {
					t.Fatalf("ContentLength = %d, want -1", request.ContentLength)
				}
				if !transport.streamed && request.ContentLength != int64(len(body)) {
					t.Fatalf("ContentLength = %d, want %d", request.ContentLength, len(body))
				}
				response := httptest.NewRecorder()
				mutation.handler(configPath, root)(response, request)
				if response.Code != http.StatusRequestEntityTooLarge {
					t.Fatalf("status = %d, want 413; body = %s", response.Code, response.Body.String())
				}
				assertConfigFileUnchanged(t, configPath, before)
			})
		}
	}
}

func TestConfigMutationBodySizeBoundaries(t *testing.T) {
	cases := configMutationCases(t)
	limit := int(ConfigMutationBodyLimit)
	for _, delta := range []int{-1, 0, 1} {
		size := limit + delta
		for _, mutation := range cases {
			t.Run(fmt.Sprintf("%s/limit%+d", mutation.name, delta), func(t *testing.T) {
				body := sizedBody(t, mutation.body, size)
				root := t.TempDir()
				configPath := createValidTestConfig(t, root)
				before, err := os.ReadFile(configPath)
				if err != nil {
					t.Fatal(err)
				}
				request := mutationRequest(http.MethodPost, mutation.path, body, false)
				response := httptest.NewRecorder()
				mutation.handler(configPath, root)(response, request)
				if delta > 0 {
					if response.Code != http.StatusRequestEntityTooLarge {
						t.Fatalf("status = %d, want 413; body = %s", response.Code, response.Body.String())
					}
					assertConfigFileUnchanged(t, configPath, before)
					return
				}
				if response.Code != mutation.parsedCode {
					t.Fatalf("status = %d, want %d; body = %s", response.Code, mutation.parsedCode, response.Body.String())
				}
			})
		}
	}
}

func TestDeployFamilyRejectsOversizedTrailingData(t *testing.T) {
	for _, mutation := range deployFamilyCases(t) {
		t.Run(mutation.name, func(t *testing.T) {
			root := t.TempDir()
			configPath := createValidTestConfig(t, root)
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			body := sizedBody(t, mutation.body, int(ConfigMutationBodyLimit)+1)
			request := mutationRequest(http.MethodPost, mutation.path, body, true)
			if request.ContentLength != -1 {
				t.Fatalf("ContentLength = %d, want -1", request.ContentLength)
			}
			response := httptest.NewRecorder()
			mutation.handler(configPath, root)(response, request)
			if response.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status = %d, want 413; body = %s", response.Code, response.Body.String())
			}
			assertConfigFileUnchanged(t, configPath, before)
		})
	}
}

func TestDeployFamilyRejectsSecondJSONValue(t *testing.T) {
	for _, mutation := range deployFamilyCases(t) {
		t.Run(mutation.name, func(t *testing.T) {
			root := t.TempDir()
			configPath := createValidTestConfig(t, root)
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			body := append(append(append([]byte{}, mutation.body...), '\n'), []byte(`{"extra":true}`)...)
			request := mutationRequest(http.MethodPost, mutation.path, body, false)
			response := httptest.NewRecorder()
			mutation.handler(configPath, root)(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 for a second JSON value; body = %s", response.Code, response.Body.String())
			}
			assertConfigFileUnchanged(t, configPath, before)
		})
	}
}

func TestDefaultsMutationRuntimeFailureRestoresPersistedConfig(t *testing.T) {
	root := t.TempDir()
	configPath := createValidTestConfig(t, root)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	original := propagateConfig
	t.Cleanup(func() { propagateConfig = original })
	calls := 0
	propagateConfig = func(string, string) error {
		calls++
		if calls == 1 {
			return errors.New("runtime rejected the config")
		}
		return nil
	}

	response := httptest.NewRecorder()
	UpdateRouterDefaultsHandler(configPath, false, root)(response, httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewReader([]byte(`{}`))))
	if response.Code != http.StatusInternalServerError || !strings.Contains(response.Body.String(), "Previous config restored.") {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	assertConfigFileUnchanged(t, configPath, before)
	if pendingActivationRecorded(configPath) {
		t.Fatal("ordinary runtime failure recorded a pending activation")
	}
	if calls != 2 {
		t.Fatalf("propagate calls = %d, want the failed apply and the successful restore", calls)
	}
}
