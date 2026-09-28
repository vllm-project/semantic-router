package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func TestConfigMutationRevokedAtFileCommitRestoresPreviousConfig(t *testing.T) {
	for _, test := range []struct {
		name    string
		path    string
		body    func(*testing.T) []byte
		handler func(string, string) http.HandlerFunc
	}{
		{
			name: "update", path: "/api/router/config/update",
			body: func(t *testing.T) []byte {
				t.Helper()
				body, err := json.Marshal(canonicalConfigBody("127.0.0.1:8000"))
				if err != nil {
					t.Fatal(err)
				}
				return body
			},
			handler: func(path, dir string) http.HandlerFunc { return UpdateConfigHandler(path, false, dir) },
		},
		{
			name: "deploy", path: "/api/router/config/deploy",
			body: func(t *testing.T) []byte {
				t.Helper()
				body, err := json.Marshal(DeployRequest{YAML: "routing:\n  decisions: []\n"})
				if err != nil {
					t.Fatal(err)
				}
				return body
			},
			handler: func(path, dir string) http.HandlerFunc { return DeployHandler(path, false, dir) },
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			configPath := createValidTestConfig(t, root)
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			var allowed atomic.Bool
			allowed.Store(true)
			var configCommits atomic.Int32
			originalRename := atomicRename
			t.Cleanup(func() { atomicRename = originalRename })
			atomicRename = func(source, destination string) error {
				renameErr := originalRename(source, destination)
				if renameErr == nil && filepath.Clean(destination) == filepath.Clean(configPath) && configCommits.Add(1) == 1 {
					allowed.Store(false)
				}
				return renameErr
			}
			request := httptest.NewRequest(http.MethodPost, test.path, bytes.NewReader(test.body(t)))
			request = request.WithContext(auth.WithPermissionRevalidator(request.Context(), func(context.Context) error {
				if !allowed.Load() {
					return auth.ErrPermissionDenied
				}
				return nil
			}))
			response := httptest.NewRecorder()
			test.handler(configPath, root)(response, request)
			if response.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want 403: %s", response.Code, response.Body.String())
			}
			if configCommits.Load() < 2 {
				t.Fatalf("config commits = %d, want new write and rollback", configCommits.Load())
			}
			after, err := os.ReadFile(configPath)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatalf("revoked config was not restored: %v", err)
			}
		})
	}
}
