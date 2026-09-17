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
)

func TestProvisionRejectsInvalidSkillSelections(t *testing.T) {
	for _, skillID := range []string{
		"", "unknown", "../github", "../../outside", "/github", `..\github`,
		"./github", "github/", "github/../github", " github", "github ",
		"-github", "github-", "github--agent", "github__agent", "github_agent-",
		"GitHub", "%2e%2e%2fgithub", "github\x00", "github\n", "ｇithub",
	} {
		for _, async := range []string{"", "?async=true"} {
			t.Run(skillID+async, func(t *testing.T) {
				dataDir := t.TempDir()
				h := newTestOpenClawHandler(t, dataDir, false)
				catalogPath := filepath.Join(t.TempDir(), "skills.json")
				if err := os.WriteFile(catalogPath, []byte(`[{"id":"github"}]`), 0o600); err != nil {
					t.Fatal(err)
				}
				t.Setenv("OPENCLAW_SKILLS_PATH", catalogPath)
				// No runtime is available: validation must reject before even
				// selecting a runtime, queueing a job, or creating a workspace.
				t.Setenv(openClawContainerRuntimeDisabledEnv, "true")
				before, err := os.ReadDir(dataDir)
				if err != nil {
					t.Fatal(err)
				}
				body, err := json.Marshal(ProvisionRequest{Skills: []string{"github", skillID}})
				if err != nil {
					t.Fatal(err)
				}
				req := httptest.NewRequest(http.MethodPost, "/api/openclaw/provision"+async, bytes.NewReader(body))
				recorder := httptest.NewRecorder()
				h.ProvisionHandler().ServeHTTP(recorder, req)
				if recorder.Code != http.StatusBadRequest {
					t.Fatalf("status = %d, want 400: %s", recorder.Code, recorder.Body.String())
				}
				var response map[string]string
				if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil || response["error"] == "" {
					t.Fatalf("expected JSON error, got %s (%v)", recorder.Body.String(), err)
				}
				after, err := os.ReadDir(dataDir)
				if err != nil {
					t.Fatal(err)
				}
				if len(after) != len(before) {
					t.Fatal("invalid skill selection changed the provisioning directory")
				}
			})
		}
	}
}

func TestProvisionSkillCatalogValidation(t *testing.T) {
	tests := []struct {
		name    string
		catalog string
		skills  []string
		status  int
	}{
		{"exact IDs", `[{"id":"github"},{"id":"coding-agent"},{"id":"skill_2"}]`, []string{"github", "coding-agent", "skill_2"}, http.StatusServiceUnavailable},
		{"no selection", `invalid`, nil, http.StatusServiceUnavailable},
		{"empty selection", `invalid`, []string{}, http.StatusServiceUnavailable},
		{"empty catalog", `[]`, []string{"github"}, http.StatusBadRequest},
		{"malformed catalog", `invalid`, []string{"github"}, http.StatusInternalServerError},
		{"catalog cannot authorize traversal", `[{"id":"../outside"}]`, []string{"../outside"}, http.StatusBadRequest},
		{"catalog cannot authorize alias", `[{"id":"./github"}]`, []string{"./github"}, http.StatusBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newTestOpenClawHandler(t, t.TempDir(), false)
			catalogPath := filepath.Join(t.TempDir(), "skills.json")
			if err := os.WriteFile(catalogPath, []byte(tt.catalog), 0o600); err != nil {
				t.Fatal(err)
			}
			t.Setenv("OPENCLAW_SKILLS_PATH", catalogPath)
			t.Setenv(openClawContainerRuntimeDisabledEnv, "true")
			body, err := json.Marshal(ProvisionRequest{Skills: tt.skills})
			if err != nil {
				t.Fatal(err)
			}
			recorder := httptest.NewRecorder()
			h.ProvisionHandler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/api/openclaw/provision", bytes.NewReader(body)))
			if recorder.Code != tt.status {
				t.Fatalf("status = %d, want %d: %s", recorder.Code, tt.status, recorder.Body.String())
			}
			if strings.Contains(recorder.Body.String(), catalogPath) {
				t.Fatal("response exposed the server catalog path")
			}
		})
	}
}
