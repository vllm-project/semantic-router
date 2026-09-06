package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLogsHandlerReadsBoundedSpoolAndRedactsSecrets(t *testing.T) {
	t.Parallel()
	root := newTestLogSpool(t)
	secret := "dashboard-log-secret"                        //nolint:gosec // Canary verifies that credentials are redacted from logs.
	openAIKey := "s" + "k-" + "runtimeRedactionCanary12345" //nolint:gosec // Built at runtime so repository secret scanners do not mistake the test fixture for a live key.
	content := strings.Join([]string{
		`[info] router ready`,
		`Authorization: Bearer ` + secret,
		`provider=https://user:pass@example.com/v1?token=` + secret + `&ok=yes`,
		`OPENAI_API_KEY=` + openAIKey,
		`-----BEGIN PRIVATE KEY-----`,
		`cHJpdmF0ZS1rZXktbWF0ZXJpYWwtdGhhdC1tdXN0LW5ldmVyLWJlLXJldHVybmVkLXRvLXVzZXJz`,
		`-----END PRIVATE KEY-----`,
		`[info] request complete`,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(root, "router.log"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/logs?component=router&lines=20", nil)
	response := httptest.NewRecorder()
	newLogsHandler(root).ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), secret) || strings.Contains(response.Body.String(), "user:pass") || strings.Contains(response.Body.String(), openAIKey) {
		t.Fatalf("logs response leaked a credential: %s", response.Body.String())
	}
	var payload LogsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if !payload.Supported || payload.DeploymentType != "local-spool" {
		t.Fatalf("unexpected spool contract: %#v", payload)
	}
	if payload.Count != len(payload.Logs) || payload.Count < 3 {
		t.Fatalf("unexpected log count: %#v", payload)
	}
	if !strings.Contains(response.Body.String(), "[REDACTED PRIVATE KEY]") {
		t.Fatalf("private-key block was not represented safely: %s", response.Body.String())
	}
	if response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("Cache-Control = %q", response.Header().Get("Cache-Control"))
	}
}

func TestLogsHandlerReportsUnsupportedWithoutSpool(t *testing.T) {
	t.Parallel()
	request := httptest.NewRequest(http.MethodGet, "/api/logs?component=all", nil)
	response := httptest.NewRecorder()
	newLogsHandler("").ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	var payload LogsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Supported || payload.DeploymentType != "unsupported" || payload.Message != logsUnsupportedNotice || payload.Error != "" {
		t.Fatalf("unexpected unsupported contract: %#v", payload)
	}
}

func TestLogsHandlerRejectsUnknownComponent(t *testing.T) {
	t.Parallel()
	request := httptest.NewRequest(http.MethodGet, "/api/logs?component=../../etc/passwd", nil)
	response := httptest.NewRecorder()
	newLogsHandler(t.TempDir()).ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "invalid_component") {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestLogsHandlerRejectsNonGETRequests(t *testing.T) {
	t.Parallel()
	request := httptest.NewRequest(http.MethodPost, "/api/logs", strings.NewReader("{}"))
	response := httptest.NewRecorder()
	newLogsHandler(t.TempDir()).ServeHTTP(response, request)

	if response.Code != http.StatusMethodNotAllowed || response.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("status = %d, Allow = %q, body = %s", response.Code, response.Header().Get("Allow"), response.Body.String())
	}
}

func TestLogsHandlerRejectsSymlinkedComponentFile(t *testing.T) {
	t.Parallel()
	root := newTestLogSpool(t)
	outside := filepath.Join(t.TempDir(), "outside.log")
	if err := os.WriteFile(outside, []byte("must-not-be-read\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "router.log")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "router.log")); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/logs?component=router", nil)
	response := httptest.NewRecorder()
	newLogsHandler(root).ServeHTTP(response, request)

	if response.Code != http.StatusOK || strings.Contains(response.Body.String(), "must-not-be-read") {
		t.Fatalf("unsafe file was exposed: status = %d, body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "could not be read safely") {
		t.Fatalf("unsafe spool did not fail closed: %s", response.Body.String())
	}
}

func TestLogsHandlerRejectsSymlinkedSpoolRoot(t *testing.T) {
	t.Parallel()
	realRoot := newTestLogSpool(t)
	if err := os.WriteFile(filepath.Join(realRoot, "router.log"), []byte("must-not-be-read\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	rootLink := filepath.Join(t.TempDir(), "spool")
	if err := os.Symlink(realRoot, rootLink); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/logs?component=router", nil)
	response := httptest.NewRecorder()
	newLogsHandler(rootLink).ServeHTTP(response, request)

	if response.Code != http.StatusOK || strings.Contains(response.Body.String(), "must-not-be-read") {
		t.Fatalf("symlinked root was exposed: status = %d, body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "unavailable or unsafe") {
		t.Fatalf("unsafe root did not fail closed: %s", response.Body.String())
	}
}

func TestLogsHandlerRejectsHardLinkedComponentFile(t *testing.T) {
	t.Parallel()
	root := newTestLogSpool(t)
	outside := filepath.Join(t.TempDir(), "outside.log")
	if err := os.WriteFile(outside, []byte("must-not-be-read\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "router.log")); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(outside, filepath.Join(root, "router.log")); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/logs?component=router", nil)
	response := httptest.NewRecorder()
	newLogsHandler(root).ServeHTTP(response, request)

	if response.Code != http.StatusOK || strings.Contains(response.Body.String(), "must-not-be-read") {
		t.Fatalf("hard-linked file was exposed: status = %d, body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "could not be read safely") {
		t.Fatalf("unsafe spool did not fail closed: %s", response.Body.String())
	}
}

func TestLogsHandlerBoundsResponseBytes(t *testing.T) {
	t.Parallel()
	root := newTestLogSpool(t)
	line := strings.Repeat(`\"`, maxRuntimeLogLineBytes/2)
	content := strings.Repeat(line+"\n", 80)
	if err := os.WriteFile(filepath.Join(root, "dashboard.log"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/logs?component=dashboard&lines=1000", nil)
	response := httptest.NewRecorder()
	newLogsHandler(root).ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body prefix = %.200s", response.Code, response.Body.String())
	}
	if response.Body.Len() > maxLogsResponseBytes {
		t.Fatalf("response bytes = %d, limit = %d", response.Body.Len(), maxLogsResponseBytes)
	}
}

func TestLogSpoolReaderBoundsIndividualLines(t *testing.T) {
	t.Parallel()
	root := newTestLogSpool(t)
	content := strings.Repeat("x", maxRuntimeLogLineBytes*2) + "\n"
	if err := os.WriteFile(filepath.Join(root, "router.log"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	reader, err := openLogSpoolReader(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	lines, err := reader.Tail("router", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 1 || len(lines[0]) != maxRuntimeLogLineBytes || !strings.HasSuffix(lines[0], runtimeLogTruncatedMarker) {
		t.Fatalf("unexpected bounded line: count = %d, bytes = %d", len(lines), len(lines[0]))
	}
}

func newTestLogSpool(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, component := range logSpoolComponents {
		if err := os.WriteFile(filepath.Join(root, component+".log"), nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}
