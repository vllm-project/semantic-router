package handlers

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type failingOpenClawRandomReader struct{}

func (failingOpenClawRandomReader) Read([]byte) (int, error) {
	return 0, errors.New("random source unavailable")
}

func TestOpenClawSecretGenerationFailsClosed(t *testing.T) {
	token, err := generateRandomHex(failingOpenClawRandomReader{}, 24)
	if err == nil || token != "" {
		t.Fatalf("token = %q, err = %v; want empty fail-closed result", token, err)
	}
	if _, err := generateRandomHex(strings.NewReader(strings.Repeat("a", 24)), 24); err != nil {
		t.Fatalf("bounded deterministic reader should work: %v", err)
	}
	if _, err := generateRandomHex(io.LimitReader(strings.NewReader("short"), 5), 24); err == nil {
		t.Fatal("short random read should fail")
	}
}

func TestProductionProvisionTokenIgnoresCallerChosenSecret(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	// This is a non-secret marker proving production ignores caller-selected input.
	const callerSelectedValue = "caller-selected-value"
	token, err := h.resolveOpenClawProvisionToken("new-worker", callerSelectedValue, true)
	if err != nil {
		t.Fatal(err)
	}
	if token == callerSelectedValue || len(token) != 48 {
		t.Fatalf("production token did not use 192 random bits: length=%d", len(token))
	}

	developmentToken, err := h.resolveOpenClawProvisionToken("dev-worker", "developer-choice", false)
	if err != nil || developmentToken != "developer-choice" {
		t.Fatalf("development token = %q, %v", developmentToken, err)
	}
}

func TestProductionReprovisionRotatesLegacyWeakToken(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	const legacyToken = "legacy-development-token"
	if err := h.saveRegistry([]ContainerEntry{{
		Name:  "legacy-worker",
		Token: legacyToken,
	}}); err != nil {
		t.Fatal(err)
	}

	first, err := h.resolveOpenClawProvisionToken("legacy-worker", "caller-token", true)
	if err != nil {
		t.Fatal(err)
	}
	second, err := h.resolveOpenClawProvisionToken("legacy-worker", "caller-token", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 48 || len(second) != 48 || first == second || first == legacyToken || second == legacyToken {
		t.Fatalf("production rotation failed: first_len=%d second_len=%d equal=%v legacy_reused=%v", len(first), len(second), first == second, first == legacyToken || second == legacyToken)
	}

	development, err := h.resolveOpenClawProvisionToken("legacy-worker", "caller-token", false)
	if err != nil || development != legacyToken {
		t.Fatalf("development compatibility token = %q, %v", development, err)
	}
}

func TestProductionProvisionRejectsUntrustedOrMissingConfiguredNetwork(t *testing.T) {
	pinnedImage := "ghcr.io/openclaw/openclaw@sha256:" + strings.Repeat("a", 64)
	for _, test := range []struct {
		name              string
		configuredNetwork string
		requestedNetwork  string
		wantStatus        int
	}{
		{
			name:              "arbitrary existing network",
			configuredNetwork: "trusted-network",
			requestedNetwork:  "attacker-existing-network",
			wantStatus:        http.StatusBadRequest,
		},
		{
			name:       "absent orchestrator default",
			wantStatus: http.StatusServiceUnavailable,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("DASHBOARD_SECURITY_PROFILE", "production")
			t.Setenv("OPENCLAW_DEFAULT_NETWORK_MODE", test.configuredNetwork)
			runtimeDir := t.TempDir()
			runtimePath := filepath.Join(runtimeDir, "docker")
			logPath := filepath.Join(runtimeDir, "runtime.log")
			script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
if [ "$1" = "network" ] && [ "$2" = "inspect" ]; then
  printf '%s\n' 'attacker-existing-network'
  exit 0
fi
exit 1
`
			if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
				t.Fatal(writeErr)
			}
			t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
			t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)
			h := newTestOpenClawHandler(t, t.TempDir(), false)
			body, err := json.Marshal(ProvisionRequest{
				TeamID: "team-a",
				Container: ContainerConfig{
					BaseImage:   pinnedImage,
					NetworkMode: test.requestedNetwork,
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, "/api/openclaw/workers", strings.NewReader(string(body)))
			resp := httptest.NewRecorder()
			h.ProvisionHandler().ServeHTTP(resp, req)
			if resp.Code != test.wantStatus {
				t.Fatalf("provision response = %d, want %d: %s", resp.Code, test.wantStatus, resp.Body.String())
			}
			if logBytes, readErr := os.ReadFile(logPath); readErr == nil && len(logBytes) != 0 {
				t.Fatalf("rejected production network triggered runtime mutation: %s", logBytes)
			}
		})
	}
}

func TestProductionProvisionMapsGenericUIValuesToConfiguredNetwork(t *testing.T) {
	for _, requested := range []string{"", "host", "HOST", "bridge", "  bridge  ", "trusted-network"} {
		resolved, err := resolveOpenClawProvisionNetworkMode(requested, "trusted-network", true)
		if err != nil || resolved != "trusted-network" {
			t.Errorf("resolve production network %q = %q, %v; want trusted-network", requested, resolved, err)
		}
	}
	if resolved, err := resolveOpenClawProvisionNetworkMode("attacker-network", "trusted-network", true); !errors.Is(err, errOpenClawProductionNetworkSelection) || resolved != "" {
		t.Fatalf("arbitrary production network = %q, %v; want selection rejection", resolved, err)
	}
}

func TestProductionProvisionRequiresPrecreatedConfiguredNetwork(t *testing.T) {
	t.Setenv("DASHBOARD_SECURITY_PROFILE", "production")
	t.Setenv("OPENCLAW_DEFAULT_NETWORK_MODE", "trusted-network")
	runtimeDir := t.TempDir()
	runtimePath := filepath.Join(runtimeDir, "docker")
	logPath := filepath.Join(runtimeDir, "runtime.log")
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
if [ "$1" = "network" ] && [ "$2" = "inspect" ]; then
  echo 'Error: No such network' >&2
  exit 1
fi
if [ "$1" = "network" ] && [ "$2" = "create" ]; then
  exit 0
fi
exit 1
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	body := `{"teamId":"team-a","container":{"baseImage":"ghcr.io/openclaw/openclaw@sha256:` + strings.Repeat("a", 64) + `"}}`
	req := httptest.NewRequest(http.MethodPost, "/api/openclaw/workers", strings.NewReader(body))
	resp := httptest.NewRecorder()
	h.ProvisionHandler().ServeHTTP(resp, req)
	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("provision response = %d: %s", resp.Code, resp.Body.String())
	}
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(logBytes)
	if !strings.Contains(logText, "network inspect -f {{.Name}} trusted-network") {
		t.Fatalf("configured network was not inspected exactly: %s", logText)
	}
	if strings.Contains(logText, "network create") {
		t.Fatalf("production attempted to create a missing network: %s", logText)
	}
}

func TestContainerRegistryPersistsPrivateFieldsWithoutExposingThem(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	entry := ContainerEntry{
		Name:      "worker-a",
		Port:      18790,
		Image:     "ghcr.io/openclaw/openclaw:latest",
		Token:     "private-token-canary",
		DataDir:   "/private/host/path-canary",
		CreatedAt: "2026-01-01T00:00:00Z",
	}
	if err := h.saveRegistry([]ContainerEntry{entry}); err != nil {
		t.Fatal(err)
	}
	loaded, err := h.loadRegistry()
	if err != nil || len(loaded) != 1 {
		t.Fatalf("load registry = %+v, %v", loaded, err)
	}
	if loaded[0].Token != entry.Token || loaded[0].DataDir != entry.DataDir {
		t.Fatalf("private persistence did not round trip: %+v", loaded[0])
	}

	encoded, err := json.Marshal(loaded[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{entry.Token, entry.DataDir, `"token"`, `"dataDir"`} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("public ContainerEntry JSON leaked %q: %s", forbidden, encoded)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/api/openclaw/workers", nil)
	resp := httptest.NewRecorder()
	h.WorkersHandler().ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("workers response = %d: %s", resp.Code, resp.Body.String())
	}
	for _, forbidden := range []string{entry.Token, entry.DataDir, `"token"`, `"dataDir"`} {
		if strings.Contains(resp.Body.String(), forbidden) {
			t.Fatalf("workers API leaked %q: %s", forbidden, resp.Body.String())
		}
	}
}

func installOpenClawOwnershipRuntime(
	t *testing.T,
	h *OpenClawHandler,
	containerLabels string,
	volumeExists bool,
) string {
	t.Helper()
	logPath := filepath.Join(t.TempDir(), "runtime.log")
	runtimePath := filepath.Join(t.TempDir(), "docker")
	ownerID, err := h.openClawOwnerID()
	if err != nil {
		t.Fatal(err)
	}
	containerID := strings.Repeat("a", 64)
	switch containerLabels {
	case "owned":
		containerLabels = containerID + "|/worker-a|true|" + ownerID
	case "unowned":
		containerLabels = containerID + "|/worker-a|true|different-dashboard-owner"
	}
	volumeResult := `echo "Error: No such volume" >&2
exit 1`
	if volumeExists {
		volumeResult = `printf '%s\n' 'openclaw-state-worker-a|true|` + ownerID + `'
exit 0`
	}
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
if [ "$1" = "inspect" ]; then
  printf '%s\n' '` + containerLabels + `'
  exit 0
fi
if [ "$1" = "volume" ] && [ "$2" = "inspect" ]; then
  ` + volumeResult + `
fi
exit 0
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)
	return logPath
}

func seedOpenClawLifecycleEntry(t *testing.T, h *OpenClawHandler) {
	t.Helper()
	if err := h.saveRegistry([]ContainerEntry{{
		Name:    "worker-a",
		Port:    18790,
		Image:   "ghcr.io/openclaw/openclaw:latest",
		Token:   "private-token",
		DataDir: h.containerDataDir("worker-a"),
	}}); err != nil {
		t.Fatal(err)
	}
}

func TestOpenClawLifecycleRefusesUnownedHostContainer(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	seedOpenClawLifecycleEntry(t, h)
	logPath := installOpenClawOwnershipRuntime(t, h, "unowned", false)

	for _, endpoint := range []struct {
		method string
		path   string
		body   string
		run    http.HandlerFunc
	}{
		{http.MethodPost, "/api/openclaw/start", `{"containerName":"worker-a"}`, h.StartHandler()},
		{http.MethodPost, "/api/openclaw/stop", `{"containerName":"worker-a"}`, h.StopHandler()},
		{http.MethodDelete, "/api/openclaw/containers/worker-a", "", h.DeleteHandler()},
	} {
		req := httptest.NewRequest(endpoint.method, endpoint.path, strings.NewReader(endpoint.body))
		resp := httptest.NewRecorder()
		endpoint.run.ServeHTTP(resp, req)
		if resp.Code != http.StatusConflict {
			t.Fatalf("%s %s = %d: %s", endpoint.method, endpoint.path, resp.Code, resp.Body.String())
		}
	}

	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(logBytes)
	containerID := strings.Repeat("a", 64)
	for _, forbidden := range []string{"start " + containerID, "stop " + containerID, "rm -f " + containerID} {
		if strings.Contains(logText, forbidden) {
			t.Fatalf("unowned resource received destructive command %q: %s", forbidden, logText)
		}
	}
}

func TestOpenClawLifecycleAllowsOwnedContainerAndVolume(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	seedOpenClawLifecycleEntry(t, h)
	logPath := installOpenClawOwnershipRuntime(t, h, "owned", true)

	startReq := httptest.NewRequest(
		http.MethodPost,
		"/api/openclaw/start",
		strings.NewReader(`{"containerName":"worker-a"}`),
	)
	startResp := httptest.NewRecorder()
	h.StartHandler().ServeHTTP(startResp, startReq)
	if startResp.Code != http.StatusOK {
		t.Fatalf("start = %d: %s", startResp.Code, startResp.Body.String())
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/openclaw/containers/worker-a", nil)
	deleteResp := httptest.NewRecorder()
	h.DeleteHandler().ServeHTTP(deleteResp, deleteReq)
	if deleteResp.Code != http.StatusOK {
		t.Fatalf("delete = %d: %s", deleteResp.Code, deleteResp.Body.String())
	}
	entries, err := h.loadRegistry()
	if err != nil || len(entries) != 0 {
		t.Fatalf("registry after delete = %+v, %v", entries, err)
	}

	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(logBytes)
	containerID := strings.Repeat("a", 64)
	for _, required := range []string{"start " + containerID, "rm -f " + containerID, "volume rm openclaw-state-worker-a"} {
		if !strings.Contains(logText, required) {
			t.Fatalf("owned lifecycle missing %q: %s", required, logText)
		}
	}
}

func TestProductionOpenClawTargetRejectsCapturedContainerName(t *testing.T) {
	for _, test := range []struct {
		name      string
		ownership string
		wantOK    bool
	}{
		{name: "owned", ownership: "owned", wantOK: true},
		{name: "captured", ownership: "unowned", wantOK: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("DASHBOARD_SECURITY_PROFILE", "production")
			h := newTestOpenClawHandler(t, t.TempDir(), false)
			seedOpenClawLifecycleEntry(t, h)
			installOpenClawOwnershipRuntime(t, h, test.ownership, false)
			_, ok := h.TargetBaseForContainer("worker-a")
			if ok != test.wantOK {
				t.Fatalf("TargetBaseForContainer() ok = %v, want %v", ok, test.wantOK)
			}
		})
	}
}

func TestOpenClawWorkerEndpointRepairRechecksOwnershipBeforeRestart(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	workerDataDir := h.containerDataDir("worker-a")
	if err := os.MkdirAll(workerDataDir, 0o700); err != nil {
		t.Fatal(err)
	}
	worker := ContainerEntry{
		Name:    "worker-a",
		Port:    18790,
		Image:   "ghcr.io/openclaw/openclaw:latest",
		DataDir: workerDataDir,
	}
	if err := h.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatal(err)
	}
	if err := writePrivateOpenClawFile(filepath.Join(workerDataDir, "openclaw.json"), []byte(`{}`)); err != nil {
		t.Fatal(err)
	}

	ownerID, err := h.openClawOwnerID()
	if err != nil {
		t.Fatal(err)
	}
	containerID := strings.Repeat("d", 64)
	runtimeDir := t.TempDir()
	runtimePath := filepath.Join(runtimeDir, "docker")
	logPath := filepath.Join(runtimeDir, "runtime.log")
	inspectCountPath := filepath.Join(runtimeDir, "inspect-count")
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
if [ "$1" = "inspect" ]; then
  count=0
  if [ -f "$OPENCLAW_TEST_INSPECT_COUNT" ]; then
    count=$(sed -n '1p' "$OPENCLAW_TEST_INSPECT_COUNT")
  fi
  count=$((count + 1))
  printf '%s\n' "$count" > "$OPENCLAW_TEST_INSPECT_COUNT"
  if [ "$count" -eq 1 ]; then
    printf '%s\n' '` + containerID + `|/worker-a|true|` + ownerID + `'
  else
    printf '%s\n' '` + containerID + `|/worker-a|true|different-dashboard-owner'
  fi
  exit 0
fi
exit 1
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)
	t.Setenv("OPENCLAW_TEST_INSPECT_COUNT", inspectCountPath)

	recovered, err := h.ensureWorkerChatEndpoint(worker)
	if err == nil || recovered {
		t.Fatalf("endpoint recovery = %v, %v; want fail-closed ownership conflict", recovered, err)
	}
	if strings.Contains(err.Error(), "different-dashboard-owner") || strings.Contains(err.Error(), runtimePath) {
		t.Fatalf("endpoint recovery exposed runtime ownership details: %v", err)
	}
	logBytes, readErr := os.ReadFile(logPath)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if strings.Contains(string(logBytes), "restart ") {
		t.Fatalf("captured container name was restarted: %s", logBytes)
	}
}

func TestOpenClawStatusRejectsForeignAndUnlabeledContainers(t *testing.T) {
	for _, test := range []struct {
		name   string
		labels string
	}{
		{name: "foreign", labels: "true|different-dashboard-owner"},
		{name: "unlabeled", labels: "|"},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newTestOpenClawHandler(t, t.TempDir(), false)
			seedOpenClawLifecycleEntry(t, h)
			containerID := strings.Repeat("e", 64)
			runtimeDir := t.TempDir()
			runtimePath := filepath.Join(runtimeDir, "docker")
			logPath := filepath.Join(runtimeDir, "runtime.log")
			script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
if [ "$1" = "inspect" ]; then
  printf '%s\n' '` + containerID + `|/worker-a|` + test.labels + `'
  exit 0
fi
exit 1
`
			if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
				t.Fatal(writeErr)
			}
			t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
			t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)

			req := httptest.NewRequest(http.MethodGet, "/api/openclaw/status?name=worker-a", nil)
			resp := httptest.NewRecorder()
			h.StatusHandler().ServeHTTP(resp, req)
			if resp.Code != http.StatusOK {
				t.Fatalf("status response = %d: %s", resp.Code, resp.Body.String())
			}
			var status OpenClawStatus
			if err := json.Unmarshal(resp.Body.Bytes(), &status); err != nil {
				t.Fatal(err)
			}
			if status.Error != "Container ownership conflict" || status.Running || status.Healthy {
				t.Fatalf("captured status = %+v", status)
			}
			if status.GatewayURL != "" {
				t.Fatalf("captured status exposed/probed gateway URL %q", status.GatewayURL)
			}
			logBytes, err := os.ReadFile(logPath)
			if err != nil {
				t.Fatal(err)
			}
			logText := string(logBytes)
			for _, forbidden := range []string{"{{.State.Running}}", "logs --tail"} {
				if strings.Contains(logText, forbidden) {
					t.Fatalf("captured container received %q status operation: %s", forbidden, logText)
				}
			}
		})
	}
}

func TestOpenClawStatusUsesImmutableIDAfterNameCapture(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	worker := ContainerEntry{
		Name:  "worker-a",
		Port:  mustServerPort(t, server.URL),
		Image: "ghcr.io/openclaw/openclaw:latest",
	}
	if err := h.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatal(err)
	}
	ownerID, err := h.openClawOwnerID()
	if err != nil {
		t.Fatal(err)
	}
	containerID := strings.Repeat("f", 64)
	runtimeDir := t.TempDir()
	runtimePath := filepath.Join(runtimeDir, "docker")
	logPath := filepath.Join(runtimeDir, "runtime.log")
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
if [ "$1" = "inspect" ] && [ "$3" = "{{.State.Running}}" ]; then
  if [ "$4" = "` + containerID + `" ]; then
    printf '%s\n' true
    exit 0
  fi
  printf '%s\n' false
  exit 0
fi
if [ "$1" = "inspect" ]; then
  printf '%s\n' '` + containerID + `|/worker-a|true|` + ownerID + `'
  exit 0
fi
if [ "$1" = "logs" ] && [ "$4" = "` + containerID + `" ]; then
  printf '%s\n' '[gateway] listening on ws://127.0.0.1'
  exit 0
fi
exit 1
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)

	status := h.checkContainerHealth(worker)
	if !status.Running || !status.Healthy || status.Error != "" {
		t.Fatalf("immutable status = %+v", status)
	}
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(logBytes)
	for _, required := range []string{
		"inspect -f {{.State.Running}} " + containerID,
		"logs --tail 80 " + containerID,
	} {
		if !strings.Contains(logText, required) {
			t.Fatalf("status did not bind to immutable ID %q: %s", required, logText)
		}
	}
	for _, forbidden := range []string{
		"inspect -f {{.State.Running}} worker-a",
		"logs --tail 80 worker-a",
	} {
		if strings.Contains(logText, forbidden) {
			t.Fatalf("status used captured mutable name %q: %s", forbidden, logText)
		}
	}
}

func TestOpenClawStatusDoesNotInspectStaleRegistryEntry(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	seedOpenClawLifecycleEntry(t, h)
	entries, err := h.loadRegistry()
	if err != nil || len(entries) != 1 {
		t.Fatalf("registry = %+v, %v", entries, err)
	}
	if saveErr := h.saveRegistry(nil); saveErr != nil {
		t.Fatal(saveErr)
	}
	runtimeDir := t.TempDir()
	runtimePath := filepath.Join(runtimeDir, "docker")
	logPath := filepath.Join(runtimeDir, "runtime.log")
	if writeErr := os.WriteFile(runtimePath, []byte("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$OPENCLAW_TEST_RUNTIME_LOG\"\nexit 1\n"), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	if writeErr := os.WriteFile(logPath, nil, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)

	status := h.checkContainerHealth(entries[0])
	if status.Error != "Container not in registry" || status.ContainerName != "" || status.GatewayURL != "" {
		t.Fatalf("stale registry status = %+v", status)
	}
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(logBytes) != 0 {
		t.Fatalf("stale registry entry triggered runtime operations: %s", logBytes)
	}
}

func TestProductionOpenClawConfigOmitsInsecureBrowserControls(t *testing.T) {
	t.Setenv("DASHBOARD_SECURITY_PROFILE", "production")
	path := filepath.Join(t.TempDir(), "openclaw.json")
	req := ProvisionRequest{Container: ContainerConfig{
		GatewayPort:    18790,
		AuthToken:      "private-token-canary",
		ModelBaseURL:   "https://models.example.test/v1",
		ModelAPIKey:    "private-model-key-canary",
		ModelName:      "router",
		BrowserEnabled: true,
	}}
	if err := writeOpenClawConfig(path, req); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{
		"dangerouslyDisableDeviceAuth",
		"allowInsecureAuth",
		"allowedOrigins",
		`"browser"`,
		`"noSandbox"`,
	} {
		if strings.Contains(string(data), forbidden) {
			t.Fatalf("production config contains %q: %s", forbidden, data)
		}
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("config mode = %o, want 600", info.Mode().Perm())
	}
}

func TestOpenClawOwnershipUsesImmutableContainerIdentity(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	ownerID, err := h.openClawOwnerID()
	if err != nil {
		t.Fatal(err)
	}
	containerID := strings.Repeat("b", 64)
	runtimePath := filepath.Join(t.TempDir(), "docker")
	script := `#!/bin/sh
if [ "$1" = "inspect" ]; then
  printf '%s\n' '` + containerID + `|/worker-a|true|` + ownerID + `'
  exit 0
fi
exit 0
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)

	if err := h.verifyProvisionedOwnedContainer("worker-a", containerID); err != nil {
		t.Fatalf("verify immutable ID: %v", err)
	}
	if err := h.verifyProvisionedOwnedContainer("captured-name", containerID); !errors.Is(err, errOpenClawResourceConflict) {
		t.Fatalf("name mismatch error = %v, want ownership conflict", err)
	}
	if _, err := parseOpenClawContainerID([]byte(containerID + "\nwarning")); err == nil {
		t.Fatal("accepted mixed stdout instead of one immutable container ID")
	}
}

func TestOpenClawInspectDoesNotTreatGenericNotFoundAsMissing(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	runtimePath := filepath.Join(t.TempDir(), "docker")
	script := `#!/bin/sh
echo "daemon endpoint not found because access is denied" >&2
exit 1
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	inspection, err := h.inspectOpenClawOwnedResource(openClawContainerResource, "worker-a")
	if err == nil || inspection.exists {
		t.Fatalf("inspection = %+v, err = %v; generic runtime failure must fail closed", inspection, err)
	}
}

func TestEnsureOwnedVolumeCreatesAndRevalidatesLabels(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	ownerID, err := h.openClawOwnerID()
	if err != nil {
		t.Fatal(err)
	}
	runtimeDir := t.TempDir()
	runtimePath := filepath.Join(runtimeDir, "docker")
	statePath := filepath.Join(runtimeDir, "created")
	logPath := filepath.Join(runtimeDir, "runtime.log")
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
if [ "$1" = "volume" ] && [ "$2" = "inspect" ]; then
  if [ -f "$OPENCLAW_TEST_VOLUME_STATE" ]; then
    printf '%s\n' 'openclaw-state-worker-a|true|` + ownerID + `'
    exit 0
  fi
  echo "Error: No such volume" >&2
  exit 1
fi
if [ "$1" = "volume" ] && [ "$2" = "create" ]; then
  : > "$OPENCLAW_TEST_VOLUME_STATE"
  printf '%s\n' 'openclaw-state-worker-a'
  exit 0
fi
exit 1
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)
	t.Setenv("OPENCLAW_TEST_VOLUME_STATE", statePath)
	if ensureErr := h.ensureOwnedVolume("openclaw-state-worker-a"); ensureErr != nil {
		t.Fatalf("ensure owned volume: %v", ensureErr)
	}
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, marker := range []string{
		"volume create",
		"--label " + openClawManagedLabel,
		"--label " + openClawOwnerLabelKey + "=" + ownerID,
	} {
		if !strings.Contains(string(logBytes), marker) {
			t.Fatalf("volume runtime missing %q: %s", marker, logBytes)
		}
	}
}
