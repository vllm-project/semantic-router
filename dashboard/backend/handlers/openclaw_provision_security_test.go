package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestOpenClawPrivateWorkspaceRejectsSymlinkCanaries(t *testing.T) {
	workspace := t.TempDir()
	target := filepath.Join(t.TempDir(), "outside-canary")
	if err := os.WriteFile(target, []byte("outside-must-not-change"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(workspace, "SOUL.md")); err != nil {
		t.Skipf("symlink unsupported: %v", err)
	}
	if err := writeIdentityFiles(workspace, IdentityConfig{Name: "attacker"}); err == nil {
		t.Fatal("identity writer followed a symlink canary")
	}
	data, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "outside-must-not-change" {
		t.Fatalf("outside symlink target changed: %q", data)
	}

	directoryTarget := t.TempDir()
	directoryLink := filepath.Join(t.TempDir(), "skills")
	if err := os.Symlink(directoryTarget, directoryLink); err != nil {
		t.Skipf("directory symlink unsupported: %v", err)
	}
	if err := ensureOpenClawDirectory(directoryLink, 0o700); err == nil {
		t.Fatal("private directory validator accepted a symlink")
	}

	identityTarget := filepath.Join(t.TempDir(), "identity-canary")
	if err := os.WriteFile(identityTarget, []byte("- **Name:** leaked-name\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	identityDataDir := t.TempDir()
	identityWorkspace := filepath.Join(identityDataDir, "workspace")
	if err := os.Mkdir(identityWorkspace, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(identityTarget, filepath.Join(identityWorkspace, "IDENTITY.md")); err != nil {
		t.Skipf("identity symlink unsupported: %v", err)
	}
	if snapshot := readIdentitySnapshot(identityDataDir); snapshot.Name != "" {
		t.Fatalf("identity reader followed symlink: %+v", snapshot)
	}
}

func TestOpenClawProvisionArtifactsAreBoundedAndSafelyEncoded(t *testing.T) {
	req := ProvisionRequest{Container: ContainerConfig{
		ContainerName: "worker-a",
		GatewayPort:   18790,
		BaseImage:     "ghcr.io/openclaw/openclaw@sha256:" + strings.Repeat("a", 64),
		NetworkMode:   "vllm-sr-network",
	}}
	dataDir := "/tmp/work dir/it's-private"
	command := generateDockerRunCmd("docker", req, dataDir, "owner-id")
	for _, marker := range []string{
		"volume create",
		"--label " + openClawManagedLabel,
		openClawShellQuote(dataDir + "/workspace:/workspace"),
		"--health-cmd",
	} {
		if !strings.Contains(command, marker) {
			t.Fatalf("generated command missing %q: %s", marker, command)
		}
	}
	compose := generateComposeYAML(req, dataDir, "owner-id")
	var document map[string]any
	if err := yaml.Unmarshal([]byte(compose), &document); err != nil {
		t.Fatalf("generated compose is invalid YAML: %v\n%s", err, compose)
	}
	for _, forbidden := range []string{"authToken", "modelApiKey"} {
		if strings.Contains(command, forbidden) || strings.Contains(compose, forbidden) {
			t.Fatalf("provision artifact leaked %q", forbidden)
		}
	}

	firstArgs := openClawContainerRunArgs(req, dataDir, "owner-id")
	req.Container.GatewayPort++
	secondArgs := openClawContainerRunArgs(req, dataDir, "owner-id")
	if strings.Join(firstArgs, " ") == strings.Join(secondArgs, " ") ||
		!strings.Contains(strings.Join(secondArgs, " "), "18791/health") {
		t.Fatalf("retry args retained stale health port: %v", secondArgs)
	}
}

func TestProductionProvisionResponseOmitsHostAndRuntimeDetails(t *testing.T) {
	response := publicOpenClawProvisionResponse(true, ProvisionResponse{
		Success:      true,
		Message:      "ok",
		WorkspaceDir: "/private/host/workspace-canary",
		ConfigPath:   "/private/host/config-canary",
		ContainerID:  strings.Repeat("c", 64),
		DockerCmd:    "private-command-canary",
		ComposeYAML:  "private-compose-canary",
	})
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	for _, canary := range []string{"private/host", "private-command", "private-compose", strings.Repeat("c", 64)} {
		if strings.Contains(string(encoded), canary) {
			t.Fatalf("production response leaked %q: %s", canary, encoded)
		}
	}
}

func TestOpenClawProvisionRequestBoundsAndCanonicalWorkerPath(t *testing.T) {
	req := ProvisionRequest{Container: ContainerConfig{GatewayPort: 65536}}
	if err := validateOpenClawProvisionRequest(req); err == nil {
		t.Fatal("accepted invalid gateway port")
	}
	req = ProvisionRequest{Identity: IdentityConfig{Principles: strings.Repeat("x", 16*1024+1)}}
	if err := validateOpenClawProvisionRequest(req); err == nil {
		t.Fatal("accepted oversized identity field")
	}

	h := newTestOpenClawHandler(t, t.TempDir(), false)
	seedOpenClawLifecycleEntry(t, h)
	httpReq := httptest.NewRequest(http.MethodGet, "/api/openclaw/workers/../worker-a", nil)
	response := httptest.NewRecorder()
	h.WorkerByIDHandler().ServeHTTP(response, httpReq)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("noncanonical worker alias status = %d: %s", response.Code, response.Body.String())
	}
}

func TestOpenClawSkillExtractionUsesConstrainedContainer(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	runtimeDir := t.TempDir()
	runtimePath := filepath.Join(runtimeDir, "docker")
	logPath := filepath.Join(runtimeDir, "runtime.log")
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$OPENCLAW_TEST_RUNTIME_LOG"
printf '%s' '# bounded skill'
`
	if writeErr := os.WriteFile(runtimePath, []byte(script), 0o755); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_TEST_RUNTIME_LOG", logPath)
	content, err := h.fetchOpenClawSkillContentForProvision("safe-skill", "safe-image:local")
	if err != nil || content != "# bounded skill" {
		t.Fatalf("skill extraction = %q, %v", content, err)
	}
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, marker := range []string{
		"--network none",
		"--read-only",
		"--user 65534:65534",
		"--cap-drop ALL",
		"--security-opt no-new-privileges:true",
		"--pids-limit 32",
		"--memory 128m",
		"--cpus 0.25",
		"--entrypoint cat",
	} {
		if !strings.Contains(string(logBytes), marker) {
			t.Fatalf("skill extraction missing %q: %s", marker, logBytes)
		}
	}
}
