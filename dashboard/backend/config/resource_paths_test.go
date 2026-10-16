package config

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestDashboardDevLauncherPassesSharedAssetRoot(t *testing.T) {
	repoRoot, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	bin := t.TempDir()
	// Observe the actual launch environment without starting a Dashboard server.
	if err := os.WriteFile(filepath.Join(bin, "go"), []byte("#!/bin/sh\nprintf 'ASSET_ROOT=%s\\n' \"$VLLM_SR_CONFIG_BASE_DIR\"\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	for _, override := range []string{"", filepath.Join(t.TempDir(), "custom assets")} {
		t.Run(override, func(t *testing.T) {
			t.Setenv("VLLM_SR_CONFIG_BASE_DIR", override)
			cmd := exec.Command("make", "--no-print-directory", "-f", "tools/make/dashboard.mk", "dashboard-dev-backend")
			cmd.Dir = repoRoot
			output, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("development launcher: %v\n%s", err, output)
			}
			want := override
			if want == "" {
				want = repoRoot
			}
			if !strings.Contains(string(output), "ASSET_ROOT="+want+"\n") {
				t.Fatalf("launcher did not pass resource root %q: %s", want, output)
			}
		})
	}
}

func TestResolveConfigPathsSeparatesResourcesFromConfigAndState(t *testing.T) {
	for _, name := range []string{"config", "assets", ".vllm-sr"} {
		t.Run(name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), name)
			if err := os.MkdirAll(root, 0o755); err != nil {
				t.Fatal(err)
			}
			state := t.TempDir()
			t.Setenv("VLLM_SR_CONFIG_BASE_DIR", root)
			t.Setenv("DASHBOARD_CONFIG_DIR", state)
			cfg := &Config{ConfigFile: filepath.Join(t.TempDir(), "runtime.yaml")}
			if err := resolveConfigPaths(cfg); err != nil {
				t.Fatal(err)
			}
			if cfg.ConfigBaseDir != root || cfg.ConfigDir != state {
				t.Fatalf("resource root = %q, state = %q; want %q, %q", cfg.ConfigBaseDir, cfg.ConfigDir, root, state)
			}
		})
	}
}

func TestResolveConfigPathsDefaultsResourcesToWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	t.Setenv("VLLM_SR_CONFIG_BASE_DIR", "")
	t.Setenv("DASHBOARD_CONFIG_DIR", t.TempDir())
	cfg := &Config{ConfigFile: filepath.Join(t.TempDir(), "config.yaml")}
	if err := resolveConfigPaths(cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.ConfigBaseDir != root {
		t.Fatalf("ConfigBaseDir = %q, want process working directory %q", cfg.ConfigBaseDir, root)
	}
}

func TestResolveConfigPathsRejectsInvalidResourceRoots(t *testing.T) {
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, []byte("fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, root := range []string{"relative-root", filepath.Join(t.TempDir(), "absent"), file} {
		t.Run(root, func(t *testing.T) {
			t.Setenv("VLLM_SR_CONFIG_BASE_DIR", root)
			if err := resolveConfigPaths(&Config{ConfigFile: "config.yaml"}); err == nil {
				t.Fatalf("invalid resource root %q was accepted", root)
			}
		})
	}
}
