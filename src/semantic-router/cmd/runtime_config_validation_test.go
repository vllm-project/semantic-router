package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Exercise the actual startup loader in a subprocess because invalid config
// terminates the process. Returning from this boundary would allow startup to
// create registries, servers, model runtimes, and background workers.
func TestStartupRejectsInvalidKubernetesGlobals(t *testing.T) {
	const configPathEnv = "VSR_TEST_STARTUP_VALIDATION_CONFIG"
	if path := os.Getenv(configPathEnv); path != "" {
		initializeRuntimeLogger()
		loadRuntimeConfigOrFatal(path)
		os.Exit(0)
	}

	cases := []struct {
		name   string
		global string
		want   string
	}{
		{"cache", "  stores: {response_cache: {enabled: true, similarity_threshold: 1.5}}", "similarity_threshold"},
		{"memory", "  stores: {memory: {default_similarity_threshold: 1.5}}", "default_similarity_threshold"},
		{"admission", "  model_catalog: {admission: {prompt_guard: {max_concurrency: 1, max_queue: -1}}}", "max_queue"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			raw := "version: v0.3\nglobal:\n  router: {config_source: kubernetes}\n" + tc.global + "\n"
			if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			//nolint:gosec // G204: runs the current Go test binary with a fixed test selector.
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestStartupRejectsInvalidKubernetesGlobals$")
			cmd.Env = append(os.Environ(), configPathEnv+"="+path)
			output, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
				t.Fatalf("startup loader must exit with code 1, got %v\n%s", err, output)
			}
			for _, want := range []string{"runtime_config_load_failed", tc.want} {
				if !strings.Contains(string(output), want) {
					t.Errorf("startup output must contain %q:\n%s", want, output)
				}
			}
		})
	}
}
