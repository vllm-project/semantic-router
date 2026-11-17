package main

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestBenchConfigsLoadInTheRouter generates both bench configurations from the
// maintained configuration and loads each as the Router does: the bench model
// name reaches the fusion decision, which grounds only in the on arm, and
// there the Router starts the deployment the grounding question asks.
func TestBenchConfigsLoadInTheRouter(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "..")
	out := t.TempDir()
	// #nosec G204 -- python3 and the generator are fixed; the paths are the repository's and the test's.
	generate := exec.Command("python3", "-m", "bench.grounded_fusion.make_configs",
		"--base", filepath.Join(root, "config", "config.yaml"), "--out-dir", out)
	generate.Dir = root
	output, err := generate.CombinedOutput()
	require.NoError(t, err, "%s", output)

	for arm, grounding := range map[string]bool{"on": true, "off": false} {
		cfg, err := config.Parse(filepath.Join(out, "config-fusion-"+arm+".yaml"))
		require.NoError(t, err, arm)
		entrypoint, ok := cfg.ResolveEntrypoint(config.ChatAPI, "vllm-sr/fusion")
		require.True(t, ok, arm)
		recipe, ok := cfg.RecipeByName(entrypoint.Recipe)
		require.True(t, ok, arm)
		require.Len(t, recipe.Profile.Decisions, 1, arm)
		fusion := recipe.Profile.Decisions[0].Algorithm.Fusion
		require.NotNil(t, fusion, arm)
		require.NotNil(t, fusion.Grounding, arm)
		require.Equal(t, grounding, fusion.Grounding.Enabled, arm)
		if grounding {
			deployment, _ := cfg.ApplicationTypeDeployment(config.SignalTypeHallucination)
			require.Contains(t, config.ModelRuntimeDeploymentsInUse(cfg), deployment, arm)
		}
	}
}
