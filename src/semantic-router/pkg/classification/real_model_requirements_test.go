package classification

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Exercise testing's actual skip/fail outcomes in a subprocess so required CI
// cannot silently pass when a download is absent or incomplete.
func TestRealModelArtifactRequirements(t *testing.T) {
	const probe = "VLLM_SR_MODEL_REQUIREMENT_PROBE"
	const override = "VLLM_SR_MODEL_REQUIREMENT_PATH"
	if os.Getenv(probe) == "1" {
		requireRealModel(t, override, config.DefaultSystemModels().PromptGuard)
		return
	}
	executable, executableErr := os.Executable()
	if executableErr != nil {
		t.Fatalf("resolve current test executable: %v", executableErr)
	}
	root := t.TempDir()
	packageDir := filepath.Join(root, "src", "semantic-router", "pkg", "classification")
	if err := os.MkdirAll(packageDir, 0o700); err != nil {
		t.Fatal(err)
	}
	// Even an apparently complete ambient cache must not activate inference.
	cached := filepath.Join(root, config.DefaultSystemModels().PromptGuard)
	if err := os.MkdirAll(cached, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "tokenizer.json"} {
		if err := os.WriteFile(filepath.Join(cached, name), []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	incomplete := filepath.Join(root, "incomplete")
	if err := os.Mkdir(incomplete, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, required, path, message string
		wantFailure                   bool
	}{
		{"ambient cache without opt in", "", "", "--- SKIP:", false},
		{"required missing explicit path", "1", "", "needs explicit", true},
		{"explicit artifact", "", cached, "--- PASS:", false},
		{"explicit absent override", "", filepath.Join(root, "missing"), "required real model", true},
		{"incomplete local download", "", incomplete, "is incomplete", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.Command(executable, "-test.run=^TestRealModelArtifactRequirements$", "-test.v")
			cmd.Dir = packageDir
			cmd.Env = append(os.Environ(), probe+"=1", "VLLM_SR_REQUIRE_MODEL_TESTS="+tc.required, override+"="+tc.path)
			output, err := cmd.CombinedOutput()
			if (err != nil) != tc.wantFailure || !strings.Contains(string(output), tc.message) {
				t.Fatalf("unexpected real-model requirement outcome: err=%v output=%s", err, output)
			}
		})
	}
}

func TestLegacyModelArtifactRequirements(t *testing.T) {
	const probe = "VLLM_SR_LEGACY_REQUIREMENT_PROBE"
	const override = "SEMANTIC_ROUTER_TEST_MODELS_DIR"
	if os.Getenv(probe) == "1" {
		requireLegacyTestModelsDir(t)
		return
	}
	executable, executableErr := os.Executable()
	if executableErr != nil {
		t.Fatal(executableErr)
	}
	root := t.TempDir()
	packageDir := filepath.Join(root, "src", "semantic-router", "pkg", "classification")
	for _, path := range []string{packageDir, filepath.Join(root, "models")} {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		name, path, message string
		wantFailure         bool
	}{
		{"ambient cache without opt in", "", "--- SKIP:", false},
		{"explicit directory", filepath.Join(root, "models"), "--- PASS:", false},
		{"explicit missing directory", filepath.Join(root, "missing"), "explicit legacy model directory", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.Command(executable, "-test.run=^TestLegacyModelArtifactRequirements$", "-test.v")
			cmd.Dir = packageDir
			cmd.Env = append(os.Environ(), probe+"=1", override+"="+tc.path)
			output, err := cmd.CombinedOutput()
			if (err != nil) != tc.wantFailure || !strings.Contains(string(output), tc.message) {
				t.Fatalf("unexpected legacy requirement outcome: err=%v output=%s", err, output)
			}
		})
	}
}
