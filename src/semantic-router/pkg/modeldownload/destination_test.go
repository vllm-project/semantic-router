package modeldownload

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildModelSpecsRejectsEscapingDownloadPath(t *testing.T) {
	escaping := "models/../config"
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{escaping: "example/public-model"},
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{ModelID: escaping},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "domain-route",
				Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"},
			}},
		},
	}

	if _, err := BuildModelSpecs(cfg); err == nil {
		t.Fatal("BuildModelSpecs accepted a path that escapes models/")
	}
}

func TestValidateModelDownloadDestinationRejectsSymlinkEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation is not reliably available on Windows")
	}
	root := t.TempDir()
	t.Chdir(root)
	if err := os.Mkdir("models", 0o755); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join("models", "escape")); err != nil {
		t.Fatal(err)
	}

	if err := validateModelDownloadDestination("models/escape/payload"); err == nil {
		t.Fatal("symlink escape was accepted")
	}
}

func TestValidateModelDownloadDestinationRejectsSymlinkedRoot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation is not reliably available on Windows")
	}
	root := t.TempDir()
	t.Chdir(root)
	if err := os.Symlink(t.TempDir(), filepath.Join(root, "models")); err != nil {
		t.Fatal(err)
	}

	if err := validateModelDownloadDestination("models/org/model"); err == nil {
		t.Fatal("symlinked models root was accepted")
	}
}

func TestValidateModelDownloadDestinationRejectsNonDirectoryRoot(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	if err := os.WriteFile(filepath.Join(root, "models"), []byte("not a directory"), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := validateModelDownloadDestination("models/org/model"); err == nil {
		t.Fatal("non-directory models root was accepted")
	}
}

func TestBuildModelSpecsRejectsSymlinkedDownloadRoot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation is not reliably available on Windows")
	}
	root := t.TempDir()
	t.Chdir(root)
	if err := os.Symlink(t.TempDir(), filepath.Join(root, "models")); err != nil {
		t.Fatal(err)
	}

	modelPath := "models/example-model"
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{modelPath: "example/public-model"},
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{ModelID: modelPath},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "domain-route",
				Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"},
			}},
		},
	}

	if _, err := BuildModelSpecs(cfg); err == nil {
		t.Fatal("model inventory accepted a symlinked models root")
	}
}

func TestValidateModelDownloadDestinationAllowsMissingContainedPath(t *testing.T) {
	t.Chdir(t.TempDir())
	if err := validateModelDownloadDestination("models/org/model"); err != nil {
		t.Fatalf("contained missing destination rejected: %v", err)
	}
}

func TestMissingTokenDoesNotMaskOrdinaryDownloadFailure(t *testing.T) {
	err := errors.New("exit status 1")
	if IsGatedModelError(err, "example/public-model", "") {
		t.Fatal("missing token incorrectly turned an ordinary failure into a gated skip")
	}
}

func TestEnsureModelsWithoutTokenFailsClosedOnDownloaderFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test helper uses a POSIX shell script")
	}
	t.Chdir(t.TempDir())
	helper := filepath.Join(t.TempDir(), "hf-fail")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	previousCommand := hfCommand
	hfCommand = helper
	t.Cleanup(func() { hfCommand = previousCommand })

	err := EnsureModels([]ModelSpec{{
		LocalPath:     "models/example/public-model",
		RepoID:        "example/public-model",
		RequiredFiles: []string{"config.json"},
	}}, DownloadConfig{})
	if err == nil {
		t.Fatal("EnsureModels reported success after an unauthenticated ordinary download failure")
	}
	if errors.Is(err, ErrGatedModelSkipped) {
		t.Fatalf("ordinary download failure was misclassified as gated: %v", err)
	}
}

func TestDownloadModelRejectsEscapingPathBeforeStartingCLI(t *testing.T) {
	if err := DownloadModelWithProgress(ModelSpec{
		LocalPath: "models/../../tmp/payload",
		RepoID:    "example/public-model",
	}, DownloadConfig{}); err == nil {
		t.Fatal("DownloadModelWithProgress accepted an escaping destination")
	}
}
