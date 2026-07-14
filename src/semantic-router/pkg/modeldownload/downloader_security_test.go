package modeldownload

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestIsGatedModelErrorRequiresNoTokenAndUnambiguousEvidence(t *testing.T) {
	tests := []struct {
		name   string
		err    error
		repoID string
		token  string
		want   bool
	}{
		{name: "known gated without token", err: errors.New("exit status 1"), repoID: "google/embeddinggemma-300m", want: true},
		{name: "known gated with token", err: errors.New("exit status 1"), repoID: "google/embeddinggemma-300m", token: "configured", want: false},
		{name: "401 without token", err: errors.New("401 Client Error: Unauthorized"), repoID: "owner/model", want: true},
		{name: "403 without token", err: errors.New("HTTP 403 Forbidden"), repoID: "owner/model", want: true},
		{name: "404 without token", err: errors.New("404 repository not found"), repoID: "owner/model", want: false},
		{name: "repository not found without token", err: errors.New("repository not found"), repoID: "owner/model", want: false},
		{name: "name merely contains gemma", err: errors.New("exit status 1"), repoID: "owner/not-really-gemma", want: false},
		{name: "auth error with configured token", err: errors.New("401 Unauthorized"), repoID: "owner/model", token: "configured", want: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := IsGatedModelError(test.err, test.repoID, test.token); got != test.want {
				t.Fatalf("IsGatedModelError() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestKnownGatedModelWithoutTokenRemainsPendingAndNonReady(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test helper uses a POSIX shell script")
	}
	t.Chdir(t.TempDir())
	installFailingHFCommand(t)

	modelPath := "models/google/embeddinggemma-300m"
	states := make([]ProgressState, 0, 4)
	err := EnsureModelsWithProgress([]ModelSpec{{
		LocalPath:     modelPath,
		RepoID:        "google/embeddinggemma-300m",
		Revision:      "main",
		RequiredFiles: []string{"config.json"},
	}}, DownloadConfig{}, func(state ProgressState) {
		states = append(states, state)
	})
	assertGatedModelUnavailable(t, err, modelPath)
	assertGatedProgressNeverReady(t, states, modelPath)
}

func installFailingHFCommand(t *testing.T) {
	t.Helper()
	helper := filepath.Join(t.TempDir(), "hf-gated-fail")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	previousCommand := hfCommand
	hfCommand = helper
	t.Cleanup(func() { hfCommand = previousCommand })
}

func assertGatedModelUnavailable(t *testing.T, err error, modelPath string) {
	t.Helper()
	if err == nil {
		t.Fatal("required gated model incorrectly produced a ready result")
	}
	if !errors.Is(err, ErrGatedModelSkipped) {
		t.Fatalf("expected gated-model cause, got %v", err)
	}
	var unavailable *RequiredModelsUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("expected typed non-ready outcome, got %T: %v", err, err)
	}
	if !slices.Equal(unavailable.PendingModels, []string{modelPath}) ||
		!slices.Equal(unavailable.GatedModels, []string{modelPath}) ||
		len(unavailable.FailedModels) != 0 {
		t.Fatalf("unexpected unavailable model details: %#v", unavailable)
	}
}

func assertGatedProgressNeverReady(t *testing.T, states []ProgressState, modelPath string) {
	t.Helper()
	if len(states) == 0 {
		t.Fatal("expected progress reports")
	}
	for _, state := range states {
		if state.ReadyModels != 0 {
			t.Fatalf("gated model increased ready count: %#v", state)
		}
		if state.Phase == "completed" {
			t.Fatalf("gated model emitted completed progress: %#v", state)
		}
		if strings.Contains(state.Message, "All required router models are ready") {
			t.Fatalf("gated model emitted a false readiness message: %#v", state)
		}
	}
	last := states[len(states)-1]
	if last.Phase != "failed" || !slices.Equal(last.PendingModels, []string{modelPath}) {
		t.Fatalf("final progress did not preserve pending model: %#v", last)
	}
}

func TestBuildModelSpecsRejectsHostileRepoIDWithoutRawControlInError(t *testing.T) {
	modelPath := "models/example-model"
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{modelPath: "--token=stolen\nforged-log"},
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             modelPath,
					CategoryMappingPath: modelPath + "/category_mapping.json",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "domain-route",
				Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"},
			}},
		},
	}

	_, err := BuildModelSpecs(cfg)
	if err == nil {
		t.Fatal("model inventory accepted a hostile Hugging Face repo ID")
	}
	if strings.Contains(err.Error(), "\n") {
		t.Fatalf("validation error contains a raw log-forging newline: %q", err.Error())
	}
}

func TestDownloadModelRejectsHostileSourceBeforeCLIAndLogging(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test helper uses a POSIX shell script")
	}
	t.Chdir(t.TempDir())
	marker := filepath.Join(t.TempDir(), "invoked")
	helper := filepath.Join(t.TempDir(), "hf-marker")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\n: > \"$MODEL_DOWNLOAD_TEST_MARKER\"\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MODEL_DOWNLOAD_TEST_MARKER", marker)
	previousCommand := hfCommand
	hfCommand = helper
	t.Cleanup(func() { hfCommand = previousCommand })

	core, observed := observer.New(zapcore.DebugLevel)
	previousLogger := zap.L()
	zap.ReplaceGlobals(zap.New(core))
	t.Cleanup(func() { zap.ReplaceGlobals(previousLogger) })

	tests := []ModelSpec{
		{LocalPath: "models/example", RepoID: "--token=stolen", Revision: "main"},
		{LocalPath: "models/example", RepoID: "owner/model\nforged-log", Revision: "main"},
		{LocalPath: "models/example", RepoID: "owner/model", Revision: "--token=stolen"},
		{LocalPath: "models/example", RepoID: "owner/model", Revision: "main\nforged-log"},
	}
	for _, spec := range tests {
		err := DownloadModelWithProgress(spec, DownloadConfig{})
		if err == nil {
			t.Fatalf("hostile source was accepted: %#v", spec)
		}
		if strings.Contains(err.Error(), "\n") {
			t.Fatalf("validation error contains a raw log-forging newline: %q", err.Error())
		}
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("Hugging Face CLI was invoked for a rejected source: %v", err)
	}
	if observed.Len() != 0 {
		t.Fatalf("rejected source reached logging seam: %#v", observed.All())
	}
}

func TestValidateHFSourceBoundsAndCanonicalForms(t *testing.T) {
	valid := []ModelSpec{
		{RepoID: "model", Revision: "main"},
		{RepoID: "owner/model_name-v1.2", Revision: "refs/pr/123"},
		{RepoID: "owner/model", Revision: strings.Repeat("a", 40)},
	}
	for _, spec := range valid {
		if err := validateModelSource(spec); err != nil {
			t.Fatalf("valid source rejected (%#v): %v", spec, err)
		}
	}

	invalid := []ModelSpec{
		{RepoID: "owner/model/extra", Revision: "main"},
		{RepoID: "owner/-model", Revision: "main"},
		{RepoID: "owner/model--copy", Revision: "main"},
		{RepoID: strings.Repeat("a", maxHFRepoIDLength+1), Revision: "main"},
		{RepoID: "owner/model", Revision: "feature//branch"},
		{RepoID: "owner/model", Revision: strings.Repeat("a", maxHFRevisionLength+1)},
	}
	for _, spec := range invalid {
		if err := validateModelSource(spec); err == nil {
			t.Fatalf("invalid source accepted: %#v", spec)
		}
	}
}
