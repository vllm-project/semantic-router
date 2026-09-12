//go:build !windows

package modeldownload

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func oldRevisionFixture(t *testing.T) ModelSpec {
	t.Helper()
	spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
	writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
	writeHFRevisionArtifact(t, spec, "model.safetensors", "native weights", true)
	writeHFRevisionArtifact(t, spec, "onnx/model_fa_fp16.onnx", string(onnxFixtureField(7, nil)), false)
	writeHFRevisionArtifact(t, spec, "onnx/model_sdpa_fp16.onnx.data", "old external weights", true)
	return spec
}

func TestPinnedHFDirectoryUpgradeArchivesDeletedGraphs(t *testing.T) {
	spec := oldRevisionFixture(t)
	notes := filepath.Join(spec.LocalPath, "user-notes.txt")
	if err := os.WriteFile(notes, []byte("keep my notes"), 0o600); err != nil {
		t.Fatal(err)
	}
	fresh := spec
	fresh.LocalPath = t.TempDir()
	fresh.Revision = "1123456789abcdef0123456789abcdef01234567"
	writeHFRevisionArtifact(t, fresh, "config.json", "{}", false)
	writeHFRevisionArtifact(t, fresh, "model.safetensors", "new native weights", true)
	writeHFRevisionArtifact(t, fresh, "onnx/model_fa.onnx", string(onnxFixtureField(7, onnxFixtureField(5, onnxFixtureTensor("model.onnx.data")))), false)
	writeHFRevisionArtifact(t, fresh, "onnx/model.onnx.data", "new external weights", true)
	// Exercise the downloader with the HF CLI's additive local-dir semantics.
	command := filepath.Join(t.TempDir(), "hf")
	if err := os.WriteFile(command, []byte("#!/bin/sh\ncp -R \"$VELA_TEST_SOURCE/.\" \"$VELA_TEST_TARGET/\"\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VELA_TEST_SOURCE", fresh.LocalPath)
	t.Setenv("VELA_TEST_TARGET", spec.LocalPath)
	previousCommand := hfCommand
	hfCommand = command
	t.Cleanup(func() { hfCommand = previousCommand })
	spec.Revision = fresh.Revision
	if err := DownloadModelWithProgressContext(context.Background(), spec, DownloadConfig{}); err != nil {
		t.Fatal(err)
	}
	if missing, err := GetMissingModels([]ModelSpec{spec}); err != nil || len(missing) != 0 {
		t.Fatalf("upgraded model not ready: %v %v", missing, err)
	}
	for _, name := range []string{"onnx/model_fa_fp16.onnx", "onnx/model_sdpa_fp16.onnx.data"} {
		if _, err := os.Stat(filepath.Join(spec.LocalPath, name)); !os.IsNotExist(err) {
			t.Fatalf("old graph can still be selected: %s: %v", name, err)
		}
		archived, err := filepath.Glob(filepath.Join(spec.LocalPath, ".router-superseded-*", name))
		if err != nil || len(archived) != 1 {
			t.Fatalf("old artifact was not preserved: %s: %v %v", name, archived, err)
		}
	}
	if data, err := os.ReadFile(notes); err != nil || string(data) != "keep my notes" {
		t.Fatalf("unmanaged notes changed: %q %v", data, err)
	}
}

func TestFailedRevisionUpgradePreservesFiles(t *testing.T) {
	for _, failure := range []string{"missing-current-weights", "only-excluded-weights", "only-current-lora-sidecar", "missing-external-data", "modified-old-file", "unmanaged-old-graph"} {
		t.Run(failure, func(t *testing.T) {
			spec := oldRevisionFixture(t)
			spec.Revision = "1123456789abcdef0123456789abcdef01234567"
			writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
			if failure != "missing-current-weights" && failure != "only-excluded-weights" && failure != "only-current-lora-sidecar" {
				writeHFRevisionArtifact(t, spec, "model.safetensors", "new native weights", true)
			}
			oldGraph := filepath.Join(spec.LocalPath, "onnx/model_fa_fp16.onnx")
			switch failure {
			case "only-current-lora-sidecar":
				writeHFRevisionArtifact(t, spec, "lora/adapter_model.safetensors", "optional new adapter", true)
			case "only-excluded-weights":
				spec.ExcludePatterns = []string{"*.onnx", "*.data"}
			case "missing-external-data":
				writeHFRevisionArtifact(t, spec, "onnx/model_fa.onnx", string(onnxFixtureField(7, onnxFixtureField(5, onnxFixtureTensor("missing.data")))), false)
			case "modified-old-file":
				if err := os.WriteFile(oldGraph, []byte("user edited graph"), 0o600); err != nil {
					t.Fatal(err)
				}
			case "unmanaged-old-graph":
				if err := os.Remove(filepath.Join(spec.LocalPath, ".cache/huggingface/download/onnx/model_fa_fp16.onnx.metadata")); err != nil {
					t.Fatal(err)
				}
			}
			before, err := os.ReadFile(oldGraph)
			if err != nil {
				t.Fatal(err)
			}
			if err := recordDownloadedModelRevision(spec); err == nil {
				t.Fatal("invalid revision upgrade succeeded")
			}
			if data, err := os.ReadFile(oldGraph); err != nil || string(data) != string(before) {
				t.Fatalf("old graph lost after failed upgrade: %q %v", data, err)
			}
			if data, err := os.ReadFile(filepath.Join(spec.LocalPath, "onnx/model_sdpa_fp16.onnx.data")); err != nil || string(data) != "old external weights" {
				t.Fatalf("old external weights lost: %q %v", data, err)
			}
			if _, err := os.Stat(filepath.Join(spec.LocalPath, modelRevisionReceipt)); !os.IsNotExist(err) {
				t.Fatalf("failed upgrade wrote receipt: %v", err)
			}
		})
	}
}

func TestPinnedRevisionNeedsPrimaryWeights(t *testing.T) {
	for _, name := range []string{"lora/adapter_model.safetensors", "2_Dense/model.safetensors", "onnx/model.onnx"} {
		t.Run(name, func(t *testing.T) {
			spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
			writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
			value := "optional weights"
			if filepath.Ext(name) == ".onnx" {
				value = string(onnxFixtureField(7, nil))
				spec.ExcludePatterns = []string{"*.onnx"}
			}
			writeHFRevisionArtifact(t, spec, name, value, true)
			if current, err := hasCurrentModelRevision(spec); err != nil || current {
				t.Fatalf("optional/excluded weights qualified a missing root model: %v %v", current, err)
			}
			spec.RequiredFiles = []string{"config.json", name}
			if current, err := hasCurrentModelRevision(spec); err != nil || !current {
				t.Fatalf("explicit primary artifact contract rejected: %v %v", current, err)
			}
		})
	}
}
