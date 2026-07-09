package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeManifest(t *testing.T, dir, body string) {
	t.Helper()
	onnxDir := filepath.Join(dir, "onnx")
	if err := os.MkdirAll(onnxDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(onnxDir, "model_config.json"), []byte(body), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
}

// The early-exit layers a request may target must come from the model's own
// onnx/model_config.json (available_layers), the single source of truth, so
// they never drift from what the shipped model actually provides. The official
// mmbert-embed-32k-2d-matryoshka ships [6, 11, 16, 22].
func TestMmBertAvailableLayers_ReadsManifest(t *testing.T) {
	dir := t.TempDir()
	writeManifest(t, dir, `{"total_layers": 22, "available_layers": [6, 11, 16, 22]}`)

	got := MmBertAvailableLayers(dir)

	want := []int{6, 11, 16, 22}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("MmBertAvailableLayers = %v, want %v", got, want)
	}
}

// When no manifest is present, fall back to the historical default layer set
// rather than rejecting every layer, preserving backward compatibility with
// models that ship without a model_config.json.
func TestMmBertAvailableLayers_FallbackWithoutManifest(t *testing.T) {
	dir := t.TempDir()

	got := MmBertAvailableLayers(dir)

	want := []int{3, 6, 11, 22}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("MmBertAvailableLayers fallback = %v, want %v", got, want)
	}
}

func TestIsValidMmBertLayer(t *testing.T) {
	available := []int{6, 11, 16, 22}
	if !IsValidMmBertLayer(16, available) {
		t.Errorf("layer 16 should be valid for %v", available)
	}
	if IsValidMmBertLayer(3, available) {
		t.Errorf("layer 3 should be invalid for %v (phantom layer)", available)
	}
}
