package modeldownload

import (
	"os"
	"path/filepath"
	"testing"
)

type isModelCompleteCase struct {
	name          string
	setup         func(t *testing.T, tmpDir string) string
	requiredFiles []string
	wantComplete  bool
	wantErr       bool
}

func TestIsModelComplete(t *testing.T) {
	runIsModelCompleteCases(t, []isModelCompleteCase{
		{
			name: "model does not exist",
			setup: func(_ *testing.T, tmpDir string) string {
				return filepath.Join(tmpDir, "nonexistent")
			},
			requiredFiles: DefaultRequiredFiles,
		},
		{
			name: "model exists with all required files",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "complete-model", map[string]string{
					"config.json":       "{}",
					"model.safetensors": "",
				})
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  true,
		},
		{
			name: "model metadata without weights is incomplete",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "metadata-only-model", map[string]string{
					"config.json": "{}",
				})
			},
			requiredFiles: DefaultRequiredFiles,
		},
		{
			name: "model exists but missing required files",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "incomplete-model", nil)
			},
			requiredFiles: DefaultRequiredFiles,
		},
		{
			name: "custom required files - all present",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "custom-model", map[string]string{
					"model.safetensors": "",
					"tokenizer.json":    "",
				})
			},
			requiredFiles: []string{"model.safetensors", "tokenizer.json"},
			wantComplete:  true,
		},
	})
}

func TestIsModelCompleteRecognizesWeightArtifacts(t *testing.T) {
	runIsModelCompleteCases(t, []isModelCompleteCase{
		{
			name: "sharded safetensors index counts as model weights",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "sharded-model", map[string]string{
					"config.json":                  "{}",
					"model.safetensors.index.json": "{}",
				})
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  true,
		},
		{
			name: "sharded pytorch bin counts as model weights",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "sharded-pytorch-model", map[string]string{
					"config.json":                  "{}",
					"pytorch_model-00001-of-2.bin": "",
				})
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  true,
		},
		{
			name: "adapter bin counts as model weights",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "adapter-bin-model", map[string]string{
					"config.json":       "{}",
					"adapter_model.bin": "",
				})
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  true,
		},
		{
			name: "torch pt artifact counts as model weights",
			setup: func(t *testing.T, tmpDir string) string {
				return writeModelDir(t, tmpDir, "torch-pt-model", map[string]string{
					"config.json": "{}",
					"model.pt":    "",
				})
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  true,
		},
	})
}

func runIsModelCompleteCases(t *testing.T, tests []isModelCompleteCase) {
	t.Helper()
	tmpDir := t.TempDir()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelPath := tt.setup(t, tmpDir)
			complete, err := IsModelComplete(modelPath, tt.requiredFiles)

			if (err != nil) != tt.wantErr {
				t.Errorf("IsModelComplete() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if complete != tt.wantComplete {
				t.Errorf("IsModelComplete() = %v, want %v", complete, tt.wantComplete)
			}
		})
	}
}

func writeModelDir(t *testing.T, tmpDir, modelName string, files map[string]string) string {
	t.Helper()
	modelDir := filepath.Join(tmpDir, modelName)
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", modelDir, err)
	}
	for name, contents := range files {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte(contents), 0o644); err != nil {
			t.Fatalf("WriteFile(%q) failed: %v", name, err)
		}
	}
	return modelDir
}

func TestGetMissingModels(t *testing.T) {
	tmpDir := t.TempDir()

	// Create one complete model
	completeDir := filepath.Join(tmpDir, "complete")
	_ = os.MkdirAll(completeDir, 0o755)
	_ = os.WriteFile(filepath.Join(completeDir, "config.json"), []byte("{}"), 0o644)
	_ = os.WriteFile(filepath.Join(completeDir, "model.safetensors"), []byte(""), 0o644)

	// Create one incomplete model
	incompleteDir := filepath.Join(tmpDir, "incomplete")
	_ = os.MkdirAll(incompleteDir, 0o755)

	specs := []ModelSpec{
		{
			LocalPath:     completeDir,
			RepoID:        "test/complete",
			RequiredFiles: DefaultRequiredFiles,
		},
		{
			LocalPath:     incompleteDir,
			RepoID:        "test/incomplete",
			RequiredFiles: DefaultRequiredFiles,
		},
		{
			LocalPath:     filepath.Join(tmpDir, "nonexistent"),
			RepoID:        "test/nonexistent",
			RequiredFiles: DefaultRequiredFiles,
		},
	}

	missing, err := GetMissingModels(specs)
	if err != nil {
		t.Fatalf("GetMissingModels() error = %v", err)
	}

	if len(missing) != 2 {
		t.Errorf("GetMissingModels() returned %d missing models, want 2", len(missing))
	}

	// Verify the missing models are the expected ones
	missingRepoIDs := make(map[string]bool)
	for _, spec := range missing {
		missingRepoIDs[spec.RepoID] = true
	}

	if !missingRepoIDs["test/incomplete"] {
		t.Error("Expected test/incomplete to be missing")
	}
	if !missingRepoIDs["test/nonexistent"] {
		t.Error("Expected test/nonexistent to be missing")
	}
	if missingRepoIDs["test/complete"] {
		t.Error("test/complete should not be missing")
	}
}
