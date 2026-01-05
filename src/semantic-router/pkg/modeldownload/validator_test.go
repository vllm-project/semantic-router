package modeldownload

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsModelComplete(t *testing.T) {
	// Create a temporary directory for testing
	tmpDir := t.TempDir()

	tests := []struct {
		name          string
		setup         func() string
		requiredFiles []string
		wantComplete  bool
		wantErr       bool
	}{
		{
			name: "model does not exist",
			setup: func() string {
				return filepath.Join(tmpDir, "nonexistent")
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  false,
			wantErr:       false,
		},
		{
			name: "model exists with all required files",
			setup: func() string {
				modelDir := filepath.Join(tmpDir, "complete-model")
				_ = os.MkdirAll(modelDir, 0o755)
				_ = os.WriteFile(filepath.Join(modelDir, "config.json"), []byte("{}"), 0o644)
				return modelDir
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  true,
			wantErr:       false,
		},
		{
			name: "model exists but missing required files",
			setup: func() string {
				modelDir := filepath.Join(tmpDir, "incomplete-model")
				_ = os.MkdirAll(modelDir, 0o755)
				// Don't create config.json
				return modelDir
			},
			requiredFiles: DefaultRequiredFiles,
			wantComplete:  false,
			wantErr:       false,
		},
		{
			name: "custom required files - all present",
			setup: func() string {
				modelDir := filepath.Join(tmpDir, "custom-model")
				_ = os.MkdirAll(modelDir, 0o755)
				_ = os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte(""), 0o644)
				_ = os.WriteFile(filepath.Join(modelDir, "tokenizer.json"), []byte(""), 0o644)
				return modelDir
			},
			requiredFiles: []string{"model.safetensors", "tokenizer.json"},
			wantComplete:  true,
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelPath := tt.setup()
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

func TestGetMissingModels(t *testing.T) {
	tmpDir := t.TempDir()

	// Create one complete model
	completeDir := filepath.Join(tmpDir, "complete")
	_ = os.MkdirAll(completeDir, 0o755)
	_ = os.WriteFile(filepath.Join(completeDir, "config.json"), []byte("{}"), 0o644)

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
