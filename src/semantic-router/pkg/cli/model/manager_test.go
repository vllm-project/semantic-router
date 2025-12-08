package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewModelManager(t *testing.T) {
	t.Run("with models dir", func(t *testing.T) {
		mgr := NewModelManager("/tmp/models")
		if mgr.ModelsDir != "/tmp/models" {
			t.Errorf("Expected ModelsDir to be /tmp/models, got %s", mgr.ModelsDir)
		}
	})

	t.Run("with empty dir defaults to ./models", func(t *testing.T) {
		mgr := NewModelManager("")
		if mgr.ModelsDir != "./models" {
			t.Errorf("Expected ModelsDir to be ./models, got %s", mgr.ModelsDir)
		}
	})
}

func TestFormatSize(t *testing.T) {
	tests := []struct {
		name     string
		bytes    int64
		expected string
	}{
		{
			name:     "bytes",
			bytes:    512,
			expected: "512 B",
		},
		{
			name:     "kilobytes",
			bytes:    1024,
			expected: "1.0 KiB",
		},
		{
			name:     "megabytes",
			bytes:    1024 * 1024,
			expected: "1.0 MiB",
		},
		{
			name:     "gigabytes",
			bytes:    1024 * 1024 * 1024,
			expected: "1.0 GiB",
		},
		{
			name:     "terabytes",
			bytes:    1024 * 1024 * 1024 * 1024,
			expected: "1.0 TiB",
		},
		{
			name:     "mixed size",
			bytes:    1536 * 1024 * 1024, // 1.5 GB
			expected: "1.5 GiB",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatSize(tt.bytes)
			if result != tt.expected {
				t.Errorf("FormatSize(%d) = %s, expected %s", tt.bytes, result, tt.expected)
			}
		})
	}
}

func TestListModels(t *testing.T) {
	t.Run("nonexistent directory", func(t *testing.T) {
		mgr := NewModelManager("/tmp/nonexistent-models-dir-12345")
		models, err := mgr.ListModels()
		// Should not error, just return empty list
		if err != nil {
			t.Errorf("Expected no error, got: %v", err)
		}

		if len(models) != 0 {
			t.Errorf("Expected empty models list, got %d models", len(models))
		}
	})

	t.Run("empty directory", func(t *testing.T) {
		// Create a temporary empty directory
		tmpDir := filepath.Join(os.TempDir(), "vsr-test-models-empty")
		_ = os.MkdirAll(tmpDir, 0o755)
		defer os.RemoveAll(tmpDir)

		mgr := NewModelManager(tmpDir)
		models, err := mgr.ListModels()
		if err != nil {
			t.Errorf("Expected no error, got: %v", err)
		}

		// Empty directory should return empty list (no models discovered)
		if len(models) != 0 {
			t.Errorf("Expected empty models list, got %d models", len(models))
		}
	})
}

func TestGetModelInfo(t *testing.T) {
	t.Run("model not found", func(t *testing.T) {
		mgr := NewModelManager("/tmp/nonexistent-models-dir-12345")
		_, err := mgr.GetModelInfo("nonexistent-model")

		if err == nil {
			t.Error("Expected error for nonexistent model, got nil")
		}
	})
}

func TestValidateModel(t *testing.T) {
	t.Run("nonexistent model", func(t *testing.T) {
		mgr := NewModelManager("/tmp/nonexistent-models-dir-12345")
		err := mgr.ValidateModel("nonexistent-model")

		if err == nil {
			t.Error("Expected error for nonexistent model, got nil")
		}
	})
}

func TestRemoveModel(t *testing.T) {
	t.Run("nonexistent model", func(t *testing.T) {
		mgr := NewModelManager("/tmp/nonexistent-models-dir-12345")
		err := mgr.RemoveModel("nonexistent-model")

		if err == nil {
			t.Error("Expected error for nonexistent model, got nil")
		}
	})
}

func TestGetDirectorySize(t *testing.T) {
	// Create a temporary directory with some files
	tmpDir := filepath.Join(os.TempDir(), "vsr-test-size")
	_ = os.MkdirAll(tmpDir, 0o755)
	defer os.RemoveAll(tmpDir)

	// Create test files
	testFile1 := filepath.Join(tmpDir, "file1.txt")
	testFile2 := filepath.Join(tmpDir, "file2.txt")

	_ = os.WriteFile(testFile1, []byte("hello"), 0o644)  // 5 bytes
	_ = os.WriteFile(testFile2, []byte("world!"), 0o644) // 6 bytes

	mgr := NewModelManager(tmpDir)
	size := mgr.getDirectorySize(tmpDir)

	// Should be 11 bytes total
	if size != 11 {
		t.Errorf("Expected size to be 11 bytes, got %d", size)
	}
}

func TestGetModelPath(t *testing.T) {
	mgr := NewModelManager("/tmp/models")

	tests := []struct {
		name     string
		modelID  string
		expected string
	}{
		{
			name:     "simple model id",
			modelID:  "test-model",
			expected: "/tmp/models/test_model",
		},
		{
			name:     "complex model id",
			modelID:  "lora-intent-classifier",
			expected: "/tmp/models/lora_intent_classifier",
		},
		{
			name:     "no dashes",
			modelID:  "simplemodel",
			expected: "/tmp/models/simplemodel",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mgr.getModelPath(tt.modelID)
			if result != tt.expected {
				t.Errorf("getModelPath(%s) = %s, expected %s", tt.modelID, result, tt.expected)
			}
		})
	}
}

func TestIsModelDownloaded(t *testing.T) {
	mgr := NewModelManager("/tmp/nonexistent-models-dir-12345")

	// Should return false for nonexistent model
	if mgr.isModelDownloaded("test-model") {
		t.Error("Expected false for nonexistent model, got true")
	}
}

func TestValidateAllModels(t *testing.T) {
	t.Run("empty models dir", func(t *testing.T) {
		tmpDir := filepath.Join(os.TempDir(), "vsr-test-validate-all")
		_ = os.MkdirAll(tmpDir, 0o755)
		defer os.RemoveAll(tmpDir)

		mgr := NewModelManager(tmpDir)
		results, err := mgr.ValidateAllModels()
		if err != nil {
			t.Errorf("Expected no error, got: %v", err)
		}

		// No models to validate
		if len(results) != 0 {
			t.Errorf("Expected 0 results, got %d", len(results))
		}
	})
}

func TestGetModelStatus(t *testing.T) {
	t.Run("returns status map", func(t *testing.T) {
		mgr := NewModelManager("/tmp/nonexistent-models-dir-12345")
		status := mgr.GetModelStatus()

		// Should return a map with at least some keys
		if status == nil {
			t.Error("Expected non-nil status map")
		}

		// Check for expected keys
		if _, hasDir := status["models_directory"]; !hasDir {
			t.Error("Expected 'models_directory' key in status")
		}

		if _, hasStatus := status["discovery_status"]; !hasStatus {
			t.Error("Expected 'discovery_status' key in status")
		}
	})
}

func TestEnsureModelsDirectory(t *testing.T) {
	t.Run("creates directory if not exists", func(t *testing.T) {
		tmpDir := filepath.Join(os.TempDir(), "vsr-test-ensure-models")
		defer os.RemoveAll(tmpDir)

		// Ensure it doesn't exist
		os.RemoveAll(tmpDir)

		mgr := NewModelManager(tmpDir)
		err := mgr.ensureModelsDirectory()
		if err != nil {
			t.Errorf("Expected no error, got: %v", err)
		}

		// Check directory was created
		if _, err := os.Stat(tmpDir); os.IsNotExist(err) {
			t.Error("Directory was not created")
		}
	})

	t.Run("no error if directory exists", func(t *testing.T) {
		tmpDir := filepath.Join(os.TempDir(), "vsr-test-ensure-models-2")
		_ = os.MkdirAll(tmpDir, 0o755)
		defer os.RemoveAll(tmpDir)

		mgr := NewModelManager(tmpDir)
		err := mgr.ensureModelsDirectory()
		if err != nil {
			t.Errorf("Expected no error for existing directory, got: %v", err)
		}
	})
}
