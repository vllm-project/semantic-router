package classification

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAutoDiscoverModels(t *testing.T) {
	// Create temporary directory structure for testing
	tempDir := t.TempDir()

	// Create mock model directories
	modernbertDir := filepath.Join(tempDir, "modernbert-base")
	intentDir := filepath.Join(tempDir, "category_classifier_modernbert-base_model")
	piiDir := filepath.Join(tempDir, "pii_classifier_modernbert-base_presidio_token_model")
	securityDir := filepath.Join(tempDir, "jailbreak_classifier_modernbert-base_model")

	// Create directories
	_ = os.MkdirAll(modernbertDir, 0o755)
	_ = os.MkdirAll(intentDir, 0o755)
	_ = os.MkdirAll(piiDir, 0o755)
	_ = os.MkdirAll(securityDir, 0o755)

	// Create mock model files
	createMockModelFile(t, modernbertDir, "config.json")
	createMockModelFile(t, intentDir, "pytorch_model.bin")
	createMockModelFile(t, piiDir, "model.safetensors")
	createMockModelFile(t, securityDir, "config.json")

	tests := []struct {
		name      string
		modelsDir string
		wantErr   bool
		checkFunc func(*ModelPaths) bool
	}{
		{
			name:      "successful discovery",
			modelsDir: tempDir,
			wantErr:   false,
			checkFunc: func(mp *ModelPaths) bool {
				return mp.IsComplete()
			},
		},
		{
			name:      "nonexistent directory",
			modelsDir: "/nonexistent/path",
			wantErr:   true,
			checkFunc: nil,
		},
		{
			name:      "empty directory",
			modelsDir: t.TempDir(), // Empty temp dir
			wantErr:   false,
			checkFunc: func(mp *ModelPaths) bool {
				return !mp.IsComplete() // Should not be complete
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			paths, err := AutoDiscoverModels(tt.modelsDir)

			if (err != nil) != tt.wantErr {
				t.Errorf("AutoDiscoverModels() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.checkFunc != nil && !tt.checkFunc(paths) {
				t.Errorf("AutoDiscoverModels() check function failed for paths: %+v", paths)
			}
		})
	}
}

func TestValidateModelPaths(t *testing.T) {
	// Create temporary directory with valid model structure
	tempDir := t.TempDir()

	modernbertDir := filepath.Join(tempDir, "modernbert-base")
	intentDir := filepath.Join(tempDir, "intent")
	piiDir := filepath.Join(tempDir, "pii")
	securityDir := filepath.Join(tempDir, "security")

	_ = os.MkdirAll(modernbertDir, 0o755)
	_ = os.MkdirAll(intentDir, 0o755)
	_ = os.MkdirAll(piiDir, 0o755)
	_ = os.MkdirAll(securityDir, 0o755)

	// Create model files
	createMockModelFile(t, modernbertDir, "config.json")
	createMockModelFile(t, intentDir, "pytorch_model.bin")
	createMockModelFile(t, piiDir, "model.safetensors")
	createMockModelFile(t, securityDir, "tokenizer.json")

	tests := []struct {
		name    string
		paths   *ModelPaths
		wantErr bool
	}{
		{
			name: "valid paths",
			paths: &ModelPaths{
				ModernBertBase:     modernbertDir,
				IntentClassifier:   intentDir,
				PIIClassifier:      piiDir,
				SecurityClassifier: securityDir,
			},
			wantErr: false,
		},
		{
			name:    "nil paths",
			paths:   nil,
			wantErr: true,
		},
		{
			name: "missing modernbert",
			paths: &ModelPaths{
				ModernBertBase:     "",
				IntentClassifier:   intentDir,
				PIIClassifier:      piiDir,
				SecurityClassifier: securityDir,
			},
			wantErr: true,
		},
		{
			name: "nonexistent path",
			paths: &ModelPaths{
				ModernBertBase:     "/nonexistent/path",
				IntentClassifier:   intentDir,
				PIIClassifier:      piiDir,
				SecurityClassifier: securityDir,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateModelPaths(tt.paths)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateModelPaths() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetModelDiscoveryInfo(t *testing.T) {
	// Create temporary directory with some models
	tempDir := t.TempDir()

	modernbertDir := filepath.Join(tempDir, "modernbert-base")
	_ = os.MkdirAll(modernbertDir, 0o755)
	createMockModelFile(t, modernbertDir, "config.json")

	info := GetModelDiscoveryInfo(tempDir)

	// Check basic structure
	if info["models_directory"] != tempDir {
		t.Errorf("Expected models_directory to be %s, got %v", tempDir, info["models_directory"])
	}

	if _, ok := info["discovered_models"]; !ok {
		t.Error("Expected discovered_models field")
	}

	if _, ok := info["missing_models"]; !ok {
		t.Error("Expected missing_models field")
	}

	// Should have incomplete status since we only have modernbert
	if info["discovery_status"] == "complete" {
		t.Error("Expected incomplete discovery status")
	}
}

func TestModelPathsIsComplete(t *testing.T) {
	tests := []struct {
		name     string
		paths    *ModelPaths
		expected bool
	}{
		{
			name: "complete paths",
			paths: &ModelPaths{
				ModernBertBase:     "/path/to/modernbert",
				IntentClassifier:   "/path/to/intent",
				PIIClassifier:      "/path/to/pii",
				SecurityClassifier: "/path/to/security",
			},
			expected: true,
		},
		{
			name: "missing modernbert",
			paths: &ModelPaths{
				ModernBertBase:     "",
				IntentClassifier:   "/path/to/intent",
				PIIClassifier:      "/path/to/pii",
				SecurityClassifier: "/path/to/security",
			},
			expected: false,
		},
		{
			name:     "missing all",
			paths:    &ModelPaths{},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.paths.IsComplete()
			if result != tt.expected {
				t.Errorf("IsComplete() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

// Helper function to create mock model files
func createMockModelFile(t *testing.T, dir, filename string) {
	filePath := filepath.Join(dir, filename)
	file, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create mock file %s: %v", filePath, err)
	}
	defer file.Close()

	// Write some dummy content
	_, _ = file.WriteString(`{"mock": "model file"}`)
}

func TestAutoDiscoverModels_RealModels(t *testing.T) {
	// Test with real models directory
	modelsDir := "../../../../../models"

	paths, err := AutoDiscoverModels(modelsDir)
	if err != nil {
		// Skip this test in environments without the real models directory
		t.Logf("AutoDiscoverModels() failed in real-models test: %v", err)
		t.Skip("Skipping real-models discovery test because models directory is unavailable")
	}

	t.Logf("Discovered paths:")
	t.Logf("  ModernBERT Base: %s", paths.ModernBertBase)
	t.Logf("  Intent Classifier: %s", paths.IntentClassifier)
	t.Logf("  PII Classifier: %s", paths.PIIClassifier)
	t.Logf("  Security Classifier: %s", paths.SecurityClassifier)
	t.Logf("  LoRA Intent Classifier: %s", paths.LoRAIntentClassifier)
	t.Logf("  LoRA PII Classifier: %s", paths.LoRAPIIClassifier)
	t.Logf("  LoRA Security Classifier: %s", paths.LoRASecurityClassifier)
	t.Logf("  LoRA Architecture: %s", paths.LoRAArchitecture)
	t.Logf("  Has LoRA Models: %v", paths.HasLoRAModels())
	t.Logf("  Prefer LoRA: %v", paths.PreferLoRA())
	t.Logf("  Is Complete: %v", paths.IsComplete())

	// Check that we found the required models; skip if not present in this environment
	if paths.IntentClassifier == "" || paths.PIIClassifier == "" || paths.SecurityClassifier == "" {
		t.Logf("One or more required models not found (intent=%q, pii=%q, security=%q)", paths.IntentClassifier, paths.PIIClassifier, paths.SecurityClassifier)
		t.Skip("Skipping real-models discovery assertions because required models are not present")
	}

	// The key test: ModernBERT base should be found (either dedicated or from classifier)
	if paths.ModernBertBase == "" {
		t.Error("ModernBERT base model not found - auto-discovery logic failed")
	} else {
		t.Logf("✅ ModernBERT base found at: %s", paths.ModernBertBase)
	}

	// Test validation
	err = ValidateModelPaths(paths)
	if err != nil {
		t.Logf("ValidateModelPaths() failed in real-models test: %v", err)
		t.Skip("Skipping real-models validation because environment lacks complete models")
	} else {
		t.Log("✅ Model paths validation successful")
	}

	// Test if paths are complete
	if !paths.IsComplete() {
		t.Error("Model paths are not complete")
	} else {
		t.Log("✅ All required models found")
	}
}

// TestAutoInitializeUnifiedClassifier tests the full initialization process
func TestAutoInitializeUnifiedClassifier(t *testing.T) {
	// Test with real models directory
	classifier, err := AutoInitializeUnifiedClassifier("../../../../../models")
	if err != nil {
		t.Logf("AutoInitializeUnifiedClassifier() failed in real-models test: %v", err)
		t.Skip("Skipping unified classifier init test because real models are unavailable")
	}

	if classifier == nil {
		t.Fatal("AutoInitializeUnifiedClassifier() returned nil classifier")
	}

	t.Logf("✅ Unified classifier initialized successfully")
	t.Logf("  Use LoRA: %v", classifier.useLoRA)
	t.Logf("  Initialized: %v", classifier.initialized)

	if classifier.useLoRA {
		t.Log("✅ Using high-confidence LoRA models")
		if classifier.loraModelPaths == nil {
			t.Error("LoRA model paths should not be nil when useLoRA is true")
		} else {
			t.Logf("  LoRA Intent Path: %s", classifier.loraModelPaths.IntentPath)
			t.Logf("  LoRA PII Path: %s", classifier.loraModelPaths.PIIPath)
			t.Logf("  LoRA Security Path: %s", classifier.loraModelPaths.SecurityPath)
			t.Logf("  LoRA Architecture: %s", classifier.loraModelPaths.Architecture)
		}
	} else {
		t.Log("Using legacy ModernBERT models")
	}
}

func BenchmarkAutoDiscoverModels(b *testing.B) {
	// Create temporary directory with model structure
	tempDir := b.TempDir()

	modernbertDir := filepath.Join(tempDir, "modernbert-base")
	intentDir := filepath.Join(tempDir, "category_classifier_modernbert-base_model")
	piiDir := filepath.Join(tempDir, "pii_classifier_modernbert-base_presidio_token_model")
	securityDir := filepath.Join(tempDir, "jailbreak_classifier_modernbert-base_model")

	_ = os.MkdirAll(modernbertDir, 0o755)
	_ = os.MkdirAll(intentDir, 0o755)
	_ = os.MkdirAll(piiDir, 0o755)
	_ = os.MkdirAll(securityDir, 0o755)

	// Create mock files using helper
	createMockModelFileForBench(b, modernbertDir, "config.json")
	createMockModelFileForBench(b, intentDir, "pytorch_model.bin")
	createMockModelFileForBench(b, piiDir, "model.safetensors")
	createMockModelFileForBench(b, securityDir, "config.json")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = AutoDiscoverModels(tempDir)
	}
}

// Helper function for benchmark
func createMockModelFileForBench(b *testing.B, dir, filename string) {
	filePath := filepath.Join(dir, filename)
	file, err := os.Create(filePath)
	if err != nil {
		b.Fatalf("Failed to create mock file %s: %v", filePath, err)
	}
	defer file.Close()
	_, _ = file.WriteString(`{"mock": "model file"}`)
}
