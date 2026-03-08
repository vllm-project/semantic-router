package classification

import (
	"os"
	"path/filepath"
	"testing"
)

func createMockModelFile(t *testing.T, dir, filename string) {
	t.Helper()

	filePath := filepath.Join(dir, filename)
	file, err := os.Create(filePath)
	if err != nil {
		t.Fatalf("Failed to create mock file %s: %v", filePath, err)
	}
	defer file.Close()

	_, _ = file.WriteString(`{"mock": "model file"}`)
}

func createMockModelFileForBench(b *testing.B, dir, filename string) {
	b.Helper()

	filePath := filepath.Join(dir, filename)
	file, err := os.Create(filePath)
	if err != nil {
		b.Fatalf("Failed to create mock file %s: %v", filePath, err)
	}
	defer file.Close()

	_, _ = file.WriteString(`{"mock": "model file"}`)
}

func createMockModelTree(t testing.TB, tempDir string) {
	t.Helper()

	modernbertDir := filepath.Join(tempDir, "modernbert-base")
	intentDir := filepath.Join(tempDir, "category_classifier_modernbert-base_model")
	piiDir := filepath.Join(tempDir, "pii_classifier_modernbert-base_presidio_token_model")
	securityDir := filepath.Join(tempDir, "jailbreak_classifier_modernbert-base_model")

	_ = os.MkdirAll(modernbertDir, 0o755)
	_ = os.MkdirAll(intentDir, 0o755)
	_ = os.MkdirAll(piiDir, 0o755)
	_ = os.MkdirAll(securityDir, 0o755)

	switch tb := t.(type) {
	case *testing.T:
		createMockModelFile(tb, modernbertDir, "config.json")
		createMockModelFile(tb, intentDir, "pytorch_model.bin")
		createMockModelFile(tb, piiDir, "model.safetensors")
		createMockModelFile(tb, securityDir, "config.json")
	case *testing.B:
		createMockModelFileForBench(tb, modernbertDir, "config.json")
		createMockModelFileForBench(tb, intentDir, "pytorch_model.bin")
		createMockModelFileForBench(tb, piiDir, "model.safetensors")
		createMockModelFileForBench(tb, securityDir, "config.json")
	}
}

func TestAutoDiscoverModels(t *testing.T) {
	tempDir := t.TempDir()
	createMockModelTree(t, tempDir)

	tests := []struct {
		name      string
		modelsDir string
		wantErr   bool
		checkFunc func(*ModelPaths) bool
	}{
		{
			name:      "successful discovery",
			modelsDir: tempDir,
			checkFunc: func(paths *ModelPaths) bool { return paths.IsComplete() },
		},
		{
			name:      "nonexistent directory",
			modelsDir: "/nonexistent/path",
			wantErr:   true,
		},
		{
			name:      "empty directory",
			modelsDir: t.TempDir(),
			checkFunc: func(paths *ModelPaths) bool { return !paths.IsComplete() },
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
	tempDir := t.TempDir()

	modernbertDir := filepath.Join(tempDir, "modernbert-base")
	intentDir := filepath.Join(tempDir, "intent")
	piiDir := filepath.Join(tempDir, "pii")
	securityDir := filepath.Join(tempDir, "security")

	_ = os.MkdirAll(modernbertDir, 0o755)
	_ = os.MkdirAll(intentDir, 0o755)
	_ = os.MkdirAll(piiDir, 0o755)
	_ = os.MkdirAll(securityDir, 0o755)

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
		},
		{name: "nil paths", paths: nil, wantErr: true},
		{
			name: "missing modernbert",
			paths: &ModelPaths{
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
	tempDir := t.TempDir()
	modernbertDir := filepath.Join(tempDir, "modernbert-base")

	_ = os.MkdirAll(modernbertDir, 0o755)
	createMockModelFile(t, modernbertDir, "config.json")

	info := GetModelDiscoveryInfo(tempDir)

	if info["models_directory"] != tempDir {
		t.Errorf("Expected models_directory to be %s, got %v", tempDir, info["models_directory"])
	}
	if _, ok := info["discovered_models"]; !ok {
		t.Error("Expected discovered_models field")
	}
	if _, ok := info["missing_models"]; !ok {
		t.Error("Expected missing_models field")
	}
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
				IntentClassifier:   "/path/to/intent",
				PIIClassifier:      "/path/to/pii",
				SecurityClassifier: "/path/to/security",
			},
		},
		{name: "missing all", paths: &ModelPaths{}, expected: false},
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

func TestAutoDiscoverModels_RealModels(t *testing.T) {
	modelsDir := getTestModelsDir()
	testModelsDirExists(t, modelsDir)

	paths, err := AutoDiscoverModels(modelsDir)
	if err != nil {
		t.Fatalf("AutoDiscoverModels() failed: %v (models directory should exist at %s)", err, modelsDir)
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

	if paths.IntentClassifier == "" || paths.PIIClassifier == "" || paths.SecurityClassifier == "" {
		t.Logf("One or more required models not found (intent=%q, pii=%q, Jailbreak=%q)", paths.IntentClassifier, paths.PIIClassifier, paths.SecurityClassifier)
		t.Skip("Skipping real-models discovery assertions because required models are not present")
	}
	if paths.ModernBertBase == "" {
		t.Error("ModernBERT base model not found - auto-discovery logic failed")
	} else {
		t.Logf("ModernBERT base found at: %s", paths.ModernBertBase)
	}

	err = ValidateModelPaths(paths)
	if err != nil {
		t.Logf("ValidateModelPaths() failed in real-models test: %v", err)
		t.Skip("Skipping real-models validation because environment lacks complete models")
	}
	if !paths.IsComplete() {
		t.Error("Model paths are not complete")
	}
}

func TestAutoInitializeUnifiedClassifier(t *testing.T) {
	modelsDir := getTestModelsDir()
	testModelsDirExists(t, modelsDir)

	classifier, err := AutoInitializeUnifiedClassifier(modelsDir)
	if err != nil {
		t.Skipf("Skipping test: AutoInitializeUnifiedClassifier() failed: %v (models directory: %s)", err, modelsDir)
	}
	if classifier == nil {
		t.Skip("Skipping test: AutoInitializeUnifiedClassifier() returned nil classifier (models not available)")
	}

	t.Logf("Unified classifier initialized successfully")
	t.Logf("  Use LoRA: %v", classifier.useLoRA)
	t.Logf("  Initialized: %v", classifier.initialized)

	if !classifier.useLoRA {
		t.Log("Using legacy ModernBERT models")
		return
	}

	t.Log("Using high-confidence LoRA models")
	if classifier.loraModelPaths == nil {
		t.Error("LoRA model paths should not be nil when useLoRA is true")
		return
	}

	t.Logf("  LoRA Intent Path: %s", classifier.loraModelPaths.IntentPath)
	t.Logf("  LoRA PII Path: %s", classifier.loraModelPaths.PIIPath)
	t.Logf("  LoRA Security Path: %s", classifier.loraModelPaths.SecurityPath)
	t.Logf("  LoRA Architecture: %s", classifier.loraModelPaths.Architecture)
}

func BenchmarkAutoDiscoverModels(b *testing.B) {
	tempDir := b.TempDir()
	createMockModelTree(b, tempDir)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = AutoDiscoverModels(tempDir)
	}
}
