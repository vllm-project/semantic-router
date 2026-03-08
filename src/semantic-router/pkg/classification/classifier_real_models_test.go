package classification

import (
	"os"
	"sync"
	"testing"
)

const (
	testModelsDir    = "../../../../models"
	testModelsDirEnv = "SEMANTIC_ROUTER_TEST_MODELS_DIR"
)

func getTestModelsDir() string {
	if override := os.Getenv(testModelsDirEnv); override != "" {
		return override
	}
	return testModelsDir
}

func testModelsDirExists(t *testing.T, modelsDir string) {
	t.Helper()
	if _, err := os.Stat(modelsDir); err != nil {
		t.Skipf("Skipping: model directory not available at %s (%v)", modelsDir, err)
	}
}

func testModelsDirExistsBench(b *testing.B, modelsDir string) {
	b.Helper()
	if _, err := os.Stat(modelsDir); err != nil {
		b.Skipf("Skipping: model directory not available at %s (%v)", modelsDir, err)
	}
}

var (
	globalTestClassifier     *UnifiedClassifier
	globalTestClassifierOnce sync.Once
)

func getTestClassifier(t *testing.T) *UnifiedClassifier {
	t.Helper()

	globalTestClassifierOnce.Do(func() {
		modelsDir := getTestModelsDir()
		testModelsDirExists(t, modelsDir)

		classifier, err := AutoInitializeUnifiedClassifier(modelsDir)
		if err != nil {
			t.Logf("Failed to initialize classifier: %v", err)
			return
		}
		if classifier != nil && classifier.IsInitialized() {
			globalTestClassifier = classifier
			t.Logf("Global test classifier initialized successfully")
		}
	})

	return globalTestClassifier
}

func getBenchmarkClassifier(b *testing.B) *UnifiedClassifier {
	b.Helper()

	globalTestClassifierOnce.Do(func() {
		modelsDir := getTestModelsDir()
		testModelsDirExistsBench(b, modelsDir)

		classifier, err := AutoInitializeUnifiedClassifier(modelsDir)
		if err != nil {
			b.Logf("Failed to initialize classifier: %v", err)
			return
		}
		if classifier != nil && classifier.IsInitialized() {
			globalTestClassifier = classifier
			b.Logf("Global benchmark classifier initialized successfully")
		}
	})

	return globalTestClassifier
}
