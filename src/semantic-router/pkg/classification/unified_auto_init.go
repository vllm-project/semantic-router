package classification

import "fmt"

// AutoInitializeUnifiedClassifier attempts to auto-discover and initialize the unified classifier.
// It prioritizes LoRA models over legacy ModernBERT models.
func AutoInitializeUnifiedClassifier(modelsDir string) (*UnifiedClassifier, error) {
	return AutoInitializeUnifiedClassifierWithRegistry(modelsDir, nil)
}

// AutoInitializeUnifiedClassifierWithRegistry auto-discovers and initializes with mom_registry.
func AutoInitializeUnifiedClassifierWithRegistry(modelsDir string, modelRegistry map[string]string) (*UnifiedClassifier, error) {
	paths, err := AutoDiscoverModelsWithRegistry(modelsDir, modelRegistry)
	if err != nil {
		return nil, fmt.Errorf("model discovery failed: %w", err)
	}

	if err := ValidateModelPaths(paths); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}

	if paths.PreferLoRA() {
		return initializeLoRAUnifiedClassifier(paths)
	}
	return initializeLegacyUnifiedClassifier(paths)
}

func initializeLoRAUnifiedClassifier(paths *ModelPaths) (*UnifiedClassifier, error) {
	classifier := &UnifiedClassifier{
		initialized: false,
		useLoRA:     true,
	}
	classifier.loraModelPaths = &LoRAModelPaths{
		IntentPath:   paths.LoRAIntentClassifier,
		PIIPath:      paths.LoRAPIIClassifier,
		SecurityPath: paths.LoRASecurityClassifier,
		Architecture: paths.LoRAArchitecture,
	}
	classifier.initialized = true

	if err := classifier.initializeLoRABindings(); err != nil {
		return nil, fmt.Errorf("failed to pre-initialize LoRA bindings: %w", err)
	}
	classifier.loraInitialized = true

	return classifier, nil
}

func initializeLegacyUnifiedClassifier(paths *ModelPaths) (*UnifiedClassifier, error) {
	labels, err := loadLegacyUnifiedLabels(paths)
	if err != nil {
		return nil, err
	}

	classifier := GetGlobalUnifiedClassifier()
	err = classifier.Initialize(
		paths.ModernBertBase,
		paths.IntentClassifier,
		paths.PIIClassifier,
		paths.SecurityClassifier,
		labels.intent,
		labels.pii,
		labels.security,
		false,
	)
	if err != nil {
		return nil, fmt.Errorf("unified classifier initialization failed: %w", err)
	}

	return classifier, nil
}
