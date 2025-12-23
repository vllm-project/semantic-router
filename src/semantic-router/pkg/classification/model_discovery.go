package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ModelPaths holds the discovered model paths
type ModelPaths struct {
	// Legacy ModernBERT models (low confidence)
	ModernBertBase     string
	IntentClassifier   string
	PIIClassifier      string
	SecurityClassifier string

	// LoRA models
	LoRAIntentClassifier   string
	LoRAPIIClassifier      string
	LoRASecurityClassifier string
	LoRAArchitecture       string // "bert", "roberta", "modernbert"
}

// IsComplete checks if all required models are found
func (mp *ModelPaths) IsComplete() bool {
	return mp.HasLoRAModels() || mp.HasLegacyModels()
}

// HasLoRAModels checks if LoRA models are available
func (mp *ModelPaths) HasLoRAModels() bool {
	return mp.LoRAIntentClassifier != "" &&
		mp.LoRAPIIClassifier != "" &&
		mp.LoRASecurityClassifier != "" &&
		mp.LoRAArchitecture != ""
}

// HasLegacyModels checks if legacy ModernBERT models are available
func (mp *ModelPaths) HasLegacyModels() bool {
	return mp.ModernBertBase != "" &&
		mp.IntentClassifier != "" &&
		mp.PIIClassifier != "" &&
		mp.SecurityClassifier != ""
}

// PreferLoRA returns true if LoRA models should be used (higher confidence)
func (mp *ModelPaths) PreferLoRA() bool {
	return mp.HasLoRAModels()
}

// ArchitectureModels holds models for a specific architecture
type ArchitectureModels struct {
	Intent   string
	PII      string
	Security string
}

// AutoDiscoverModels automatically discovers model files in the models directory
// Uses intelligent architecture selection: BERT > RoBERTa > ModernBERT
func AutoDiscoverModels(modelsDir string) (*ModelPaths, error) {
	return AutoDiscoverModelsWithRegistry(modelsDir, nil)
}

// AutoDiscoverModelsWithRegistry discovers models using mom_registry for LoRA detection
// modelRegistry maps local paths to HuggingFace repo IDs (e.g., "models/mom-domain-classifier" -> "LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model")
func AutoDiscoverModelsWithRegistry(modelsDir string, modelRegistry map[string]string) (*ModelPaths, error) {
	if modelsDir == "" {
		modelsDir = "./models"
	}

	// Resolve symlinks to handle cases where models directory is a symlink (e.g., CI uses /mnt/models)
	resolved, err := filepath.EvalSymlinks(modelsDir)
	if err == nil && resolved != "" {
		modelsDir = resolved
	}

	// Check if models directory exists
	if _, statErr := os.Stat(modelsDir); os.IsNotExist(statErr) {
		return nil, fmt.Errorf("models directory does not exist: %s", modelsDir)
	}

	// Collect all available LoRA models by architecture
	architectureModels := map[string]*ArchitectureModels{
		"bert":       {Intent: "", PII: "", Security: ""},
		"roberta":    {Intent: "", PII: "", Security: ""},
		"modernbert": {Intent: "", PII: "", Security: ""},
	}

	// Legacy models for fallback
	legacyPaths := &ModelPaths{}

	// Walk through the models directory to collect all models
	err = filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip files, we're looking for directories
		if !info.IsDir() {
			return nil
		}

		dirName := strings.ToLower(info.Name())

		// Check if this is a LoRA model by looking up in mom_registry
		isLoRAIntent, isLORAPII, isLORASecurity := false, false, false
		if modelRegistry != nil {
			// Get the HuggingFace repo ID from mom_registry
			repoID, exists := modelRegistry[path]
			if exists {
				repoIDLower := strings.ToLower(repoID)
				// Check if the repo ID contains "lora_intent_classifier" or similar patterns
				isLoRAIntent = strings.Contains(repoIDLower, "lora_intent_classifier") ||
					strings.Contains(repoIDLower, "lora_domain_classifier")
				isLORAPII = strings.Contains(repoIDLower, "lora_pii_detector") ||
					strings.Contains(repoIDLower, "lora_pii_classifier")
				isLORASecurity = strings.Contains(repoIDLower, "lora_jailbreak_classifier") ||
					strings.Contains(repoIDLower, "lora_security_classifier")
			}
		}

		// Collect LoRA models by architecture
		// Use mom_registry to determine if it's a LoRA model, fallback to directory name
		switch {
		case isLoRAIntent || strings.HasPrefix(dirName, "lora_intent_classifier"):
			arch := detectArchitectureFromPath(dirName)
			if architectureModels[arch].Intent == "" {
				architectureModels[arch].Intent = path
			}

		case isLORAPII || strings.HasPrefix(dirName, "lora_pii_detector"):
			arch := detectArchitectureFromPath(dirName)
			if architectureModels[arch].PII == "" {
				architectureModels[arch].PII = path
			}

		case isLORASecurity || strings.HasPrefix(dirName, "lora_jailbreak_classifier"):
			arch := detectArchitectureFromPath(dirName)
			if architectureModels[arch].Security == "" {
				architectureModels[arch].Security = path
			}

		// Legacy ModernBERT models (fallback for backward compatibility)
		case strings.Contains(dirName, "modernbert") && strings.Contains(dirName, "base") && !strings.Contains(dirName, "classifier"):
			// ModernBERT base model: "modernbert-base", "modernbert_base", etc.
			if legacyPaths.ModernBertBase == "" {
				legacyPaths.ModernBertBase = path
			}

		case strings.Contains(dirName, "category") && strings.Contains(dirName, "classifier"):
			if legacyPaths.IntentClassifier == "" {
				legacyPaths.IntentClassifier = path
			}
			if legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				legacyPaths.ModernBertBase = path
			}

		case strings.Contains(dirName, "pii") && strings.Contains(dirName, "classifier"):
			if legacyPaths.PIIClassifier == "" {
				legacyPaths.PIIClassifier = path
			}
			if legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				legacyPaths.ModernBertBase = path
			}

		case (strings.Contains(dirName, "jailbreak") || strings.Contains(dirName, "security")) && strings.Contains(dirName, "classifier"):
			if legacyPaths.SecurityClassifier == "" {
				legacyPaths.SecurityClassifier = path
			}
			if legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				legacyPaths.ModernBertBase = path
			}
		}

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error scanning models directory: %w", err)
	}

	// Intelligent architecture selection based on performance priority: BERT > RoBERTa > ModernBERT
	architecturePriority := []string{"bert", "roberta", "modernbert"}

	for _, arch := range architecturePriority {
		models := architectureModels[arch]
		// Check if this architecture has a complete set of models
		if models.Intent != "" && models.PII != "" && models.Security != "" {
			return &ModelPaths{
				LoRAIntentClassifier:   models.Intent,
				LoRAPIIClassifier:      models.PII,
				LoRASecurityClassifier: models.Security,
				LoRAArchitecture:       arch,
				// Copy legacy paths for fallback compatibility
				ModernBertBase:     legacyPaths.ModernBertBase,
				IntentClassifier:   legacyPaths.IntentClassifier,
				PIIClassifier:      legacyPaths.PIIClassifier,
				SecurityClassifier: legacyPaths.SecurityClassifier,
			}, nil
		}
	}

	// If no complete LoRA architecture set found, return legacy models
	return legacyPaths, nil
}

// detectArchitectureFromPath detects model architecture from directory name
func detectArchitectureFromPath(dirName string) string {
	switch {
	case strings.Contains(dirName, "bert-base-uncased"):
		return "bert"
	case strings.Contains(dirName, "roberta-base"):
		return "roberta"
	case strings.Contains(dirName, "modernbert-base"):
		return "modernbert"
	default:
		// Default fallback
		return "bert"
	}
}

// ValidateModelPaths validates that all discovered paths contain valid model files
func ValidateModelPaths(paths *ModelPaths) error {
	if paths == nil {
		return fmt.Errorf("model paths is nil")
	}

	// If LoRA models are available, validate them
	if paths.HasLoRAModels() {
		loraChecks := map[string]string{
			"LoRA Intent classifier":   paths.LoRAIntentClassifier,
			"LoRA PII classifier":      paths.LoRAPIIClassifier,
			"LoRA Security classifier": paths.LoRASecurityClassifier,
		}

		for name, path := range loraChecks {
			if path == "" {
				return fmt.Errorf("%s model not found", name)
			}

			// Check if directory exists and contains model files
			if err := validateModelDirectory(path, name); err != nil {
				return err
			}
		}
		return nil
	}

	// If no LoRA models, validate legacy models
	if paths.HasLegacyModels() {
		legacyChecks := map[string]string{
			"ModernBERT base":     paths.ModernBertBase,
			"Intent classifier":   paths.IntentClassifier,
			"PII classifier":      paths.PIIClassifier,
			"Security classifier": paths.SecurityClassifier,
		}

		for name, path := range legacyChecks {
			if path == "" {
				return fmt.Errorf("%s model not found", name)
			}

			// Check if directory exists and contains model files
			if err := validateModelDirectory(path, name); err != nil {
				return err
			}
		}
		return nil
	}

	return fmt.Errorf("no valid models found (neither LoRA nor legacy)")
}

// validateModelDirectory checks if a directory contains valid model files
func validateModelDirectory(path, modelName string) error {
	// Check if directory exists
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return fmt.Errorf("%s model directory does not exist: %s", modelName, path)
	}
	if !info.IsDir() {
		return fmt.Errorf("%s model path is not a directory: %s", modelName, path)
	}

	// Check for common model files (at least one should exist)
	commonModelFiles := []string{
		"config.json",
		"pytorch_model.bin",
		"model.safetensors",
		"tokenizer.json",
		"vocab.txt",
	}

	hasModelFile := false
	for _, filename := range commonModelFiles {
		if _, err := os.Stat(filepath.Join(path, filename)); err == nil {
			hasModelFile = true
			break
		}
	}

	if !hasModelFile {
		return fmt.Errorf("%s model directory appears to be empty or invalid: %s", modelName, path)
	}

	return nil
}

// GetModelDiscoveryInfo returns detailed information about model discovery
func GetModelDiscoveryInfo(modelsDir string) map[string]interface{} {
	info := map[string]interface{}{
		"models_directory":  modelsDir,
		"discovery_status":  "failed",
		"discovered_models": map[string]interface{}{},
		"missing_models":    []string{},
		"errors":            []string{},
	}

	paths, err := AutoDiscoverModels(modelsDir)
	if err != nil {
		info["errors"] = append(info["errors"].([]string), err.Error())
		return info
	}

	// Add discovered models
	discovered := map[string]interface{}{
		"modernbert_base":     paths.ModernBertBase,
		"intent_classifier":   paths.IntentClassifier,
		"pii_classifier":      paths.PIIClassifier,
		"security_classifier": paths.SecurityClassifier,
	}
	info["discovered_models"] = discovered

	// Check for missing models
	missing := []string{}
	if paths.ModernBertBase == "" {
		missing = append(missing, "ModernBERT base model")
	}
	if paths.IntentClassifier == "" {
		missing = append(missing, "Intent classifier")
	}
	if paths.PIIClassifier == "" {
		missing = append(missing, "PII classifier")
	}
	if paths.SecurityClassifier == "" {
		missing = append(missing, "Security classifier")
	}
	info["missing_models"] = missing

	// Validate discovered models
	if err := ValidateModelPaths(paths); err != nil {
		info["errors"] = append(info["errors"].([]string), err.Error())
		info["discovery_status"] = "incomplete"
	} else {
		info["discovery_status"] = "complete"
	}

	return info
}

// AutoInitializeUnifiedClassifier attempts to auto-discover and initialize the unified classifier
// Prioritizes LoRA models over legacy ModernBERT models
func AutoInitializeUnifiedClassifier(modelsDir string) (*UnifiedClassifier, error) {
	return AutoInitializeUnifiedClassifierWithRegistry(modelsDir, nil)
}

// AutoInitializeUnifiedClassifierWithRegistry auto-discovers and initializes with mom_registry
func AutoInitializeUnifiedClassifierWithRegistry(modelsDir string, modelRegistry map[string]string) (*UnifiedClassifier, error) {
	// Discover models using mom_registry for LoRA detection
	paths, err := AutoDiscoverModelsWithRegistry(modelsDir, modelRegistry)
	if err != nil {
		return nil, fmt.Errorf("model discovery failed: %w", err)
	}

	// Validate paths
	if err := ValidateModelPaths(paths); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}

	// Check if we should use LoRA models
	if paths.PreferLoRA() {
		return initializeLoRAUnifiedClassifier(paths)
	}

	// Fallback to legacy ModernBERT initialization
	return initializeLegacyUnifiedClassifier(paths)
}

// initializeLoRAUnifiedClassifier initializes with LoRA models
func initializeLoRAUnifiedClassifier(paths *ModelPaths) (*UnifiedClassifier, error) {
	// Create unified classifier instance with LoRA mode
	classifier := &UnifiedClassifier{
		initialized: false,
		useLoRA:     true, // Mark as LoRA mode for high confidence
	}

	// Store LoRA model paths for later initialization
	// The actual C initialization will be done in unified_classifier.go
	classifier.loraModelPaths = &LoRAModelPaths{
		IntentPath:   paths.LoRAIntentClassifier,
		PIIPath:      paths.LoRAPIIClassifier,
		SecurityPath: paths.LoRASecurityClassifier,
		Architecture: paths.LoRAArchitecture,
	}

	// Mark as initialized - the actual C initialization will be lazy-loaded
	classifier.initialized = true

	// Pre-initialize LoRA C bindings to avoid lazy loading during first API call
	if err := classifier.initializeLoRABindings(); err != nil {
		return nil, fmt.Errorf("failed to pre-initialize LoRA bindings: %w", err)
	}
	classifier.loraInitialized = true

	return classifier, nil
}

// initializeLegacyUnifiedClassifier initializes with legacy ModernBERT models
func initializeLegacyUnifiedClassifier(paths *ModelPaths) (*UnifiedClassifier, error) {
	// Load intent labels from the actual model's mapping file
	categoryMappingPath := filepath.Join(paths.IntentClassifier, "category_mapping.json")
	categoryMapping, err := LoadCategoryMapping(categoryMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load category mapping from %s: %w", categoryMappingPath, err)
	}

	// Extract intent labels in correct order (by index)
	intentLabels := make([]string, len(categoryMapping.IdxToCategory))
	for i := 0; i < len(categoryMapping.IdxToCategory); i++ {
		if label, exists := categoryMapping.IdxToCategory[fmt.Sprintf("%d", i)]; exists {
			intentLabels[i] = label
		} else {
			return nil, fmt.Errorf("missing label for index %d in category mapping", i)
		}
	}

	// Load PII labels from the actual model's mapping file
	var piiLabels []string
	piiMappingPath := filepath.Join(paths.PIIClassifier, "pii_type_mapping.json")
	if _, statErr := os.Stat(piiMappingPath); statErr == nil {
		piiMapping, loadErr := LoadPIIMapping(piiMappingPath)
		if loadErr != nil {
			return nil, fmt.Errorf("failed to load PII mapping from %s: %w", piiMappingPath, loadErr)
		}
		// Extract labels from PII mapping (ordered by index)
		piiLabels = make([]string, len(piiMapping.IdxToLabel))
		for i := 0; i < len(piiMapping.IdxToLabel); i++ {
			if label, exists := piiMapping.IdxToLabel[fmt.Sprintf("%d", i)]; exists {
				piiLabels[i] = label
			} else {
				return nil, fmt.Errorf("missing PII label for index %d", i)
			}
		}
	} else {
		return nil, fmt.Errorf("PII mapping file not found at %s - required for unified classifier", piiMappingPath)
	}

	// Load security labels from the actual model's mapping file
	var securityLabels []string
	securityMappingPath := filepath.Join(paths.SecurityClassifier, "jailbreak_type_mapping.json")
	if _, statErr2 := os.Stat(securityMappingPath); statErr2 == nil {
		jailbreakMapping, loadErr2 := LoadJailbreakMapping(securityMappingPath)
		if loadErr2 != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping from %s: %w", securityMappingPath, loadErr2)
		}
		// Extract labels from jailbreak mapping (ordered by index)
		securityLabels = make([]string, len(jailbreakMapping.IdxToLabel))
		for i := 0; i < len(jailbreakMapping.IdxToLabel); i++ {
			if label, exists := jailbreakMapping.IdxToLabel[fmt.Sprintf("%d", i)]; exists {
				securityLabels[i] = label
			} else {
				return nil, fmt.Errorf("missing security label for index %d", i)
			}
		}
	} else {
		return nil, fmt.Errorf("security mapping file not found at %s - required for unified classifier", securityMappingPath)
	}

	// Get global unified classifier instance
	classifier := GetGlobalUnifiedClassifier()

	// Initialize with discovered paths and config-based labels
	err = classifier.Initialize(
		paths.ModernBertBase,
		paths.IntentClassifier,
		paths.PIIClassifier,
		paths.SecurityClassifier,
		intentLabels,
		piiLabels,
		securityLabels,
		false, // Default to GPU, will fallback to CPU if needed
	)
	if err != nil {
		return nil, fmt.Errorf("unified classifier initialization failed: %w", err)
	}

	return classifier, nil
}
