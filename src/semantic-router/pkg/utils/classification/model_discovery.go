package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ModelPaths holds the discovered model paths
type ModelPaths struct {
	ModernBertBase     string
	IntentClassifier   string
	PIIClassifier      string
	SecurityClassifier string
}

// IsComplete checks if all required models are found
func (mp *ModelPaths) IsComplete() bool {
	return mp.ModernBertBase != "" &&
		mp.IntentClassifier != "" &&
		mp.PIIClassifier != "" &&
		mp.SecurityClassifier != ""
}

// AutoDiscoverModels automatically discovers model files in the models directory
func AutoDiscoverModels(modelsDir string) (*ModelPaths, error) {
	if modelsDir == "" {
		modelsDir = "./models"
	}

	// Check if models directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("models directory does not exist: %s", modelsDir)
	}

	paths := &ModelPaths{}

	// Walk through the models directory
	err := filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip files, we're looking for directories
		if !info.IsDir() {
			return nil
		}

		dirName := strings.ToLower(info.Name())

		// Match known model patterns
		switch {
		case strings.Contains(dirName, "modernbert") && strings.Contains(dirName, "base") && !strings.Contains(dirName, "classifier"):
			// ModernBERT base model: "modernbert-base", "modernbert_base", etc.
			if paths.ModernBertBase == "" { // Take the first match
				paths.ModernBertBase = path
			}

		case strings.Contains(dirName, "category") && strings.Contains(dirName, "classifier"):
			// Intent/Category classifier: "category_classifier_modernbert-base_model", etc.
			if paths.IntentClassifier == "" {
				paths.IntentClassifier = path
			}
			// If no dedicated ModernBERT base found, use this classifier as shared encoder
			if paths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				paths.ModernBertBase = path
			}

		case strings.Contains(dirName, "pii") && strings.Contains(dirName, "classifier"):
			// PII classifier: "pii_classifier_modernbert-base_presidio_token_model", etc.
			if paths.PIIClassifier == "" {
				paths.PIIClassifier = path
			}
			// If no dedicated ModernBERT base found, use this classifier as shared encoder
			if paths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				paths.ModernBertBase = path
			}

		case (strings.Contains(dirName, "jailbreak") || strings.Contains(dirName, "security")) && strings.Contains(dirName, "classifier"):
			// Security/Jailbreak classifier: "jailbreak_classifier_modernbert-base_model", etc.
			if paths.SecurityClassifier == "" {
				paths.SecurityClassifier = path
			}
			// If no dedicated ModernBERT base found, use this classifier as shared encoder
			if paths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				paths.ModernBertBase = path
			}
		}

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error scanning models directory: %v", err)
	}

	return paths, nil
}

// ValidateModelPaths validates that all discovered paths contain valid model files
func ValidateModelPaths(paths *ModelPaths) error {
	if paths == nil {
		return fmt.Errorf("model paths is nil")
	}

	// Check each path
	modelChecks := map[string]string{
		"ModernBERT base":     paths.ModernBertBase,
		"Intent classifier":   paths.IntentClassifier,
		"PII classifier":      paths.PIIClassifier,
		"Security classifier": paths.SecurityClassifier,
	}

	for name, path := range modelChecks {
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
func AutoInitializeUnifiedClassifier(modelsDir string) (*UnifiedClassifier, error) {
	// Discover models
	paths, err := AutoDiscoverModels(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("model discovery failed: %v", err)
	}

	// Validate paths
	if err := ValidateModelPaths(paths); err != nil {
		return nil, fmt.Errorf("model validation failed: %v", err)
	}

	// Load intent labels from the actual model's mapping file
	categoryMappingPath := filepath.Join(paths.IntentClassifier, "category_mapping.json")
	categoryMapping, err := LoadCategoryMapping(categoryMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load category mapping from %s: %v", categoryMappingPath, err)
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
	if _, err := os.Stat(piiMappingPath); err == nil {
		piiMapping, err := LoadPIIMapping(piiMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping from %s: %v", piiMappingPath, err)
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
	if _, err := os.Stat(securityMappingPath); err == nil {
		jailbreakMapping, err := LoadJailbreakMapping(securityMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping from %s: %v", securityMappingPath, err)
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
		return nil, fmt.Errorf("unified classifier initialization failed: %v", err)
	}

	return classifier, nil
}
