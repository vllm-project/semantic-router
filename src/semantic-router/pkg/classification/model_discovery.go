package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
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
	resolvedModelsDir, err := resolveModelsDir(modelsDir)
	if err != nil {
		return nil, err
	}

	architectureModels := newArchitectureModels()
	legacyPaths := &ModelPaths{}
	if err := collectDiscoveredModels(resolvedModelsDir, modelRegistry, architectureModels, legacyPaths); err != nil {
		return nil, err
	}

	return selectPreferredDiscoveredModels(architectureModels, legacyPaths), nil
}

func resolveModelsDir(modelsDir string) (string, error) {
	if modelsDir == "" {
		modelsDir = "./models"
	}

	resolved, err := filepath.EvalSymlinks(modelsDir)
	if err == nil && resolved != "" {
		modelsDir = resolved
	}

	if _, statErr := os.Stat(modelsDir); os.IsNotExist(statErr) {
		return "", fmt.Errorf("models directory does not exist: %s", modelsDir)
	}

	return modelsDir, nil
}

func newArchitectureModels() map[string]*ArchitectureModels {
	return map[string]*ArchitectureModels{
		"bert":       {Intent: "", PII: "", Security: ""},
		"roberta":    {Intent: "", PII: "", Security: ""},
		"modernbert": {Intent: "", PII: "", Security: ""},
	}
}

func collectDiscoveredModels(modelsDir string, modelRegistry map[string]string, architectureModels map[string]*ArchitectureModels, legacyPaths *ModelPaths) error {
	walkErr := filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return nil
		}

		dirName := strings.ToLower(info.Name())
		isLoRAIntent, isLORAPII, isLORASecurity := detectLoRAModelKinds(path, modelRegistry)
		recordDiscoveredModel(
			path,
			dirName,
			isLoRAIntent,
			isLORAPII,
			isLORASecurity,
			architectureModels,
			legacyPaths,
		)
		return nil
	})
	if walkErr != nil {
		return fmt.Errorf("error scanning models directory: %w", walkErr)
	}
	return nil
}

func detectLoRAModelKinds(path string, modelRegistry map[string]string) (bool, bool, bool) {
	if modelRegistry == nil {
		return false, false, false
	}

	repoID, exists := modelRegistry[path]
	if !exists {
		return false, false, false
	}

	repoIDLower := strings.ToLower(repoID)
	isLoRAIntent := strings.Contains(repoIDLower, "lora_intent_classifier") ||
		strings.Contains(repoIDLower, "lora_domain_classifier")
	isLORAPII := strings.Contains(repoIDLower, "lora_pii_detector") ||
		strings.Contains(repoIDLower, "lora_pii_classifier")
	isLORASecurity := strings.Contains(repoIDLower, "lora_jailbreak_classifier") ||
		strings.Contains(repoIDLower, "lora_security_classifier")
	return isLoRAIntent, isLORAPII, isLORASecurity
}

func recordDiscoveredModel(path, dirName string, isLoRAIntent, isLORAPII, isLORASecurity bool, architectureModels map[string]*ArchitectureModels, legacyPaths *ModelPaths) {
	if recordDiscoveredLoRAModel(path, dirName, isLoRAIntent, isLORAPII, isLORASecurity, architectureModels) {
		return
	}
	recordDiscoveredLegacyModel(path, dirName, legacyPaths)
}

func recordDiscoveredLoRAModel(path, dirName string, isLoRAIntent, isLORAPII, isLORASecurity bool, architectureModels map[string]*ArchitectureModels) bool {
	if isLoRAIntent || strings.HasPrefix(dirName, "lora_intent_classifier") {
		arch := detectArchitectureFromPath(dirName)
		if architectureModels[arch].Intent == "" {
			architectureModels[arch].Intent = path
		}
		return true
	}
	if isLORAPII || strings.HasPrefix(dirName, "lora_pii_detector") {
		arch := detectArchitectureFromPath(dirName)
		if architectureModels[arch].PII == "" {
			architectureModels[arch].PII = path
		}
		return true
	}
	if isLORASecurity || strings.HasPrefix(dirName, "lora_jailbreak_classifier") {
		arch := detectArchitectureFromPath(dirName)
		if architectureModels[arch].Security == "" {
			architectureModels[arch].Security = path
		}
		return true
	}
	return false
}

func recordDiscoveredLegacyModel(path, dirName string, legacyPaths *ModelPaths) {
	if isLegacyModernBertBase(dirName) {
		if legacyPaths.ModernBertBase == "" {
			legacyPaths.ModernBertBase = path
		}
		return
	}
	if recordLegacyIntentClassifier(path, dirName, legacyPaths) {
		return
	}
	if recordLegacyPIIClassifier(path, dirName, legacyPaths) {
		return
	}
	recordLegacySecurityClassifier(path, dirName, legacyPaths)
}

func isLegacyModernBertBase(dirName string) bool {
	return strings.Contains(dirName, "modernbert") &&
		strings.Contains(dirName, "base") &&
		!strings.Contains(dirName, "classifier")
}

func recordLegacyIntentClassifier(path, dirName string, legacyPaths *ModelPaths) bool {
	if !strings.Contains(dirName, "category") || !strings.Contains(dirName, "classifier") {
		return false
	}
	if legacyPaths.IntentClassifier == "" {
		legacyPaths.IntentClassifier = path
	}
	updateLegacyModernBertBase(path, dirName, legacyPaths)
	return true
}

func recordLegacyPIIClassifier(path, dirName string, legacyPaths *ModelPaths) bool {
	if !strings.Contains(dirName, "pii") || !strings.Contains(dirName, "classifier") {
		return false
	}
	if legacyPaths.PIIClassifier == "" {
		legacyPaths.PIIClassifier = path
	}
	updateLegacyModernBertBase(path, dirName, legacyPaths)
	return true
}

func recordLegacySecurityClassifier(path, dirName string, legacyPaths *ModelPaths) {
	if (!strings.Contains(dirName, "jailbreak") && !strings.Contains(dirName, "security")) ||
		!strings.Contains(dirName, "classifier") {
		return
	}
	if legacyPaths.SecurityClassifier == "" {
		legacyPaths.SecurityClassifier = path
	}
	updateLegacyModernBertBase(path, dirName, legacyPaths)
}

func updateLegacyModernBertBase(path, dirName string, legacyPaths *ModelPaths) {
	if legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
		legacyPaths.ModernBertBase = path
	}
}

func selectPreferredDiscoveredModels(architectureModels map[string]*ArchitectureModels, legacyPaths *ModelPaths) *ModelPaths {
	for _, arch := range []string{"bert", "roberta", "modernbert"} {
		models := architectureModels[arch]
		if models.Intent != "" && models.PII != "" && models.Security != "" {
			return &ModelPaths{
				LoRAIntentClassifier:   models.Intent,
				LoRAPIIClassifier:      models.PII,
				LoRASecurityClassifier: models.Security,
				LoRAArchitecture:       arch,
				ModernBertBase:         legacyPaths.ModernBertBase,
				IntentClassifier:       legacyPaths.IntentClassifier,
				PIIClassifier:          legacyPaths.PIIClassifier,
				SecurityClassifier:     legacyPaths.SecurityClassifier,
			}
		}
	}
	return legacyPaths
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
	if err := candle_binding.CurrentBackendContract().RequireFeature(candle_binding.FeatureLoRABatchInference); err != nil {
		return nil, err
	}

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
	if err := candle_binding.CurrentBackendContract().RequireFeature(candle_binding.FeatureUnifiedClassification); err != nil {
		return nil, err
	}

	intentLabels, err := loadLegacyIntentLabels(paths.IntentClassifier)
	if err != nil {
		return nil, err
	}
	piiLabels, err := loadLegacyPIILabels(paths.PIIClassifier)
	if err != nil {
		return nil, err
	}
	securityLabels, err := loadLegacySecurityLabels(paths.SecurityClassifier)
	if err != nil {
		return nil, err
	}

	return initializeLegacyUnifiedBindings(paths, intentLabels, piiLabels, securityLabels)
}

func loadLegacyIntentLabels(intentClassifierPath string) ([]string, error) {
	categoryMappingPath := filepath.Join(intentClassifierPath, "category_mapping.json")
	categoryMapping, err := LoadCategoryMapping(categoryMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load category mapping from %s: %w", categoryMappingPath, err)
	}
	return orderedCategoryLabels(categoryMapping)
}

func orderedCategoryLabels(categoryMapping *CategoryMapping) ([]string, error) {
	intentLabels := make([]string, len(categoryMapping.IdxToCategory))
	for i := 0; i < len(categoryMapping.IdxToCategory); i++ {
		label, exists := categoryMapping.IdxToCategory[fmt.Sprintf("%d", i)]
		if !exists {
			return nil, fmt.Errorf("missing label for index %d in category mapping", i)
		}
		intentLabels[i] = label
	}
	return intentLabels, nil
}

func loadLegacyPIILabels(piiClassifierPath string) ([]string, error) {
	piiMappingPath := filepath.Join(piiClassifierPath, "pii_type_mapping.json")
	if err := ensureRequiredClassifierMappingFile(piiMappingPath, "PII mapping file"); err != nil {
		return nil, err
	}
	piiMapping, err := LoadPIIMapping(piiMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load PII mapping from %s: %w", piiMappingPath, err)
	}
	return orderedPIILabels(piiMapping)
}

func orderedPIILabels(piiMapping *PIIMapping) ([]string, error) {
	piiLabels := make([]string, len(piiMapping.IdxToLabel))
	for i := 0; i < len(piiMapping.IdxToLabel); i++ {
		label, exists := piiMapping.IdxToLabel[fmt.Sprintf("%d", i)]
		if !exists {
			return nil, fmt.Errorf("missing PII label for index %d", i)
		}
		piiLabels[i] = label
	}
	return piiLabels, nil
}

func loadLegacySecurityLabels(securityClassifierPath string) ([]string, error) {
	securityMappingPath := filepath.Join(securityClassifierPath, "jailbreak_type_mapping.json")
	if err := ensureRequiredClassifierMappingFile(securityMappingPath, "security mapping file"); err != nil {
		return nil, err
	}
	jailbreakMapping, err := LoadJailbreakMapping(securityMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load jailbreak mapping from %s: %w", securityMappingPath, err)
	}
	return orderedSecurityLabels(jailbreakMapping)
}

func orderedSecurityLabels(jailbreakMapping *JailbreakMapping) ([]string, error) {
	numLabels := jailbreakMapping.GetJailbreakTypeCount()
	securityLabels := make([]string, numLabels)
	for i := 0; i < numLabels; i++ {
		label, exists := jailbreakMapping.GetJailbreakTypeFromIndex(i)
		if !exists {
			return nil, fmt.Errorf("missing security label for index %d", i)
		}
		securityLabels[i] = label
	}
	return securityLabels, nil
}

func ensureRequiredClassifierMappingFile(path string, description string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	return fmt.Errorf("%s not found at %s - required for unified classifier", description, path)
}

func initializeLegacyUnifiedBindings(
	paths *ModelPaths,
	intentLabels []string,
	piiLabels []string,
	securityLabels []string,
) (*UnifiedClassifier, error) {
	classifier := GetGlobalUnifiedClassifier()
	if err := classifier.Initialize(
		paths.ModernBertBase,
		paths.IntentClassifier,
		paths.PIIClassifier,
		paths.SecurityClassifier,
		intentLabels,
		piiLabels,
		securityLabels,
		false, // Default to GPU, will fallback to CPU if needed
	); err != nil {
		return nil, fmt.Errorf("unified classifier initialization failed: %w", err)
	}
	return classifier, nil
}
