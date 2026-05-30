package classification

import "strings"

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
	modelsDir, err := normalizeModelDiscoveryDir(modelsDir)
	if err != nil {
		return nil, err
	}

	discovered, err := collectDiscoveredModels(modelsDir, modelRegistry)
	if err != nil {
		return nil, err
	}
	return discovered.selectedPaths(), nil
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
