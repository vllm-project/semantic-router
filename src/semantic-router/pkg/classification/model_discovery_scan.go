package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type discoveredModels struct {
	architectureModels map[string]*ArchitectureModels
	legacyPaths        *ModelPaths
}

func normalizeModelDiscoveryDir(modelsDir string) (string, error) {
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

func collectDiscoveredModels(modelsDir string, modelRegistry map[string]string) (*discoveredModels, error) {
	discovered := newDiscoveredModels()
	err := filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return nil
		}
		discovered.collectDirectory(path, strings.ToLower(info.Name()), modelRegistry)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error scanning models directory: %w", err)
	}
	return discovered, nil
}

func newDiscoveredModels() *discoveredModels {
	return &discoveredModels{
		architectureModels: map[string]*ArchitectureModels{
			"bert":       {},
			"roberta":    {},
			"modernbert": {},
		},
		legacyPaths: &ModelPaths{},
	}
}

func (d *discoveredModels) collectDirectory(path string, dirName string, modelRegistry map[string]string) {
	loraMatch := classifyLoRAModel(path, dirName, modelRegistry)
	switch loraMatch.kind {
	case loraIntentModel:
		d.setLoRAIntent(loraMatch.architecture, path)
	case loraPIIModel:
		d.setLoRAPII(loraMatch.architecture, path)
	case loraSecurityModel:
		d.setLoRASecurity(loraMatch.architecture, path)
	default:
		d.collectLegacyDirectory(path, dirName)
	}
}

func (d *discoveredModels) setLoRAIntent(architecture string, path string) {
	if d.architectureModels[architecture].Intent == "" {
		d.architectureModels[architecture].Intent = path
	}
}

func (d *discoveredModels) setLoRAPII(architecture string, path string) {
	if d.architectureModels[architecture].PII == "" {
		d.architectureModels[architecture].PII = path
	}
}

func (d *discoveredModels) setLoRASecurity(architecture string, path string) {
	if d.architectureModels[architecture].Security == "" {
		d.architectureModels[architecture].Security = path
	}
}

func (d *discoveredModels) collectLegacyDirectory(path string, dirName string) {
	switch {
	case isLegacyModernBertBaseDir(dirName):
		if d.legacyPaths.ModernBertBase == "" {
			d.legacyPaths.ModernBertBase = path
		}
	case isLegacyIntentDir(dirName):
		if d.legacyPaths.IntentClassifier == "" {
			d.legacyPaths.IntentClassifier = path
		}
		d.setModernBertBaseFromClassifier(path, dirName)
	case isLegacyPIIDir(dirName):
		if d.legacyPaths.PIIClassifier == "" {
			d.legacyPaths.PIIClassifier = path
		}
		d.setModernBertBaseFromClassifier(path, dirName)
	case isLegacySecurityDir(dirName):
		if d.legacyPaths.SecurityClassifier == "" {
			d.legacyPaths.SecurityClassifier = path
		}
		d.setModernBertBaseFromClassifier(path, dirName)
	}
}

func (d *discoveredModels) setModernBertBaseFromClassifier(path string, dirName string) {
	if d.legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
		d.legacyPaths.ModernBertBase = path
	}
}

func (d *discoveredModels) selectedPaths() *ModelPaths {
	for _, architecture := range []string{"bert", "roberta", "modernbert"} {
		models := d.architectureModels[architecture]
		if !hasCompleteArchitectureModels(models) {
			continue
		}
		return &ModelPaths{
			LoRAIntentClassifier:   models.Intent,
			LoRAPIIClassifier:      models.PII,
			LoRASecurityClassifier: models.Security,
			LoRAArchitecture:       architecture,
			ModernBertBase:         d.legacyPaths.ModernBertBase,
			IntentClassifier:       d.legacyPaths.IntentClassifier,
			PIIClassifier:          d.legacyPaths.PIIClassifier,
			SecurityClassifier:     d.legacyPaths.SecurityClassifier,
		}
	}
	return d.legacyPaths
}

func hasCompleteArchitectureModels(models *ArchitectureModels) bool {
	return models != nil && models.Intent != "" && models.PII != "" && models.Security != ""
}

func isLegacyModernBertBaseDir(dirName string) bool {
	return strings.Contains(dirName, "modernbert") &&
		strings.Contains(dirName, "base") &&
		!strings.Contains(dirName, "classifier")
}

func isLegacyIntentDir(dirName string) bool {
	return strings.Contains(dirName, "category") && strings.Contains(dirName, "classifier")
}

func isLegacyPIIDir(dirName string) bool {
	return strings.Contains(dirName, "pii") && strings.Contains(dirName, "classifier")
}

func isLegacySecurityDir(dirName string) bool {
	return (strings.Contains(dirName, "jailbreak") || strings.Contains(dirName, "security")) &&
		strings.Contains(dirName, "classifier")
}

type loraModelKind int

const (
	notLoRAModel loraModelKind = iota
	loraIntentModel
	loraPIIModel
	loraSecurityModel
)

type loraModelMatch struct {
	kind         loraModelKind
	architecture string
}

func classifyLoRAModel(path string, dirName string, modelRegistry map[string]string) loraModelMatch {
	intent, pii, security := loraRegistryMatch(path, modelRegistry)
	switch {
	case intent || strings.HasPrefix(dirName, "lora_intent_classifier"):
		return loraModelMatch{kind: loraIntentModel, architecture: detectArchitectureFromPath(dirName)}
	case pii || strings.HasPrefix(dirName, "lora_pii_detector"):
		return loraModelMatch{kind: loraPIIModel, architecture: detectArchitectureFromPath(dirName)}
	case security || strings.HasPrefix(dirName, "lora_jailbreak_classifier"):
		return loraModelMatch{kind: loraSecurityModel, architecture: detectArchitectureFromPath(dirName)}
	default:
		return loraModelMatch{kind: notLoRAModel}
	}
}

func loraRegistryMatch(path string, modelRegistry map[string]string) (bool, bool, bool) {
	if modelRegistry == nil {
		return false, false, false
	}
	repoID, exists := modelRegistry[path]
	if !exists {
		return false, false, false
	}

	repoIDLower := strings.ToLower(repoID)
	isIntent := strings.Contains(repoIDLower, "lora_intent_classifier") ||
		strings.Contains(repoIDLower, "lora_domain_classifier")
	isPII := strings.Contains(repoIDLower, "lora_pii_detector") ||
		strings.Contains(repoIDLower, "lora_pii_classifier")
	isSecurity := strings.Contains(repoIDLower, "lora_jailbreak_classifier") ||
		strings.Contains(repoIDLower, "lora_security_classifier")
	return isIntent, isPII, isSecurity
}
