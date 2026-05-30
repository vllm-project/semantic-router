package classification

import (
	"fmt"
	"os"
	"path/filepath"
)

type legacyUnifiedLabels struct {
	intent   []string
	pii      []string
	security []string
}

func loadLegacyUnifiedLabels(paths *ModelPaths) (legacyUnifiedLabels, error) {
	if paths == nil {
		return legacyUnifiedLabels{}, fmt.Errorf("model paths is nil")
	}

	intentLabels, err := loadLegacyIntentLabels(paths.IntentClassifier)
	if err != nil {
		return legacyUnifiedLabels{}, err
	}
	piiLabels, err := loadLegacyPIILabels(paths.PIIClassifier)
	if err != nil {
		return legacyUnifiedLabels{}, err
	}
	securityLabels, err := loadLegacySecurityLabels(paths.SecurityClassifier)
	if err != nil {
		return legacyUnifiedLabels{}, err
	}

	return legacyUnifiedLabels{
		intent:   intentLabels,
		pii:      piiLabels,
		security: securityLabels,
	}, nil
}

func loadLegacyIntentLabels(modelDir string) ([]string, error) {
	mappingPath := filepath.Join(modelDir, "category_mapping.json")
	categoryMapping, err := LoadCategoryMapping(mappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load category mapping from %s: %w", mappingPath, err)
	}

	labels := make([]string, len(categoryMapping.IdxToCategory))
	for i := 0; i < len(categoryMapping.IdxToCategory); i++ {
		if label, exists := categoryMapping.IdxToCategory[fmt.Sprintf("%d", i)]; exists {
			labels[i] = label
			continue
		}
		return nil, fmt.Errorf("missing label for index %d in category mapping", i)
	}
	return labels, nil
}

func loadLegacyPIILabels(modelDir string) ([]string, error) {
	mappingPath := filepath.Join(modelDir, "pii_type_mapping.json")
	if _, statErr := os.Stat(mappingPath); statErr != nil {
		return nil, fmt.Errorf("PII mapping file not found at %s - required for unified classifier", mappingPath)
	}

	piiMapping, err := LoadPIIMapping(mappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load PII mapping from %s: %w", mappingPath, err)
	}

	labels := make([]string, len(piiMapping.IdxToLabel))
	for i := 0; i < len(piiMapping.IdxToLabel); i++ {
		if label, exists := piiMapping.IdxToLabel[fmt.Sprintf("%d", i)]; exists {
			labels[i] = label
			continue
		}
		return nil, fmt.Errorf("missing PII label for index %d", i)
	}
	return labels, nil
}

func loadLegacySecurityLabels(modelDir string) ([]string, error) {
	mappingPath := filepath.Join(modelDir, "jailbreak_type_mapping.json")
	if _, statErr := os.Stat(mappingPath); statErr != nil {
		return nil, fmt.Errorf("security mapping file not found at %s - required for unified classifier", mappingPath)
	}

	jailbreakMapping, err := LoadJailbreakMapping(mappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load jailbreak mapping from %s: %w", mappingPath, err)
	}

	numLabels := jailbreakMapping.GetJailbreakTypeCount()
	labels := make([]string, numLabels)
	for i := 0; i < numLabels; i++ {
		if label, exists := jailbreakMapping.GetJailbreakTypeFromIndex(i); exists {
			labels[i] = label
			continue
		}
		return nil, fmt.Errorf("missing security label for index %d", i)
	}
	return labels, nil
}
