package classification

// GetModelDiscoveryInfo returns detailed information about model discovery.
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

	info["discovered_models"] = map[string]interface{}{
		"modernbert_base":     paths.ModernBertBase,
		"intent_classifier":   paths.IntentClassifier,
		"pii_classifier":      paths.PIIClassifier,
		"security_classifier": paths.SecurityClassifier,
	}
	info["missing_models"] = missingDiscoveredModels(paths)

	if err := ValidateModelPaths(paths); err != nil {
		info["errors"] = append(info["errors"].([]string), err.Error())
		info["discovery_status"] = "incomplete"
		return info
	}

	info["discovery_status"] = "complete"
	return info
}

func missingDiscoveredModels(paths *ModelPaths) []string {
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
	return missing
}
