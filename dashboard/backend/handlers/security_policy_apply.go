package handlers

import "log"

// applySecurityFragment merges the generated fragment into config.yaml and
// triggers a runtime hot-reload. Returns true if the apply succeeded.
func applySecurityFragment(fragment *GeneratedRouterFragment) bool {
	if securityPolicyConfigPath == "" {
		return false
	}

	deployMu.Lock()
	defer deployMu.Unlock()

	yamlBytes, err := toCanonicalYAML(fragment)
	if err != nil {
		log.Printf("[SecurityPolicy] failed to marshal canonical YAML: %v", err)
		return false
	}

	previous, err := captureConfigFileSnapshot(securityPolicyConfigPath)
	if err != nil {
		log.Printf("[SecurityPolicy] failed to read config file %s: %v", securityPolicyConfigPath, err)
		return false
	}
	existingData := previous.data

	merged, err := mergeDeployPayload(existingData, DeployRequest{YAML: string(yamlBytes)})
	if err != nil {
		log.Printf("[SecurityPolicy] failed to merge fragment into config: %v", err)
		return false
	}

	if err := validateMergedSecurityConfig(merged); err != nil {
		log.Printf("[SecurityPolicy] merged config validation failed: %v", err)
		return false
	}

	if err := writeConfigAtomically(securityPolicyConfigPath, merged); err != nil {
		log.Printf("[SecurityPolicy] failed to write config: %v", err)
		return false
	}

	log.Printf("[SecurityPolicy] Config written to %s (%d bytes)", securityPolicyConfigPath, len(merged))

	if err := applyWrittenConfig(securityPolicyConfigPath, securityPolicyConfigDir, previous, true); err != nil {
		log.Printf("[SecurityPolicy] failed to apply config to runtime: %v", err)
		return false
	}

	return true
}
