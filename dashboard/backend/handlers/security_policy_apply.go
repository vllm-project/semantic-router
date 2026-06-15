package handlers

import (
	"errors"
	"log"
	"os"
)

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

	existingData, err := os.ReadFile(securityPolicyConfigPath)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			log.Printf("[SecurityPolicy] failed to read config file %s: %v", securityPolicyConfigPath, err)
			return false
		}
		existingData = nil
	}

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

	if err := applyWrittenConfig(securityPolicyConfigPath, securityPolicyConfigDir, existingData, true); err != nil {
		log.Printf("[SecurityPolicy] failed to apply config to runtime: %v", err)
		return false
	}

	return true
}
