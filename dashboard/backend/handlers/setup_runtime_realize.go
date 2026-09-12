package handlers

import (
	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Setup imports contain routing intent, even when their destination is already
// the runtime-owned path. Realize that candidate before publication; the normal
// sync path intentionally leaves an already-realized runtime file untouched.
func realizeSetupCandidateConfig(configPath string, candidate *setupConfigFile, skipKBBootstrap bool) (*setupConfigFile, error) {
	if !managedSplitManagementReachabilityRequired() {
		return candidate, nil
	}
	// Check source references before the CLI can bootstrap their KB assets.
	// The caller also validates the realized output before publishing it.
	if !skipKBBootstrap {
		if err := validateSetupCandidate(configPath, candidate); err != nil {
			return nil, err
		}
	}
	raw, err := marshalYAMLBytes(candidate.CanonicalConfig)
	if err != nil {
		return nil, err
	}
	realized, err := realizeRuntimeConfigWithCLI(raw, configPath, runtimeMaterialization{managedListener: true, skipKBBootstrap: skipKBBootstrap})
	if err != nil {
		return nil, err
	}
	var result setupConfigFile
	if candidate.Global == nil {
		// Preserve setup's full Router defaults without turning its missing
		// listener into authored standalone intent before CLI realization.
		defaults := routerconfig.DefaultCanonicalGlobal()
		result.Global = &defaults
	}
	if err := yaml.Unmarshal(realized, &result); err != nil {
		return nil, err
	}
	return &result, nil
}
