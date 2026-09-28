package handlers

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
	// The typed global holds effective Router defaults. Only the authored
	// override may become CLI input, or its absent listener becomes explicit
	// loopback intent and omitted false values can revert to defaults.
	raw, err := marshalYAMLBytes(candidate.canonicalTransport())
	if err != nil {
		return nil, err
	}
	realized, err := realizeRuntimeConfigWithCLI(raw, configPath, runtimeMaterialization{managedListener: true, skipKBBootstrap: skipKBBootstrap})
	if err != nil {
		return nil, err
	}
	// Resolve defaults for validation while retaining the realized raw global
	// for HTTP responses and publication, including explicit false and zero.
	result, err := decodeYAMLTaggedBytes[setupConfigFile](realized)
	if err != nil {
		return nil, err
	}
	return &result, nil
}
