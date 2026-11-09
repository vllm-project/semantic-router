package handlers

// Setup imports contain routing intent, even when their destination is already
// the runtime-owned path. Realize that candidate before publication; the normal
// sync path intentionally leaves an already-realized runtime file untouched.
// The caller validates the realized output before publishing it.
func realizeSetupCandidateConfig(configPath string, candidate *setupConfigFile) (*setupConfigFile, error) {
	if !managedSplitManagementReachabilityRequired() {
		return candidate, nil
	}
	// The typed global holds effective Router defaults. Only the authored
	// override may become CLI input, or its absent listener becomes explicit
	// loopback intent and omitted false values can revert to defaults.
	raw, err := marshalYAMLBytes(candidate.canonicalTransport())
	if err != nil {
		return nil, err
	}
	realized, err := realizeRuntimeConfigWithCLI(raw, configPath, runtimeMaterialization{managedListener: true})
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
