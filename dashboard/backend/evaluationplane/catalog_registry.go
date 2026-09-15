package evaluationplane

import (
	"fmt"
	"sort"
	"strings"
)

func resolveRegistryOptions(values []RegistryOptions) (RegistryOptions, error) {
	if len(values) > 1 {
		return RegistryOptions{}, fmt.Errorf("only one registry options value is accepted")
	}
	if len(values) == 0 {
		return RegistryOptions{}, nil
	}
	return values[0], nil
}

func validateRegistryOptions(options RegistryOptions) error {
	if len(options.Mixtures) > 0 && len(options.DeploymentTargets) > 0 {
		return fmt.Errorf("default and deployment-scoped evaluation targets cannot be registered together")
	}
	if options.DefaultConfigDigest != "" && !digestPattern.MatchString(options.DefaultConfigDigest) {
		return fmt.Errorf("default evaluation config digest is invalid")
	}
	if err := validateTargetContract(options.RouterAPIKey, options.EnvoyAPIKey, nil, ""); err != nil {
		return err
	}
	for _, mixture := range options.Mixtures {
		if err := validateMixtureTargetSnapshot(mixture); err != nil {
			return err
		}
	}
	seenDeploymentTargets := make(map[string]struct{}, len(options.DeploymentTargets))
	for _, target := range options.DeploymentTargets {
		if !portableIDPattern.MatchString(target.TargetID) ||
			!deploymentIDPattern.MatchString(target.DeploymentID) ||
			target.DeploymentName == "" || target.DeploymentName != strings.TrimSpace(target.DeploymentName) ||
			target.Description != strings.TrimSpace(target.Description) ||
			!digestPattern.MatchString(target.ConfigDigest) {
			return fmt.Errorf("invalid deployment-scoped evaluation target %q", target.TargetID)
		}
		if target.TargetID != deploymentTargetID(target.DeploymentID, target.Mixture.Mixture.ID) {
			return fmt.Errorf("deployment-scoped evaluation target %q has an invalid identity", target.TargetID)
		}
		if _, duplicate := seenDeploymentTargets[target.TargetID]; duplicate {
			return fmt.Errorf("duplicate deployment-scoped evaluation target %q", target.TargetID)
		}
		seenDeploymentTargets[target.TargetID] = struct{}{}
		if err := validateServerOrigin(target.RouterAPIURL); err != nil || target.RouterAPIURL == "" {
			return fmt.Errorf("deployment-scoped Router origin for %q is invalid", target.TargetID)
		}
		if err := validateServerOrigin(target.EnvoyURL); err != nil || target.EnvoyURL == "" {
			return fmt.Errorf("deployment-scoped Envoy origin for %q is invalid", target.TargetID)
		}
		if err := validateEndpointCredentialBindings(
			target.RouterAPIURL, target.EnvoyURL, options.RouterAPIKey, options.EnvoyAPIKey,
		); err != nil {
			return fmt.Errorf("deployment-scoped evaluation credentials for %q: %w", target.TargetID, err)
		}
		if err := validateMixtureTargetSnapshot(target.Mixture); err != nil {
			return err
		}
	}
	endpoints := []struct {
		name     string
		endpoint *ServiceEndpoint
	}{
		{"agent_task_ledger", options.AgentTaskLedger},
		{"fault_recovery_ledger", options.FaultRecoveryLedger},
		{"hard_policy_ledger", options.HardPolicyLedger},
		{"production_experiment_ledger", options.ProductionExperimentLedger},
	}
	for _, item := range endpoints {
		if err := validateServiceEndpoint(item.name, item.endpoint); err != nil {
			return err
		}
	}
	if err := validateDistinctTargetCredentials(map[string]*SecretRef{
		"router_api_key":               options.RouterAPIKey,
		"envoy_api_key":                options.EnvoyAPIKey,
		"agent_task_ledger":            endpointSecretRef(options.AgentTaskLedger),
		"fault_recovery_ledger":        endpointSecretRef(options.FaultRecoveryLedger),
		"hard_policy_ledger":           endpointSecretRef(options.HardPolicyLedger),
		"production_experiment_ledger": endpointSecretRef(options.ProductionExperimentLedger),
	}); err != nil {
		return err
	}
	return nil
}

func validateMixtureTargetSnapshot(mixture MixtureTargetSnapshot) error {
	if err := validateMixtureContract(&mixture.Mixture); err != nil {
		return err
	}
	if mixture.ConfigDigest != "" && !digestPattern.MatchString(mixture.ConfigDigest) {
		return fmt.Errorf("mixture config digest is invalid")
	}
	return validateTargetContract(
		nil, nil, mixture.Mixture.ModelArms, mixture.BackendTopologyDigest,
	)
}

func validateRegistryOrigins(
	routerAPIURL, envoyURL string,
	options RegistryOptions,
) error {
	if err := validateServerOrigin(routerAPIURL); err != nil {
		return fmt.Errorf("router evaluation target: %w", err)
	}
	if err := validateServerOrigin(envoyURL); err != nil {
		return fmt.Errorf("envoy evaluation target: %w", err)
	}
	if err := validateEndpointCredentialBindings(
		routerAPIURL, envoyURL, options.RouterAPIKey, options.EnvoyAPIKey,
	); err != nil {
		return fmt.Errorf("evaluation target credentials: %w", err)
	}
	if err := validateDistinctLedgerOrigins(routerAPIURL, envoyURL, options); err != nil {
		return err
	}
	for _, target := range options.DeploymentTargets {
		if err := validateDistinctLedgerOrigins(target.RouterAPIURL, target.EnvoyURL, options); err != nil {
			return fmt.Errorf("deployment-scoped evaluation target %q: %w", target.TargetID, err)
		}
	}
	return nil
}

func validateDistinctLedgerOrigins(routerAPIURL, envoyURL string, options RegistryOptions) error {
	owners := make(map[string]string, 6)
	for name, origin := range map[string]string{"router": routerAPIURL, "envoy": envoyURL} {
		if origin == "" {
			continue
		}
		key, err := serverOriginKey(origin)
		if err != nil {
			return fmt.Errorf("evaluation service origin %s: %w", name, err)
		}
		owners[key] = name
	}
	for name, endpoint := range map[string]*ServiceEndpoint{
		"agent_task_ledger":            options.AgentTaskLedger,
		"fault_recovery_ledger":        options.FaultRecoveryLedger,
		"hard_policy_ledger":           options.HardPolicyLedger,
		"production_experiment_ledger": options.ProductionExperimentLedger,
	} {
		if endpoint == nil {
			continue
		}
		key, err := serverOriginKey(endpoint.URL)
		if err != nil {
			return fmt.Errorf("evaluation service origin %s: %w", name, err)
		}
		if owner, duplicate := owners[key]; duplicate {
			return fmt.Errorf("evaluation service origins %s and %s must be distinct", owner, name)
		}
		owners[key] = name
	}
	return nil
}

func emptyRegistry() *Registry {
	return &Registry{
		tracks:         make(map[TrackID]CatalogTrack),
		suites:         make(map[string]CatalogSuite),
		executors:      make(map[string]executorContract),
		targets:        make(map[string]targetDefinition),
		changeProfiles: make(map[ChangeProfile]CatalogChangeProfile),
	}
}

func (registry *Registry) registerCatalogDefinitions(options RegistryOptions) error {
	if err := validateReleaseProfileDefinitions(releaseProfileDefinitions(), releaseGateDefinitions()); err != nil {
		return fmt.Errorf("invalid release gate contract: %w", err)
	}
	for _, contract := range builtinExecutorContracts() {
		if err := registry.registerExecutor(contract); err != nil {
			return err
		}
	}
	for _, track := range builtinTracks() {
		registry.tracks[track.ID] = track
	}
	for _, suite := range builtinSuitesFor(options) {
		if err := registry.registerSuite(suite); err != nil {
			return err
		}
	}
	if err := registry.registerInstalledSuites(options.InstalledSuites); err != nil {
		return err
	}
	profiles := builtinChangeProfiles()
	if err := validateCampaignCatalogContracts(registry, profiles); err != nil {
		return err
	}
	for _, profile := range profiles {
		registry.changeProfiles[profile.ID] = profile
	}
	return nil
}

func (registry *Registry) registerInstalledSuites(values []CatalogSuite) error {
	suites := append([]CatalogSuite(nil), values...)
	sort.Slice(suites, func(left, right int) bool { return suites[left].ID < suites[right].ID })
	for _, suite := range suites {
		valid := portableSuiteIDPattern.MatchString(suite.ID) &&
			validNormalizedSuiteExecutors(suite, registry.executors) &&
			len(suite.TrackIDs) > 0 && canonicalTrackOrder(suite.TrackIDs) &&
			digestPattern.MatchString(suite.Revision) &&
			evidenceLevelRank(suite.EvidenceLevel) >= 0 && suite.CaseCount > 0
		if !valid {
			return fmt.Errorf("invalid installed normalized suite %q", suite.ID)
		}
		if err := registry.registerSuite(suite); err != nil {
			return err
		}
	}
	return nil
}

func (registry *Registry) registerRecordedTargets(installedSuites bool) error {
	healthy := true
	fixture := targetDefinition{Public: CatalogTarget{
		ID: "fixture", Name: "Built-in evaluation sample", Kind: "builtin-fixture",
		Description:       "A small deterministic replay for checking the full evaluation workflow without calling a live system.",
		Modes:             []Mode{ModeReplay},
		AcceptedExecutors: map[Mode][]string{ModeReplay: {fixtureReplayExecutorID}},
		EvidenceLevel:     "E0", Healthy: &healthy,
		Labels: map[string]string{"execution": "local", "network": "none"},
	}, Contract: targetContract{
		ExecutionProfile: targetProfileRecorded, PolicySnapshot: policySnapshotFixture,
		TrackRequirements: recordedTrackRequirements(),
	}, ConfigDigest: emptyConfigDigest}
	if err := registry.registerTarget(fixture); err != nil {
		return err
	}
	benchmarkSource := targetDefinition{Public: CatalogTarget{
		ID: "benchmark-source", Name: "Imported benchmark results",
		Kind:              "normalized-benchmark-source",
		Description:       "Replay saved results from pinned benchmark revisions. This evaluates imported observations, not the live system.",
		Modes:             []Mode{ModeReplay},
		AcceptedExecutors: map[Mode][]string{ModeReplay: {normalizedSuiteExecutorID}},
		Healthy:           &installedSuites,
		Labels: map[string]string{
			"execution": "recorded-source", "identity": "suite-revision-bound", "network": "none",
		},
	}, Contract: targetContract{
		ExecutionProfile: targetProfileRecorded, PolicySnapshot: policySnapshotNormalized,
		TrackRequirements: recordedTrackRequirements(),
	}, ConfigDigest: emptyConfigDigest}
	return registry.registerTarget(benchmarkSource)
}

func (registry *Registry) registerMixtureTargets(
	routerAPIURL, envoyURL string,
	options RegistryOptions,
) error {
	if len(options.DeploymentTargets) > 0 {
		for _, snapshot := range options.DeploymentTargets {
			target := deploymentMixtureTargetDefinition(snapshot, options)
			if err := registry.registerReadyMixtureTarget(target, snapshot.Mixture.Ready); err != nil {
				return err
			}
		}
		return nil
	}
	effectiveRouterURL := routerAPIURL
	if options.RouterAuthRequired && options.RouterAPIKey == nil {
		effectiveRouterURL = ""
	}
	for _, snapshot := range options.Mixtures {
		target := mixtureTargetDefinition(
			snapshot.Mixture.ID, "", snapshot,
			effectiveRouterURL, envoyURL, options,
		)
		if err := registry.registerReadyMixtureTarget(target, snapshot.Ready); err != nil {
			return err
		}
	}
	return nil
}

func mixtureTargetDefinition(
	targetID, deploymentName string,
	snapshot MixtureTargetSnapshot,
	routerAPIURL, envoyURL string,
	options RegistryOptions,
) targetDefinition {
	mixture := copyManifestMixture(&snapshot.Mixture)
	healthy := false
	labels := map[string]string{
		"capabilities": "mixture-bound", "credentials": "server-brokered", "model_arms": "server-owned",
	}
	if options.RouterAuthRequired && options.RouterAPIKey == nil {
		labels["router_auth"] = "dedicated-evaluation-credential-unavailable"
	} else if options.RouterAPIKey != nil {
		labels["router_auth"] = "dedicated-evaluation-credential-configured"
	}
	description := mixture.RecipeDescription
	name := mixture.EntrypointModel
	if deploymentName != "" {
		labels["deployment"] = deploymentName
		name = deploymentName + " · " + mixture.EntrypointModel
	}
	if description == "" {
		description = "Evaluate this routing recipe and its model pool together as one system."
	}
	topologyDigest := snapshot.BackendTopologyDigest
	if !snapshot.Ready {
		topologyDigest = ""
	}
	return targetDefinition{
		Public: CatalogTarget{
			ID: targetID, Name: name, Kind: "mixture-of-models",
			Description: description, Modes: []Mode{ModeReplay, ModeLive},
			AcceptedExecutors: map[Mode][]string{
				ModeReplay: {momReplayExecutorID},
				ModeLive:   {liveRuntimeExecutorID, normalizedSuiteLiveExecutorID},
			},
			Healthy: &healthy, Labels: labels, Mixture: catalogMixtureFromManifest(mixture),
		},
		Contract: targetContract{
			ExecutionProfile: targetProfileRuntime, PolicySnapshot: policySnapshotRuntime,
			TrackRequirements: runtimeTrackRequirements(),
		},
		RouterAPIURL: routerAPIURL, EnvoyURL: envoyURL,
		RouterAPIKey: copySecretRef(options.RouterAPIKey), EnvoyAPIKey: copySecretRef(options.EnvoyAPIKey),
		AgentTaskLedger:            copyServiceEndpoint(options.AgentTaskLedger),
		FaultRecoveryLedger:        copyServiceEndpoint(options.FaultRecoveryLedger),
		HardPolicyLedger:           copyServiceEndpoint(options.HardPolicyLedger),
		ProductionExperimentLedger: copyServiceEndpoint(options.ProductionExperimentLedger),
		Mixture:                    mixture, ConfigDigest: mixtureConfigDigest(snapshot, options.DefaultConfigDigest),
		BackendTopologyDigest: topologyDigest,
	}
}

func deploymentMixtureTargetDefinition(
	snapshot DeploymentTargetSnapshot,
	options RegistryOptions,
) targetDefinition {
	routerAPIURL := snapshot.RouterAPIURL
	if options.RouterAuthRequired && options.RouterAPIKey == nil {
		routerAPIURL = ""
	}
	target := mixtureTargetDefinition(
		snapshot.TargetID, snapshot.DeploymentName,
		snapshot.Mixture, routerAPIURL, snapshot.EnvoyURL, options,
	)
	target.ConfigDigest = snapshot.ConfigDigest
	return target
}

func mixtureConfigDigest(snapshot MixtureTargetSnapshot, fallback string) string {
	if digestPattern.MatchString(snapshot.ConfigDigest) {
		return snapshot.ConfigDigest
	}
	if digestPattern.MatchString(fallback) {
		return fallback
	}
	return emptyConfigDigest
}

func (registry *Registry) registerReadyMixtureTarget(target targetDefinition, ready bool) error {
	if err := registry.registerTarget(target); err != nil {
		return err
	}
	registered, _ := registry.target(target.Public.ID)
	available := ready && len(registered.Public.TrackIDs) > 0
	registered.Public.Healthy = &available
	registry.targets[target.Public.ID] = registered
	return nil
}
