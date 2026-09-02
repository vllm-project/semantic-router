package evaluationplane

import "strings"

type registryConstructor func(
	routerAPIURL, envoyURL string,
	registryOptions ...RegistryOptions,
) (*Registry, error)

// runtimeRegistrySource is the single frozen input boundary used both for
// startup admission and for every live catalog snapshot. Mutable suite/config
// files are reloaded by snapshot; credentials and service endpoints remain the
// server-owned values accepted when the Service was created.
type runtimeRegistrySource struct {
	suiteStorePath             string
	configPath                 string
	deploymentsDir             string
	routerAPIURL               string
	envoyURL                   string
	routerAPIKeyEnv            string
	envoyAPIKeyEnv             string
	agentTaskLedger            *ServiceEndpoint
	faultRecoveryLedger        *ServiceEndpoint
	hardPolicyLedger           *ServiceEndpoint
	productionExperimentLedger *ServiceEndpoint
	routerAuthRequired         bool
	constructor                registryConstructor
}

func newRuntimeRegistrySource(
	options *Options,
	suiteStorePath string,
	routerAuthRequired bool,
	constructor registryConstructor,
) runtimeRegistrySource {
	return runtimeRegistrySource{
		suiteStorePath:             suiteStorePath,
		configPath:                 options.ConfigPath,
		deploymentsDir:             strings.TrimSpace(options.DeploymentsDir),
		routerAPIURL:               options.RouterAPIURL,
		envoyURL:                   options.EnvoyURL,
		routerAPIKeyEnv:            strings.TrimSpace(options.RouterAPIKeyEnv),
		envoyAPIKeyEnv:             strings.TrimSpace(options.EnvoyAPIKeyEnv),
		agentTaskLedger:            copyServiceEndpoint(options.AgentTaskLedger),
		faultRecoveryLedger:        copyServiceEndpoint(options.FaultRecoveryLedger),
		hardPolicyLedger:           copyServiceEndpoint(options.HardPolicyLedger),
		productionExperimentLedger: copyServiceEndpoint(options.ProductionExperimentLedger),
		routerAuthRequired:         routerAuthRequired,
		constructor:                constructor,
	}
}

// Catalog returns the immutable catalog assembled from the active runtime,
// installed suites, and deployment registry.
func (s *Service) Catalog() (Catalog, error) {
	release, err := s.beginOperation()
	if err != nil {
		return Catalog{}, err
	}
	defer release()
	registry, err := s.registrySnapshot()
	if err != nil {
		return Catalog{}, err
	}
	return registry.Catalog(), nil
}

func (s *Service) registrySnapshot() (*Registry, error) {
	return s.registrySource.snapshot()
}

func (source runtimeRegistrySource) snapshot() (*Registry, error) {
	// The model runtime revision is not available from the Router config. Keep it
	// unset instead of conflating it with the evaluation source revision.
	snapshot, err := LoadModelArmSnapshot(source.configPath, "")
	if err != nil {
		return nil, err
	}
	installedSuites, err := loadInstalledCatalogSuites(source.suiteStorePath)
	if err != nil {
		return nil, err
	}
	deploymentTargets, err := LoadEvaluationDeploymentRegistry(source.deploymentsDir, "")
	if err != nil {
		return nil, err
	}
	mixtures := snapshot.Mixtures
	if len(deploymentTargets) > 0 {
		mixtures = nil
	}
	registry, err := source.constructor(source.routerAPIURL, source.envoyURL, RegistryOptions{
		RouterAPIKey: configuredRuntimeSecretRef(
			source.routerAPIURL, deploymentTargets, source.routerAPIKeyEnv, true,
		),
		EnvoyAPIKey: configuredRuntimeSecretRef(
			source.envoyURL, deploymentTargets, source.envoyAPIKeyEnv, false,
		),
		AgentTaskLedger:            copyServiceEndpoint(source.agentTaskLedger),
		FaultRecoveryLedger:        copyServiceEndpoint(source.faultRecoveryLedger),
		HardPolicyLedger:           copyServiceEndpoint(source.hardPolicyLedger),
		ProductionExperimentLedger: copyServiceEndpoint(source.productionExperimentLedger),
		Mixtures:                   mixtures,
		DeploymentTargets:          deploymentTargets,
		DefaultConfigDigest:        snapshot.ConfigDigest,
		RouterAuthRequired:         source.routerAuthRequired,
		InstalledSuites:            installedSuites,
	})
	if err != nil {
		return nil, err
	}
	return registry, nil
}

func configuredSecretRef(endpointURL, envName string) *SecretRef {
	if strings.TrimSpace(endpointURL) == "" || strings.TrimSpace(envName) == "" {
		return nil
	}
	return &SecretRef{SchemaVersion: SchemaVersion, Env: strings.TrimSpace(envName)}
}

func configuredRuntimeSecretRef(
	defaultOrigin string,
	deployments []DeploymentTargetSnapshot,
	envName string,
	router bool,
) *SecretRef {
	if strings.TrimSpace(envName) == "" {
		return nil
	}
	if strings.TrimSpace(defaultOrigin) != "" {
		return configuredSecretRef(defaultOrigin, envName)
	}
	for _, deployment := range deployments {
		origin := deployment.EnvoyURL
		if router {
			origin = deployment.RouterAPIURL
		}
		if origin != "" {
			return &SecretRef{SchemaVersion: SchemaVersion, Env: strings.TrimSpace(envName)}
		}
	}
	return nil
}
