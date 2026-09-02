package evaluationplane

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadEvaluationDeploymentRegistryFreezesDistinctTargetSnapshots(t *testing.T) {
	root := t.TempDir()
	baselineConfig := []byte(modelArmTestYAML)
	candidateConfig := []byte(strings.Replace(
		modelArmTestYAML,
		"routing:\n  modelCards:",
		"routing:\n  strategy: confidence\n  modelCards:",
		1,
	))
	writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{
		{
			ID: "baseline", Name: "Baseline", Description: "Current production recipe",
			ConfigFile: "baseline.yaml", RouterOrigin: "https://baseline-router.internal",
			EnvoyOrigin: "https://baseline-envoy.internal",
		},
		{
			ID: "candidate", Name: "Candidate", Description: "Candidate recipe",
			ConfigFile: "candidate.yaml", RouterOrigin: "https://candidate-router.internal",
			EnvoyOrigin: "https://candidate-envoy.internal",
		},
	}, map[string][]byte{"baseline.yaml": baselineConfig, "candidate.yaml": candidateConfig})

	targets, err := LoadEvaluationDeploymentRegistry(root, "")
	if err != nil {
		t.Fatalf("LoadEvaluationDeploymentRegistry: %v", err)
	}
	if len(targets) != 2 {
		t.Fatalf("targets=%d, want 2", len(targets))
	}
	baseline, candidate := targets[0], targets[1]
	if baseline.TargetID == candidate.TargetID ||
		baseline.Mixture.Mixture.ID != candidate.Mixture.Mixture.ID ||
		baseline.Mixture.Mixture.RecipeName != candidate.Mixture.Mixture.RecipeName {
		t.Fatalf("deployment target identity is not scoped over one logical subject: %+v %+v", baseline, candidate)
	}
	if baseline.ConfigDigest != digestBytes(baselineConfig) ||
		candidate.ConfigDigest != digestBytes(candidateConfig) ||
		baseline.ConfigDigest == candidate.ConfigDigest {
		t.Fatalf("deployment config digests are not bound to exact bytes: baseline=%q candidate=%q", baseline.ConfigDigest, candidate.ConfigDigest)
	}
	if baseline.Mixture.ConfigDigest != baseline.ConfigDigest ||
		candidate.Mixture.ConfigDigest != candidate.ConfigDigest {
		t.Fatalf("Mixture target snapshots lost deployment config identity")
	}
}

func TestEvaluationDeploymentRegistryFailsClosed(t *testing.T) {
	validDefinition := evaluationDeploymentDefinition{
		ID: "baseline", Name: "Baseline", ConfigFile: "config.yaml",
		RouterOrigin: "https://router.internal", EnvoyOrigin: "https://envoy.internal",
	}
	tests := []struct {
		name    string
		prepare func(*testing.T, string)
		match   string
	}{
		{
			name: "unknown registry field", match: "unknown field",
			prepare: func(t *testing.T, root string) {
				writeRawDeploymentRegistry(t, root, `{"schema_version":"evaluation-deployments.v1","deployments":[],"extra":true}`)
			},
		},
		{
			name: "empty deployments", match: "at least one deployment",
			prepare: func(t *testing.T, root string) {
				writeRawDeploymentRegistry(t, root, `{"schema_version":"evaluation-deployments.v1","deployments":[]}`)
			},
		},
		{
			name: "unknown deployment field", match: "unknown field",
			prepare: func(t *testing.T, root string) {
				writeRawDeploymentRegistry(t, root, `{"schema_version":"evaluation-deployments.v1","deployments":[{"id":"baseline","name":"Baseline","config_file":"config.yaml","router_origin":"https://router.internal","envoy_origin":"https://envoy.internal","secret":"forbidden"}]}`)
			},
		},
		{
			name: "duplicate deployment", match: "duplicate evaluation deployment id",
			prepare: func(t *testing.T, root string) {
				writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{validDefinition, validDefinition}, map[string][]byte{"config.yaml": []byte(modelArmTestYAML)})
			},
		},
		{
			name: "traversal", match: "config_file",
			prepare: func(t *testing.T, root string) {
				definition := validDefinition
				definition.ConfigFile = "../config.yaml"
				writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{definition}, nil)
			},
		},
		{
			name: "noncanonical origin", match: "router_origin",
			prepare: func(t *testing.T, root string) {
				definition := validDefinition
				definition.RouterOrigin = "https://router.internal/"
				writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{definition}, map[string][]byte{"config.yaml": []byte(modelArmTestYAML)})
			},
		},
		{
			name: "unsafe public label", match: "portable",
			prepare: func(t *testing.T, root string) {
				definition := validDefinition
				definition.Name = "https://private.internal/config"
				writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{definition}, map[string][]byte{"config.yaml": []byte(modelArmTestYAML)})
			},
		},
		{
			name: "config symlink", match: "symlink",
			prepare: func(t *testing.T, root string) {
				outside := filepath.Join(t.TempDir(), "outside.yaml")
				if err := os.WriteFile(outside, []byte(modelArmTestYAML), 0o600); err != nil {
					t.Fatal(err)
				}
				writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{validDefinition}, nil)
				if err := os.Symlink(outside, filepath.Join(root, "config.yaml")); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "invalid config file", match: "load evaluation deployment",
			prepare: func(t *testing.T, root string) {
				writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{validDefinition}, map[string][]byte{
					"config.yaml": []byte("version: [invalid"),
				})
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			test.prepare(t, root)
			if _, err := LoadEvaluationDeploymentRegistry(root, ""); err == nil ||
				!strings.Contains(err.Error(), test.match) {
				t.Fatalf("error=%v, want %q", err, test.match)
			}
		})
	}
}

func TestRegistryRejectsDuplicateResultingDeploymentTargetIDs(t *testing.T) {
	root := t.TempDir()
	writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{{
		ID: "baseline", Name: "Baseline", ConfigFile: "config.yaml",
		RouterOrigin: "https://router.internal", EnvoyOrigin: "https://envoy.internal",
	}}, map[string][]byte{"config.yaml": []byte(modelArmTestYAML)})
	targets, err := LoadEvaluationDeploymentRegistry(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRegistry("", "", RegistryOptions{
		DeploymentTargets: []DeploymentTargetSnapshot{targets[0], targets[0]},
	}); err == nil || !strings.Contains(err.Error(), "duplicate deployment-scoped evaluation target") {
		t.Fatalf("duplicate resulting target IDs error=%v", err)
	}
}

func TestEvaluationDeploymentRegistryRejectsSymlinkRootAndRegistry(t *testing.T) {
	realRoot := t.TempDir()
	writeRawDeploymentRegistry(t, realRoot, `{"schema_version":"evaluation-deployments.v1","deployments":[]}`)
	linkedRoot := filepath.Join(t.TempDir(), "registry-link")
	if err := os.Symlink(realRoot, linkedRoot); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadEvaluationDeploymentRegistry(linkedRoot, ""); err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("symlink root error=%v", err)
	}

	root := t.TempDir()
	linkedRegistry := filepath.Join(root, evaluationDeploymentRegistryFile)
	if err := os.Symlink(filepath.Join(realRoot, evaluationDeploymentRegistryFile), linkedRegistry); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadEvaluationDeploymentRegistry(root, ""); err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("symlink registry error=%v", err)
	}
}

func TestDeploymentCatalogIsPrivateAndManifestUsesTargetConfigDigest(t *testing.T) {
	root := t.TempDir()
	configBytes := []byte(modelArmTestYAML)
	writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{{
		ID: "candidate", Name: "Candidate", Description: "Review candidate",
		ConfigFile: "private/config.yaml", RouterOrigin: "https://router.private.internal",
		EnvoyOrigin: "https://envoy.private.internal",
	}}, map[string][]byte{"private/config.yaml": configBytes})
	defaultConfig := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(defaultConfig, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		DataDir: filepath.Join(t.TempDir(), "store"), PythonPath: "python3",
		ConfigPath: defaultConfig, DeploymentsDir: root, CodeRevision: testSourceRevision,
		Process: &controlledProcess{},
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatal(err)
	}
	var target CatalogTarget
	for _, item := range catalog.Targets {
		if item.Kind == "mixture-of-models" {
			target = item
			break
		}
	}
	if target.ID == "" || target.Labels["deployment"] != "Candidate" || target.ID == target.Mixture.ID {
		t.Fatalf("deployment-scoped catalog target=%+v", target)
	}
	encoded, err := json.Marshal(catalog)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{
		"router.private.internal", "envoy.private.internal", "private/config.yaml", root, "Review candidate",
	} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("catalog leaked private deployment value %q: %s", forbidden, encoded)
		}
	}

	request := CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "deployment config freeze",
		SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
		Mode: ModeLive, TargetID: target.ID, ChangeProfile: "recipe",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	manifest, _, err := service.readDurableManifest(run.ID)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.ConfigDigest != digestBytes(configBytes) || manifest.Target.ID != target.ID ||
		manifest.Target.Mixture.ID != target.Mixture.ID {
		t.Fatalf("manifest did not freeze target-specific config: %+v", manifest)
	}
	configPath := filepath.Join(root, "private", "config.yaml")
	if err := os.WriteFile(configPath, append(configBytes, []byte("\n# byte-only deployment revision\n")...), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err == nil ||
		!strings.Contains(err.Error(), "config digest") {
		t.Fatalf("start after deployment config byte drift error=%v, want target-specific config digest rejection", err)
	}
}

func TestLoadedDeploymentsMakeOneLogicalMixtureControlledPairAddressable(t *testing.T) {
	root := t.TempDir()
	writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{
		{
			ID: "baseline", Name: "Baseline", ConfigFile: "baseline.yaml",
			RouterOrigin: "https://baseline-router.internal", EnvoyOrigin: "https://baseline-envoy.internal",
		},
		{
			ID: "candidate", Name: "Candidate", ConfigFile: "candidate.yaml",
			RouterOrigin: "https://candidate-router.internal", EnvoyOrigin: "https://candidate-envoy.internal",
		},
	}, map[string][]byte{
		"baseline.yaml":  []byte(modelArmTestYAML),
		"candidate.yaml": []byte(modelArmTestYAML),
	})
	deployments, err := LoadEvaluationDeploymentRegistry(root, "")
	if err != nil {
		t.Fatal(err)
	}
	registry, err := NewRegistry("", "", RegistryOptions{DeploymentTargets: deployments})
	if err != nil {
		t.Fatal(err)
	}
	baseline, baselineOK := registry.target(deployments[0].TargetID)
	candidate, candidateOK := registry.target(deployments[1].TargetID)
	if !baselineOK || !candidateOK || baseline.Public.ID == candidate.Public.ID ||
		baseline.Mixture.ID != candidate.Mixture.ID || baseline.Mixture.RecipeName != candidate.Mixture.RecipeName {
		t.Fatalf("loaded deployments do not preserve one logical Mixture subject: baseline=%+v candidate=%+v", baseline, candidate)
	}
	manifestFor := func(target targetDefinition) RunManifest {
		return RunManifest{
			Mode: ModeLive, SuiteIDs: []string{"live-mom-core"},
			TrackIDs:       []TrackID{"routing", "model_pool", "joint"},
			SuiteExecutors: map[string]string{"live-mom-core": liveRuntimeExecutorID},
			ConfigDigest:   target.ConfigDigest,
			Target: ManifestTarget{
				SchemaVersion: SchemaVersion, ID: target.Public.ID, Kind: target.Public.Kind,
				RouterAPIURL: target.RouterAPIURL, EnvoyURL: target.EnvoyURL,
				RouterAPIKey: copySecretRef(target.RouterAPIKey), EnvoyAPIKey: copySecretRef(target.EnvoyAPIKey),
				AgentTaskLedger:            copyServiceEndpoint(target.AgentTaskLedger),
				FaultRecoveryLedger:        copyServiceEndpoint(target.FaultRecoveryLedger),
				HardPolicyLedger:           copyServiceEndpoint(target.HardPolicyLedger),
				ProductionExperimentLedger: copyServiceEndpoint(target.ProductionExperimentLedger),
				Mixture:                    copyManifestMixture(target.Mixture),
				BackendTopologyDigest:      target.BackendTopologyDigest,
			},
		}
	}
	baselineManifest, candidateManifest := manifestFor(baseline), manifestFor(candidate)
	if err := validateControlledPairRegistryTargets(registry, baselineManifest, candidateManifest); err != nil {
		t.Fatalf("two deployment manifests did not resolve against their own frozen targets: %v", err)
	}
	if err := validateControlledPairAddressability(baselineManifest, candidateManifest); err != nil {
		t.Fatalf("two loaded deployment origins did not make G3 addressable: %v", err)
	}
	candidateConfigDrift := candidateManifest
	candidateConfigDrift.ConfigDigest = digestString("candidate-config-drift")
	if err := validateControlledPairRegistryTargets(registry, baselineManifest, candidateConfigDrift); err == nil ||
		!strings.Contains(err.Error(), "candidate target") {
		t.Fatalf("candidate target-specific config drift error=%v", err)
	}
	candidateManifest.Target.RouterAPIURL = baselineManifest.Target.RouterAPIURL
	if err := validateControlledPairAddressability(baselineManifest, candidateManifest); err == nil ||
		!strings.Contains(err.Error(), "distinct server-owned Router origins") {
		t.Fatalf("shared Router origin error=%v", err)
	}
	candidateManifest = manifestFor(candidate)
	candidateManifest.Target.RouterAPIURL = baselineManifest.Target.RouterAPIURL + ":443"
	if err := validateControlledPairAddressability(baselineManifest, candidateManifest); err == nil ||
		!strings.Contains(err.Error(), "distinct server-owned Router origins") {
		t.Fatalf("shared effective Router origin error=%v", err)
	}
	candidateManifest = manifestFor(candidate)
	candidateManifest.Target.EnvoyURL = baselineManifest.Target.EnvoyURL
	if err := validateControlledPairAddressability(baselineManifest, candidateManifest); err == nil ||
		!strings.Contains(err.Error(), "distinct server-owned Envoy origins") {
		t.Fatalf("shared Envoy origin error=%v", err)
	}
	candidateManifest = manifestFor(candidate)
	candidateManifest.Target.EnvoyURL = baselineManifest.Target.EnvoyURL + ":443"
	if err := validateControlledPairAddressability(baselineManifest, candidateManifest); err == nil ||
		!strings.Contains(err.Error(), "distinct server-owned Envoy origins") {
		t.Fatalf("shared effective Envoy origin error=%v", err)
	}
}

func TestZeroDeploymentDirectoryPreservesDefaultRuntimeTarget(t *testing.T) {
	snapshot, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "")
	if err != nil {
		t.Fatal(err)
	}
	registry, err := NewRegistry(
		"https://router.internal", "https://envoy.internal",
		RegistryOptions{Mixtures: snapshot.Mixtures, DefaultConfigDigest: snapshot.ConfigDigest},
	)
	if err != nil {
		t.Fatal(err)
	}
	target, ok := registry.target(snapshot.Mixtures[0].Mixture.ID)
	if !ok || target.Public.ID != target.Mixture.ID || target.ConfigDigest != snapshot.ConfigDigest {
		t.Fatalf("default zero-registry target changed: %+v", target)
	}
}

func writeDeploymentRegistryFixture(
	t *testing.T,
	root string,
	deployments []evaluationDeploymentDefinition,
	configs map[string][]byte,
) {
	t.Helper()
	for relative, data := range configs {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	data, err := json.Marshal(evaluationDeploymentRegistry{
		SchemaVersion: evaluationDeploymentRegistryVersion,
		Deployments:   deployments,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, evaluationDeploymentRegistryFile), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeRawDeploymentRegistry(t *testing.T, root, data string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, evaluationDeploymentRegistryFile), []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
}
