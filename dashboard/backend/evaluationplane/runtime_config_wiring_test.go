package evaluationplane

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestAuthenticatedRegistryUsesDedicatedRouterEvaluationCredential(t *testing.T) {
	mixture := catalogTestMixtureSnapshot(
		[]ModelArm{catalogTestArm("deep", []string{"text"}), catalogTestArm("fast", []string{"text"})},
		catalogTopologyDigest,
	)
	registry, err := NewRegistry("https://router.internal", "https://envoy.internal", RegistryOptions{
		RouterAPIKey:       &SecretRef{SchemaVersion: SchemaVersion, Env: "ROUTER_EVAL_TOKEN"},
		RouterAuthRequired: true,
		Mixtures:           []MixtureTargetSnapshot{mixture},
	})
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	target := registry.targets[mixture.Mixture.ID]
	if target.RouterAPIURL != "https://router.internal" || target.RouterAPIKey == nil ||
		target.RouterAPIKey.Env != "ROUTER_EVAL_TOKEN" ||
		!containsTrack(target.Public.TrackIDs, "routing") ||
		target.Public.Labels["router_auth"] != "dedicated-evaluation-credential-configured" {
		t.Fatalf("authenticated evaluation target = %+v", target)
	}
}

func TestDedicatedRouterCredentialIsServerBrokeredAndNeverWorkerVisible(t *testing.T) {
	t.Setenv("ROUTER_EVAL_TOKEN", "router-evaluation-secret")
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		authorization = request.Header.Get("Authorization")
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"selected_model":"model-fast","selection_status":"selected","selection_method":"static"}`))
	}))
	t.Cleanup(server.Close)

	manifest := RunManifest{Concurrency: 1, SampleLimit: 1, Target: ManifestTarget{
		RouterAPIURL: server.URL,
		RouterAPIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "ROUTER_EVAL_TOKEN"},
		Mixture:      brokerTestMixture(),
	}}
	process := NewCommandProcess("python3")
	process.routerAPIKeyEnv = "ROUTER_EVAL_TOKEN"
	credentials, err := process.brokerCredentials(manifest)
	if err != nil || credentials.router != "router-evaluation-secret" {
		t.Fatalf("Router broker credentials = %#v err=%v", credentials, err)
	}
	broker := newWorkerHTTPBroker(manifest, credentials)
	broker.models[manifest.Target.Mixture.EntrypointModel] = manifest.Target.Mixture.RecipeName
	broker.modelsValid = true
	payload := json.RawMessage(`{"model":"virtual-entrypoint","messages":[{"role":"user","content":"route me"}],"evaluate_all_signals":true}`)
	response := broker.execute(context.Background(), workerBrokerRequest{
		ID: 1, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
		CaseID: "case-1", AttemptID: "attempt-1", Payload: payload, TimeoutMS: 1_000,
	})
	if !response.Success || authorization != "Bearer router-evaluation-secret" {
		t.Fatalf("Router broker response=%+v authorization=%q", response, authorization)
	}
	environment := isolatedWorkerEnvironment(t.TempDir())
	if value, present := environmentValue(environment, "ROUTER_EVAL_TOKEN"); present || value != "" {
		t.Fatalf("Router evaluation secret leaked to worker environment: value=%q present=%v", value, present)
	}
	encoded, err := json.Marshal(manifest)
	if err != nil || strings.Contains(string(encoded), "router-evaluation-secret") {
		t.Fatalf("manifest leaked Router evaluation secret: %s err=%v", encoded, err)
	}
}

func TestDedicatedRouterCredentialConfigurationFailsClosed(t *testing.T) {
	provider := staticCredentialProvider{token: "dashboard-management-secret"}
	for name, values := range map[string][3]string{
		"missing value":         {"ROUTER_EVAL_TOKEN", "", "unavailable"},
		"management env":        {routerManagementCredentialEnv, "different-secret", "management credential"},
		"same credential value": {"ROUTER_EVAL_TOKEN", "dashboard-management-secret", "distinct"},
	} {
		t.Run(name, func(t *testing.T) {
			ref, value, match := values[0], values[1], values[2]
			if value == "" {
				_ = os.Unsetenv(ref)
			} else {
				t.Setenv(ref, value)
			}
			_, err := resolveRouterAuthentication(ref, provider)
			if err == nil || !strings.Contains(err.Error(), match) {
				t.Fatalf("resolveRouterAuthentication(%q) error=%v, want %q", ref, err, match)
			}
		})
	}
	if required, err := resolveRouterAuthentication("", provider); err != nil || !required {
		t.Fatalf("unconfigured dedicated credential did not preserve fail-closed auth state: required=%v err=%v", required, err)
	}
	if required, err := resolveRouterAuthentication("", staticCredentialProvider{err: os.ErrNotExist}); err != nil || required {
		t.Fatalf("auth-disabled Router unexpectedly required a credential: required=%v err=%v", required, err)
	}
}

func TestRegistryRejectsSharedLedgerOriginsAndCredentialReferences(t *testing.T) {
	base := RegistryOptions{
		RouterAPIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "ROUTER_EVAL_TOKEN"},
		EnvoyAPIKey:  &SecretRef{SchemaVersion: SchemaVersion, Env: "ENVOY_EVAL_TOKEN"},
		AgentTaskLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: "https://agent-task.internal",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "AGENT_TASK_TOKEN"}, TimeoutSeconds: 30,
		},
		FaultRecoveryLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: "https://fault.internal",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "FAULT_TOKEN"}, TimeoutSeconds: 30,
		},
		HardPolicyLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: "https://policy.internal",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "POLICY_TOKEN"}, TimeoutSeconds: 30,
		},
	}
	if _, err := NewRegistry("https://router.internal", "https://envoy.internal", base); err != nil {
		t.Fatalf("distinct Registry options rejected: %v", err)
	}
	sharedOrigin := base
	sharedOrigin.HardPolicyLedger = copyServiceEndpoint(base.HardPolicyLedger)
	sharedOrigin.HardPolicyLedger.URL = "https://router.internal"
	if _, err := NewRegistry("https://router.internal", "https://envoy.internal", sharedOrigin); err == nil {
		t.Fatal("ledger sharing a Router origin was accepted")
	}
	sharedDefaultPortOrigin := base
	sharedDefaultPortOrigin.HardPolicyLedger = copyServiceEndpoint(base.HardPolicyLedger)
	sharedDefaultPortOrigin.HardPolicyLedger.URL = "https://router.internal:443"
	if _, err := NewRegistry("https://router.internal", "https://envoy.internal", sharedDefaultPortOrigin); err == nil {
		t.Fatal("ledger sharing a Router effective origin through the default port was accepted")
	}
	sharedLedgerOrigin := base
	sharedLedgerOrigin.HardPolicyLedger = copyServiceEndpoint(base.HardPolicyLedger)
	sharedLedgerOrigin.HardPolicyLedger.URL = "https://agent-task.internal:443"
	if _, err := NewRegistry("https://router.internal", "https://envoy.internal", sharedLedgerOrigin); err == nil {
		t.Fatal("ledgers sharing an effective origin through the default port were accepted")
	}
	sharedCredential := base
	sharedCredential.HardPolicyLedger = copyServiceEndpoint(base.HardPolicyLedger)
	sharedCredential.HardPolicyLedger.APIKey.Env = "ROUTER_EVAL_TOKEN"
	if _, err := NewRegistry("https://router.internal", "https://envoy.internal", sharedCredential); err == nil {
		t.Fatal("ledger sharing a Router credential ref was accepted")
	}
}

func TestServiceFreezesConfiguredProductionEndpointsAndRouterSecretRef(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(t.TempDir(), "router.yaml")
	if err := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ROUTER_EVAL_TOKEN", "router-evaluation-secret")
	service, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		RouterAPIURL: "https://router.internal", EnvoyURL: "https://envoy.internal",
		RouterAPIKeyEnv: "ROUTER_EVAL_TOKEN", CodeRevision: testSourceRevision,
		CredentialProvider: staticCredentialProvider{token: "dashboard-management-secret"},
		AgentTaskLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: "https://agent-task.internal",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "AGENT_TASK_TOKEN"}, TimeoutSeconds: 10,
		},
		FaultRecoveryLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: "https://fault.internal",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "FAULT_TOKEN"}, TimeoutSeconds: 15,
		},
		Process: &controlledProcess{},
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	registry, err := service.registrySnapshot()
	if err != nil {
		t.Fatal(err)
	}
	target, ok := registry.target(mixtureTargetID("default"))
	if !ok || target.RouterAPIKey == nil || target.RouterAPIKey.Env != "ROUTER_EVAL_TOKEN" ||
		!reflect.DeepEqual(target.AgentTaskLedger, service.registrySource.agentTaskLedger) ||
		!reflect.DeepEqual(target.FaultRecoveryLedger, service.registrySource.faultRecoveryLedger) {
		t.Fatalf("service registry target = %+v", target)
	}
	if target.AgentTaskLedger.TimeoutSeconds != 10 || target.FaultRecoveryLedger.TimeoutSeconds != 15 ||
		!containsTrack(target.Public.TrackIDs, "agentic") {
		t.Fatalf("agentic ledger runtime capabilities missing: %+v", target)
	}
}

func TestExecutionRegistryAcceptsAuthenticatedRuntimeTarget(t *testing.T) {
	target := targetDefinition{
		Public: CatalogTarget{
			ID: "mom-runtime", Kind: "mixture-of-models", Modes: []Mode{ModeLive},
			AcceptedExecutors: map[Mode][]string{ModeLive: {liveRuntimeExecutorID}},
		},
		Contract:     targetContract{ExecutionProfile: targetProfileRuntime, PolicySnapshot: policySnapshotRuntime},
		RouterAPIURL: "https://router.internal", EnvoyURL: "https://envoy.internal",
		RouterAPIKey:          &SecretRef{SchemaVersion: SchemaVersion, Env: "ROUTER_EVAL_TOKEN"},
		Mixture:               brokerTestMixture(),
		BackendTopologyDigest: digestString("topology"),
	}
	manifest := RunManifest{Target: ManifestTarget{
		RouterAPIURL: target.RouterAPIURL, EnvoyURL: target.EnvoyURL,
		RouterAPIKey: target.RouterAPIKey, Mixture: target.Mixture,
		BackendTopologyDigest: target.BackendTopologyDigest,
	}}
	if err := validateManifestTargetProfile(manifest, targetProfileRuntime); err != nil {
		t.Fatalf("authenticated runtime profile rejected: %v", err)
	}
}
