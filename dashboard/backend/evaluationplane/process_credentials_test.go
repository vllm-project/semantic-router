package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type staticCredentialProvider struct {
	token string
	err   error
}

func (provider staticCredentialProvider) ManagementCredential() (string, error) {
	return provider.token, provider.err
}

type authenticatedRouterFixture struct {
	service      *Service
	manifestPath string
	manifest     RunManifest
	provider     staticCredentialProvider
}

func newAuthenticatedRouterFixture(t *testing.T) authenticatedRouterFixture {
	t.Helper()
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create private store: %v", err)
	}
	configPath := filepath.Join(t.TempDir(), "router.yaml")
	if err := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write Router config: %v", err)
	}
	provider := staticCredentialProvider{token: "server-owned-router-token"}
	service, serviceErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		RouterAPIURL: "http://router.internal", EnvoyURL: "http://envoy.internal",
		EnvoyAPIKeyEnv: "VLLM_SR_EVALUATION_ENVOY_API_KEY", CredentialProvider: provider,
		CodeRevision: testSourceRevision, Process: &controlledProcess{},
	})
	if serviceErr != nil {
		t.Fatalf("NewService: %v", serviceErr)
	}
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "capacity live", SuiteIDs: []string{"live-capacity"}, TrackIDs: []TrackID{"capacity"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "runtime_capacity",
		SampleLimit: 4, Concurrency: 2, Seed: 17,
		CapacitySLO:          testCapacitySLO(1),
		CapacityLoadProtocol: defaultCapacityLoadProtocol(2),
	})
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
	var manifest RunManifest
	if err := readJSON(manifestPath, &manifest); err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	return authenticatedRouterFixture{service: service, manifestPath: manifestPath, manifest: manifest, provider: provider}
}

func assertAuthenticatedRouterManifestIsRedacted(t *testing.T, fixture authenticatedRouterFixture) {
	t.Helper()
	manifest := fixture.manifest
	if manifest.Target.RouterAPIURL != "" || manifest.Target.RouterAPIKey != nil {
		t.Fatalf("authenticated Router leaked into a generic worker target: %#v", manifest.Target)
	}
	if manifest.Target.EnvoyAPIKey == nil || manifest.Target.EnvoyAPIKey.Env != "VLLM_SR_EVALUATION_ENVOY_API_KEY" {
		t.Fatalf("Envoy SecretRef = %#v", manifest.Target.EnvoyAPIKey)
	}
	if manifest.Target.Mixture == nil || len(manifest.Target.Mixture.ModelArms) != 2 {
		t.Fatalf("mixture = %#v, want two server-owned arms", manifest.Target.Mixture)
	}
	if !digestPattern.MatchString(manifest.Target.BackendTopologyDigest) {
		t.Fatalf("backend topology digest = %q", manifest.Target.BackendTopologyDigest)
	}
	encoded, marshalErr := json.Marshal(manifest)
	if marshalErr != nil {
		t.Fatalf("marshal manifest: %v", marshalErr)
	}
	for _, forbidden := range []string{
		fixture.provider.token, "literal-test-secret", "PrivateOrg/Secret-Upstream-ID",
		"private.models.example.test", "local-private.example.test",
	} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("manifest leaked private model or credential value %q", forbidden)
		}
	}
}

func assertGenericRoutingFailsClosed(t *testing.T, service *Service) {
	t.Helper()
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "blocked routing", SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "recipe",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("generic runtime accepted routing with broad Router auth: %v", err)
	}
}

func assertWorkerCredentialScope(t *testing.T, manifestPath string) *CommandProcess {
	t.Helper()
	t.Setenv(routerManagementCredentialEnv, "source-management-token")
	t.Setenv(routerEvaluationCredentialEnv, "stale-evaluation-token")
	t.Setenv("DASHBOARD_JWT_SECRET", "dashboard-secret")
	t.Setenv("ADMIN_PASSWORD", "admin-secret")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
	t.Setenv("GITHUB_TOKEN", "github-secret")
	t.Setenv("DATABASE_URL", "database-secret")
	t.Setenv("VLLM_SR_EVALUATION_ENVOY_API_KEY", "server-owned-envoy-token")
	worker := NewCommandProcess("python3")
	worker.envoyAPIKeyEnv = "VLLM_SR_EVALUATION_ENVOY_API_KEY"
	sandboxRoot := t.TempDir()
	manifest, _, manifestErr := readRunManifestStrict(manifestPath)
	if manifestErr != nil {
		t.Fatalf("read worker manifest: %v", manifestErr)
	}
	credentials, credentialErr := worker.brokerCredentials(manifest)
	if credentialErr != nil {
		t.Fatalf("resolve worker broker credentials: %v", credentialErr)
	}
	environment := isolatedWorkerEnvironment(sandboxRoot)
	if value, ok := environmentValue(environment, routerManagementCredentialEnv); ok || value != "" {
		t.Fatal("source Router management credential leaked to the worker")
	}
	if value, ok := environmentValue(environment, routerEvaluationCredentialEnv); ok || value != "" {
		t.Fatal("broad Router credential leaked to the worker")
	}
	if value, ok := environmentValue(environment, "VLLM_SR_EVALUATION_ENVOY_API_KEY"); ok || value != "" {
		t.Fatalf("Envoy credential leaked to worker environment: value=%q present=%v", value, ok)
	}
	if credentials.envoy != "server-owned-envoy-token" {
		t.Fatalf("server-owned broker credential = %q", credentials.envoy)
	}
	if got := os.Getenv(routerEvaluationCredentialEnv); got != "stale-evaluation-token" {
		t.Fatalf("worker environment mutated the server process: %q", got)
	}
	for _, key := range []string{"DASHBOARD_JWT_SECRET", "ADMIN_PASSWORD", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "DATABASE_URL"} {
		if value, present := environmentValue(environment, key); present || value != "" {
			t.Fatalf("non-worker credential %s leaked: value=%q present=%v", key, value, present)
		}
	}
	for _, key := range []string{"PATH", "PYTHONPATH"} {
		if value, present := environmentValue(environment, key); present || value != "" {
			t.Fatalf("ambient process environment %s leaked: value=%q present=%v", key, value, present)
		}
	}
	if value, present := environmentValue(environment, "HOME"); !present || value != filepath.Join(sandboxRoot, "home") {
		t.Fatalf("worker HOME=%q present=%v, want private sandbox home", value, present)
	}
	return worker
}

func assertTamperedEnvoyCredentialRefIsRejected(t *testing.T, worker *CommandProcess, manifestPath string, manifest RunManifest) {
	t.Helper()
	manifest.Target.EnvoyAPIKey.Env = "OTHER_ENVOY_CREDENTIAL"
	refreshedDigest, digestErr := manifestSemanticDigest(manifest)
	if digestErr != nil {
		t.Fatalf("refresh manifest digest: %v", digestErr)
	}
	manifest.ManifestDigest = refreshedDigest
	if err := writeJSONAtomic(manifestPath, manifest); err != nil {
		t.Fatalf("stage tampered Envoy ref: %v", err)
	}
	t.Setenv("OTHER_ENVOY_CREDENTIAL", "other-token")
	tampered, _, err := readRunManifestStrict(manifestPath)
	if err != nil {
		t.Fatalf("read tampered worker manifest: %v", err)
	}
	if _, err := worker.brokerCredentials(tampered); err == nil || !strings.Contains(err.Error(), "unsupported Envoy credential") {
		t.Fatalf("tampered Envoy credential ref error=%v", err)
	}
}

func TestAuthenticatedRouterIsFailClosedAndEnvoyCredentialRemainsScoped(t *testing.T) {
	fixture := newAuthenticatedRouterFixture(t)
	assertAuthenticatedRouterManifestIsRedacted(t, fixture)
	assertGenericRoutingFailsClosed(t, fixture.service)
	worker := assertWorkerCredentialScope(t, fixture.manifestPath)
	assertTamperedEnvoyCredentialRefIsRejected(t, worker, fixture.manifestPath, fixture.manifest)
}

func TestReplayAndAuthDisabledRunsDoNotRequireRouterCredentials(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("replay create unexpectedly consulted Router credentials: %v", createErr)
	}
	var manifest RunManifest
	if err := readJSON(filepath.Join(root, "runs", run.ID, manifestFileName), &manifest); err != nil {
		t.Fatalf("read replay manifest: %v", err)
	}
	if manifest.Target.RouterAPIKey != nil {
		t.Fatalf("replay manifest contains Router SecretRef: %#v", manifest.Target.RouterAPIKey)
	}
	t.Setenv("VLLM_SR_EVALUATION_ENVOY_API_KEY", "must-not-reach-replay")
	worker := NewCommandProcess("python3")
	worker.envoyAPIKeyEnv = "VLLM_SR_EVALUATION_ENVOY_API_KEY"
	credentials, credentialErr := worker.brokerCredentials(manifest)
	if credentialErr != nil {
		t.Fatalf("replay worker credentials: %v", credentialErr)
	}
	if credentials != (workerBrokerCredentials{}) {
		t.Fatalf("replay worker resolved unused credentials: %#v", credentials)
	}
	environment := isolatedWorkerEnvironment(t.TempDir())
	if value, present := environmentValue(environment, "VLLM_SR_EVALUATION_ENVOY_API_KEY"); present || value != "" {
		t.Fatalf("unused Envoy credential leaked to replay worker: value=%q present=%v", value, present)
	}
}

func TestPendingRoutingRunCannotStartAfterRouterAuthBecomesRequired(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	if err := os.WriteFile(filepath.Join(root, "config.yaml"), []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write live Router config: %v", err)
	}
	run, err := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "routing before auth", SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "recipe",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("create unauthenticated routing run: %v", err)
	}
	service.registrySource.routerAuthRequired = true
	if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); !errors.Is(startErr, ErrConflict) {
		t.Fatalf("StartRun after Router auth error=%v, want ErrConflict", startErr)
	}
	stored, err := service.GetRunAs(SystemActor(), run.ID)
	if err != nil || stored.Status != StatusPending {
		t.Fatalf("rejected run status=%s err=%v, want pending", stored.Status, err)
	}
}

func TestTargetContractRejectsLiteralLikeAndInvalidEnvironmentReferences(t *testing.T) {
	validArm := catalogTestArm("arm", []string{"text"})
	crossCollisionA := catalogTestArm("first", []string{"text"})
	crossCollisionA.Model = "second"
	crossCollisionB := catalogTestArm("second", []string{"text"})
	crossCollisionB.Model = "third"
	for _, options := range []RegistryOptions{
		{RouterAPIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "literal-secret"}},
		{EnvoyAPIKey: &SecretRef{SchemaVersion: "evaluation.v2", Env: "VALID_ENV"}},
		{EnvoyAPIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: routerManagementCredentialEnv}},
		{
			RouterAPIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "SHARED_ENV"},
			EnvoyAPIKey:  &SecretRef{SchemaVersion: SchemaVersion, Env: "SHARED_ENV"},
		},
		{Mixtures: []MixtureTargetSnapshot{catalogTestMixtureSnapshot([]ModelArm{{ID: "arm", Model: "model", ProviderModelIDDigest: "raw-provider-id"}}, catalogTopologyDigest)}},
		{Mixtures: []MixtureTargetSnapshot{catalogTestMixtureSnapshot([]ModelArm{validArm, validArm}, catalogTopologyDigest)}},
		{Mixtures: []MixtureTargetSnapshot{catalogTestMixtureSnapshot([]ModelArm{crossCollisionA, crossCollisionB}, catalogTopologyDigest)}},
	} {
		if _, err := NewRegistry("http://router.test", "http://envoy.test", options); err == nil {
			t.Fatalf("unsafe target options accepted: %#v", options)
		}
	}
	if _, err := NewRegistry("", "http://envoy.test", RegistryOptions{
		RouterAPIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "ROUTER_EVAL_KEY"},
	}); err == nil {
		t.Fatal("Router credential reference without Router URL was accepted")
	}
	if _, err := NewRegistry("http://router.test", "", RegistryOptions{
		EnvoyAPIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "ENVOY_EVAL_KEY"},
	}); err == nil {
		t.Fatal("Envoy credential reference without Envoy URL was accepted")
	}
}

func environmentValue(environment []string, key string) (string, bool) {
	for _, entry := range environment {
		name, value, found := strings.Cut(entry, "=")
		if found && name == key {
			return value, true
		}
	}
	return "", false
}
