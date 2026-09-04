package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const catalogTopologyDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

func TestCatalogUsesServerOwnedTargetsWithoutDisclosingURLs(t *testing.T) {
	registry, err := NewRegistry(
		"https://router.example.internal/base",
		"https://envoy.example.internal",
		RegistryOptions{BackendTopologyDigest: catalogTopologyDigest},
	)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	catalog := registry.Catalog()
	encoded, err := json.Marshal(catalog)
	if err != nil {
		t.Fatalf("Marshal catalog: %v", err)
	}
	for _, secret := range []string{
		"router.example.internal", "envoy.example.internal", "router_api_url", "envoy_url",
		"backend_topology_digest", catalogTopologyDigest,
	} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("catalog disclosed server-owned target data %q: %s", secret, encoded)
		}
	}
	targets := make(map[string]CatalogTarget, len(catalog.Targets))
	for _, target := range catalog.Targets {
		targets[target.ID] = target
	}
	if _, ok := targets["fixture"]; !ok {
		t.Fatal("fixture target missing")
	}
	runtime, ok := targets["runtime"]
	if !ok {
		t.Fatal("runtime target missing")
	}
	if want := []TrackID{"routing", "capacity"}; !reflect.DeepEqual(runtime.TrackIDs, want) {
		t.Fatalf("runtime capabilities=%v, want %v", runtime.TrackIDs, want)
	}
	if catalog.GateContractVersion != GateContractVersion ||
		!reflect.DeepEqual(catalog.ChangeProfiles, builtinChangeProfiles()) {
		t.Fatalf("catalog gate contract drift: version=%q profiles=%+v", catalog.GateContractVersion, catalog.ChangeProfiles)
	}
	for _, test := range []struct {
		name string
		url  string
	}{
		{name: "credentials", url: "https://user:password@example.test"},
		{name: "query", url: "https://example.test?token=value"},
		{name: "non-http", url: "file:///tmp/socket"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewRegistry(test.url, ""); err == nil {
				t.Fatalf("unsafe server URL %q was accepted", test.url)
			}
		})
	}
}

func TestRuntimeCapabilitiesFollowConfiguredServices(t *testing.T) {
	textArms := []ModelArm{catalogTestArm("fast", []string{"text"}), catalogTestArm("deep", []string{"text"})}
	multimodalArms := append(copyModelArms(textArms), catalogTestArm("vision", []string{"text", "image"}))
	tests := []struct {
		name   string
		router string
		envoy  string
		arms   []ModelArm
		want   []TrackID
	}{
		{name: "router only", router: "http://router.test", want: []TrackID{"routing"}},
		{name: "envoy without explicit arms", envoy: "http://envoy.test", want: []TrackID{"capacity"}},
		{name: "both without explicit arms", router: "http://router.test", envoy: "http://envoy.test", want: []TrackID{"routing", "capacity"}},
		{name: "model arms do not imply a direct arm seam", envoy: "http://envoy.test", arms: textArms, want: []TrackID{"capacity"}},
		{name: "router envoy and arms still cannot attest pool or joint", router: "http://router.test", envoy: "http://envoy.test", arms: textArms, want: []TrackID{"routing", "capacity"}},
		{name: "multimodal requires a capable arm", envoy: "http://envoy.test", arms: multimodalArms, want: []TrackID{"multimodal", "capacity"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			registry, err := NewRegistry(test.router, test.envoy, RegistryOptions{
				ModelArms: test.arms, BackendTopologyDigest: catalogTopologyDigest,
			})
			if err != nil {
				t.Fatalf("NewRegistry: %v", err)
			}
			got := registry.targets["runtime"].Public.TrackIDs
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("runtime tracks=%v, want %v", got, test.want)
			}
		})
	}
	withoutTopology, err := NewRegistry(
		"http://router.test", "http://envoy.test", RegistryOptions{ModelArms: textArms},
	)
	if err != nil {
		t.Fatalf("NewRegistry without topology: %v", err)
	}
	if got := withoutTopology.targets["runtime"].Public.TrackIDs; len(got) != 0 {
		t.Fatalf("runtime advertised live capabilities without backend topology identity: %v", got)
	}
	authenticated, err := NewRegistry(
		"http://router.test", "http://envoy.test",
		RegistryOptions{ModelArms: multimodalArms, BackendTopologyDigest: catalogTopologyDigest, RouterAuthRequired: true},
	)
	if err != nil {
		t.Fatalf("NewRegistry authenticated: %v", err)
	}
	target := authenticated.targets["runtime"]
	if target.RouterAPIURL != "" || !reflect.DeepEqual(target.Public.TrackIDs, []TrackID{"multimodal", "capacity"}) ||
		target.Public.Labels["router_auth"] != "dedicated-evaluation-credential-unavailable" {
		t.Fatalf("authenticated runtime target was not fail-closed: %+v", target)
	}
}

func TestGenericLiveSuitesRemainDiagnosticE0(t *testing.T) {
	for _, suite := range builtinSuites() {
		if containsMode(suite.Modes, ModeLive) && suite.EvidenceLevel != "E0" {
			t.Fatalf("generic live suite %q evidence=%q, want diagnostic E0", suite.ID, suite.EvidenceLevel)
		}
	}
}

func catalogTestArm(id string, modalities []string) ModelArm {
	return ModelArm{
		ID: id, Model: "org/" + id,
		ProviderModelIDDigest: "sha256:" + strings.Repeat("a", 64),
		Modalities:            modalities,
	}
}

func TestCreateRunAllowlistAndCanonicalManifest(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	run, createErr := service.CreateRun(context.Background(), request)
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
	var manifest RunManifest
	if err := readJSON(manifestPath, &manifest); err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.SchemaVersion != SchemaVersion || manifest.Target.SchemaVersion != SchemaVersion {
		t.Fatalf("manifest versions are not strict: %+v", manifest)
	}
	if run.ChangeProfile != request.ChangeProfile || manifest.ChangeProfile != request.ChangeProfile {
		t.Fatalf("change profile was not frozen across run and manifest: run=%q manifest=%q", run.ChangeProfile, manifest.ChangeProfile)
	}
	if manifest.PolicySnapshotDigest != fixturePolicySnapshotDigest {
		t.Fatalf("fixture policy digest = %q, want builtin fixture identity", manifest.PolicySnapshotDigest)
	}
	if manifest.Target.ID != "fixture" || manifest.Target.RouterAPIURL != "" || manifest.Target.EnvoyURL != "" {
		t.Fatalf("fixture manifest contains unexpected target data: %+v", manifest.Target)
	}
	if manifest.ConfigDigest == "" || !digestPattern.MatchString(manifest.PolicySnapshotDigest) ||
		!digestPattern.MatchString(manifest.ManifestDigest) ||
		manifest.CodeRevision != testSourceRevision || manifest.GateContractVersion != GateContractVersion ||
		!reflect.DeepEqual(manifest.SuiteRevisions, map[string]string{"evaluation-smoke": "builtin-v1"}) {
		t.Fatalf("manifest provenance is incomplete: %+v", manifest)
	}
	if recomputed, err := manifestSemanticDigest(manifest); err != nil || recomputed != manifest.ManifestDigest {
		t.Fatalf("manifest server-owned digest=%q recomputed=%q err=%v", manifest.ManifestDigest, recomputed, err)
	}

	tests := []struct {
		name   string
		mutate func(*CreateRunRequest)
	}{
		{name: "unknown target", mutate: func(r *CreateRunRequest) { r.TargetID = "https://attacker.invalid" }},
		{name: "unknown suite", mutate: func(r *CreateRunRequest) { r.SuiteIDs = []string{"private-suite"} }},
		{name: "unknown track", mutate: func(r *CreateRunRequest) { r.TrackIDs = []TrackID{"private"} }},
		{name: "unknown change profile", mutate: func(r *CreateRunRequest) { r.ChangeProfile = "untrusted" }},
		{name: "target mode", mutate: func(r *CreateRunRequest) { r.Mode, r.TargetID = ModeLive, "fixture" }},
		{name: "suite requires complete target capabilities", mutate: func(r *CreateRunRequest) {
			r.Mode, r.TargetID = ModeLive, "runtime"
			r.SuiteIDs, r.TrackIDs = []string{"live-joint"}, []TrackID{"routing"}
		}},
		{name: "negative seed", mutate: func(r *CreateRunRequest) { r.Seed = -1 }},
		{name: "oversized seed", mutate: func(r *CreateRunRequest) { r.Seed = 1 << 32 }},
		{name: "track outside suite", mutate: func(r *CreateRunRequest) {
			r.Mode, r.TargetID = ModeLive, "runtime"
			r.SuiteIDs, r.TrackIDs = []string{"live-routing-core"}, []TrackID{"capacity"}
		}},
		{name: "auto start", mutate: func(r *CreateRunRequest) { r.AutoStart = true }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			invalid := validCreateRequest()
			test.mutate(&invalid)
			if _, invalidErr := service.CreateRun(context.Background(), invalid); !errors.Is(invalidErr, ErrInvalid) {
				t.Fatalf("CreateRun error=%v, want ErrInvalid", invalidErr)
			}
		})
	}
	runs, listErr := service.ListRuns()
	if listErr != nil || len(runs) != 1 {
		t.Fatalf("invalid requests changed store: runs=%d err=%v", len(runs), listErr)
	}
	if _, statErr := os.Stat(filepath.Join(root, "runs", run.ID, "events.jsonl")); !os.IsNotExist(statErr) {
		t.Fatalf("Go control plane must not create Python evidence events.jsonl; err=%v", statErr)
	}
}

func TestCatalogAndManifestRefreshFromTheSameCurrentConfigBytes(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	findRuntime := func(catalog Catalog) CatalogTarget {
		for _, target := range catalog.Targets {
			if target.ID == "runtime" {
				return target
			}
		}
		t.Fatal("runtime target missing")
		return CatalogTarget{}
	}
	if got := findRuntime(service.Catalog()).TrackIDs; !reflect.DeepEqual(got, []TrackID{"routing", "capacity"}) {
		t.Fatalf("initial runtime tracks=%v", got)
	}
	configBytes := []byte(modelArmTestYAML)
	if err := os.WriteFile(filepath.Join(root, "config.yaml"), configBytes, 0o600); err != nil {
		t.Fatalf("deploy updated config: %v", err)
	}
	if got := findRuntime(service.Catalog()).TrackIDs; !reflect.DeepEqual(got, []TrackID{"routing", "multimodal", "capacity"}) {
		t.Fatalf("refreshed runtime tracks=%v", got)
	}
	run, err := service.CreateRun(context.Background(), CreateRunRequest{
		Name: "multimodal snapshot", SuiteIDs: []string{"live-multimodal"}, TrackIDs: []TrackID{"multimodal"},
		Mode: ModeLive, TargetID: "runtime", ChangeProfile: "agent_multimodal",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	var manifest RunManifest
	if err := readJSON(filepath.Join(root, "runs", run.ID, manifestFileName), &manifest); err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if len(manifest.Target.ModelArms) != 2 || manifest.ConfigDigest != digestBytes(configBytes) ||
		!digestPattern.MatchString(manifest.PolicySnapshotDigest) ||
		!reflect.DeepEqual(manifest.SuiteRevisions, map[string]string{"live-multimodal": "executor-v1"}) {
		t.Fatalf("manifest was not frozen from deployed bytes: %+v", manifest)
	}
	poolRequest := CreateRunRequest{
		Name: "unattested pool", SuiteIDs: []string{"live-model-pool"}, TrackIDs: []TrackID{"model_pool"},
		Mode: ModeLive, TargetID: "runtime", ChangeProfile: "model_pool",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	if _, err := service.CreateRun(context.Background(), poolRequest); !errors.Is(err, ErrInvalid) {
		t.Fatalf("generic runtime accepted unattested live pool execution: %v", err)
	}
}

func TestBaselineCandidateCreationRequiresAComparableCohort(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	baseline, err := service.CreateRun(context.Background(), validCreateRequest())
	if err != nil {
		t.Fatalf("create baseline: %v", err)
	}
	baseline.Status = StatusCompleted
	if err := service.store.UpdateRun(baseline); err != nil {
		t.Fatalf("complete baseline: %v", err)
	}
	valid := validCreateRequest()
	valid.Name = "candidate"
	valid.BaselineRunID = baseline.ID
	if _, err := service.CreateRun(context.Background(), valid); err != nil {
		t.Fatalf("comparable candidate rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*CreateRunRequest)
	}{
		{name: "mode and target", mutate: func(r *CreateRunRequest) { r.Mode, r.TargetID = ModeLive, "runtime" }},
		{name: "suite", mutate: func(r *CreateRunRequest) { r.SuiteIDs = []string{"live-routing-core"} }},
		{name: "track", mutate: func(r *CreateRunRequest) { r.TrackIDs = []TrackID{"joint"} }},
		{name: "sample limit", mutate: func(r *CreateRunRequest) { r.SampleLimit++ }},
		{name: "concurrency", mutate: func(r *CreateRunRequest) { r.Concurrency++ }},
		{name: "seed", mutate: func(r *CreateRunRequest) { r.Seed++ }},
		{name: "change profile", mutate: func(r *CreateRunRequest) { r.ChangeProfile = "recipe" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := valid
			candidate.SuiteIDs = append([]string(nil), valid.SuiteIDs...)
			candidate.TrackIDs = append([]TrackID(nil), valid.TrackIDs...)
			test.mutate(&candidate)
			if _, err := service.CreateRun(context.Background(), candidate); !errors.Is(err, ErrInvalid) {
				t.Fatalf("candidate error=%v, want ErrInvalid", err)
			}
		})
	}
	if err := os.WriteFile(
		filepath.Join(root, "config.yaml"),
		[]byte("version: v0.3\nrouting:\n  strategy: confidence\n  modelCards: []\n"),
		0o600,
	); err != nil {
		t.Fatalf("change config snapshot: %v", err)
	}
	if _, err := service.CreateRun(context.Background(), valid); err != nil {
		t.Fatalf("fixture candidate rejected an unrelated Router config change: %v", err)
	}
}

func TestLiveCandidateCreationFreezesOnlyTheProfileOwnedPolicyFactor(t *testing.T) {
	for _, test := range []struct {
		profile ChangeProfile
		wantErr bool
	}{
		{profile: "schema_adapter", wantErr: true},
		{profile: "recipe", wantErr: false},
	} {
		t.Run(string(test.profile), func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			configPath := filepath.Join(root, "config.yaml")
			if err := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); err != nil {
				t.Fatalf("write baseline config: %v", err)
			}
			request := CreateRunRequest{
				Name: "live policy baseline", SuiteIDs: []string{"live-routing-core"}, TrackIDs: []TrackID{"routing"},
				Mode: ModeLive, TargetID: "runtime", ChangeProfile: test.profile,
				SampleLimit: 4, Concurrency: 1, Seed: 17,
			}
			baseline, err := service.CreateRun(context.Background(), request)
			if err != nil {
				t.Fatalf("create baseline: %v", err)
			}
			baseline.Status = StatusCompleted
			if updateErr := service.store.UpdateRun(baseline); updateErr != nil {
				t.Fatalf("complete baseline: %v", updateErr)
			}
			changed := strings.Replace(modelArmTestYAML, "routing:\n  modelCards:", "routing:\n  strategy: confidence\n  modelCards:", 1)
			if writeErr := os.WriteFile(configPath, []byte(changed), 0o600); writeErr != nil {
				t.Fatalf("write candidate config: %v", writeErr)
			}
			request.Name = "live policy candidate"
			request.BaselineRunID = baseline.ID
			_, err = service.CreateRun(context.Background(), request)
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("frozen policy change error=%v, want ErrInvalid", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("allowed policy treatment rejected: %v", err)
			}
		})
	}
}
