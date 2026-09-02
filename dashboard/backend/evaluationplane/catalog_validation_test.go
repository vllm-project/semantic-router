package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
)

const catalogTopologyDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

func TestCatalogMethodEvidenceSourceInventoryDrivesValidation(t *testing.T) {
	if len(catalogMethodEvidenceSourceInventory) == 0 {
		t.Fatal("catalog method evidence source inventory is empty")
	}
	seen := make(map[CatalogMethodEvidenceSource]struct{}, len(catalogMethodEvidenceSourceInventory))
	for _, source := range catalogMethodEvidenceSourceInventory {
		if !validCatalogMethodEvidenceSource(source) {
			t.Fatalf("canonical catalog method evidence source %q was rejected", source)
		}
		if _, duplicate := seen[source]; duplicate {
			t.Fatalf("duplicate catalog method evidence source %q", source)
		}
		seen[source] = struct{}{}
		gateIDs := []string{}
		if source == CatalogMethodEvidenceSourceServerBrokeredLive {
			gateIDs = []string{"G4"}
		}
		suite := CatalogSuite{
			ID: "inventory-source", TrackIDs: []TrackID{"routing"},
			Methods: []CatalogMethod{{
				ID: "inventory.source.v1", TrackID: "routing", QualifiedGateIDs: gateIDs,
				EvidenceSource: source, Status: "configured",
			}},
		}
		if err := validateCatalogMethods(suite); err != nil {
			t.Fatalf("canonical catalog method evidence source %q failed validation: %v", source, err)
		}
	}

	unknown := CatalogMethodEvidenceSource("unknown_source")
	if validCatalogMethodEvidenceSource(unknown) {
		t.Fatal("unknown catalog method evidence source was admitted")
	}
	if err := validateCatalogMethods(CatalogSuite{
		ID: "unknown-source", TrackIDs: []TrackID{"routing"},
		Methods: []CatalogMethod{{
			ID: "unknown.source.v1", TrackID: "routing", QualifiedGateIDs: []string{},
			EvidenceSource: unknown, Status: "configured",
		}},
	}); err == nil {
		t.Fatal("catalog method validation accepted an unknown evidence source")
	}
}

func TestCatalogUsesServerOwnedTargetsWithoutDisclosingURLs(t *testing.T) {
	arms := []ModelArm{catalogTestArm("deep", []string{"text"}), catalogTestArm("fast", []string{"text"})}
	mixture := catalogTestMixtureSnapshot(arms, catalogTopologyDigest)
	registry, err := NewRegistry(
		"https://router.example.internal",
		"https://envoy.example.internal",
		RegistryOptions{Mixtures: []MixtureTargetSnapshot{mixture}},
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
	runtime, ok := targets[mixture.Mixture.ID]
	if !ok {
		t.Fatal("Mixture-of-Models target missing")
	}
	if runtime.Kind != "mixture-of-models" || runtime.Mixture == nil {
		t.Fatalf("catalog target is not mixture-bound: %+v", runtime)
	}
	if want := []TrackID{"routing", "model_pool", "joint", "capacity"}; !reflect.DeepEqual(runtime.TrackIDs, want) {
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
		{name: "empty query marker", url: "https://example.test?"},
		{name: "API path", url: "https://example.test/v1"},
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
	textArms := []ModelArm{catalogTestArm("deep", []string{"text"}), catalogTestArm("fast", []string{"text"})}
	multimodalArms := append(copyModelArms(textArms), catalogTestArm("vision", []string{"text", "image"}))
	tests := []struct {
		name   string
		router string
		envoy  string
		arms   []ModelArm
		want   []TrackID
	}{
		{name: "router only cannot reach a live chat endpoint", router: "http://router.test", arms: textArms, want: []TrackID{}},
		{name: "envoy without explicit arms", envoy: "http://envoy.test", want: []TrackID{"capacity"}},
		{name: "both without explicit arms", router: "http://router.test", envoy: "http://envoy.test", want: []TrackID{"routing", "capacity"}},
		{name: "direct and routed pool probes use Envoy without Router diagnostics", envoy: "http://envoy.test", arms: textArms, want: []TrackID{"model_pool", "joint", "capacity"}},
		{name: "router envoy and two frozen arms enable core MoM evaluation", router: "http://router.test", envoy: "http://envoy.test", arms: textArms, want: []TrackID{"routing", "model_pool", "joint", "capacity"}},
		{name: "multimodal requires a capable arm", envoy: "http://envoy.test", arms: multimodalArms, want: []TrackID{"model_pool", "joint", "multimodal", "capacity"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mixture := catalogTestMixtureSnapshot(test.arms, catalogTopologyDigest)
			registry, err := NewRegistry(test.router, test.envoy, RegistryOptions{
				Mixtures: []MixtureTargetSnapshot{mixture},
			})
			if err != nil {
				t.Fatalf("NewRegistry: %v", err)
			}
			got := registry.targets[mixture.Mixture.ID].Public.TrackIDs
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("runtime tracks=%v, want %v", got, test.want)
			}
		})
	}
	withoutTopology, err := NewRegistry(
		"http://router.test", "http://envoy.test", RegistryOptions{Mixtures: []MixtureTargetSnapshot{catalogTestMixtureSnapshot(textArms, "")}},
	)
	if err != nil {
		t.Fatalf("NewRegistry without topology: %v", err)
	}
	mixtureID := mixtureTargetID("default")
	if got := withoutTopology.targets[mixtureID].Public.TrackIDs; len(got) != 0 {
		t.Fatalf("runtime advertised live capabilities without backend topology identity: %v", got)
	}
	withLedgers, err := NewRegistry(
		"http://router.test", "http://envoy.test", RegistryOptions{
			Mixtures: []MixtureTargetSnapshot{catalogTestMixtureSnapshot(textArms, catalogTopologyDigest)},
			FaultRecoveryLedger: &ServiceEndpoint{
				SchemaVersion: SchemaVersion, URL: "http://recovery-ledger.test", TimeoutSeconds: 30,
			},
			HardPolicyLedger: &ServiceEndpoint{
				SchemaVersion: SchemaVersion, URL: "http://policy-ledger.test", TimeoutSeconds: 30,
			},
			ProductionExperimentLedger: &ServiceEndpoint{
				SchemaVersion: SchemaVersion, URL: "http://experiment-ledger.test", TimeoutSeconds: 30,
			},
		},
	)
	if err != nil {
		t.Fatalf("NewRegistry with ledgers: %v", err)
	}
	if got, want := withLedgers.targets[mixtureID].Public.TrackIDs,
		[]TrackID{"routing", "model_pool", "joint", "agentic", "preference", "safety", "capacity"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("runtime ledger capabilities=%v, want %v", got, want)
	}
	authenticated, err := NewRegistry(
		"http://router.test", "http://envoy.test",
		RegistryOptions{Mixtures: []MixtureTargetSnapshot{catalogTestMixtureSnapshot(multimodalArms, catalogTopologyDigest)}, RouterAuthRequired: true},
	)
	if err != nil {
		t.Fatalf("NewRegistry authenticated: %v", err)
	}
	target := authenticated.targets[mixtureID]
	if target.RouterAPIURL != "" || !reflect.DeepEqual(target.Public.TrackIDs, []TrackID{"model_pool", "joint", "multimodal", "capacity"}) ||
		target.Public.Labels["router_auth"] != "dedicated-evaluation-credential-unavailable" {
		t.Fatalf("authenticated runtime target was not fail-closed: %+v", target)
	}
}

func TestSingleArmMixtureDoesNotAdvertisePoolOrJointTracks(t *testing.T) {
	snapshot := catalogTestMixtureSnapshot([]ModelArm{catalogTestArm("only", []string{"text"})}, catalogTopologyDigest)
	registry, err := NewRegistry("http://router.test", "http://envoy.test", RegistryOptions{
		Mixtures: []MixtureTargetSnapshot{snapshot},
	})
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	got := registry.targets[snapshot.Mixture.ID].Public.TrackIDs
	if want := []TrackID{"routing", "capacity"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("single-arm mixture tracks=%v, want %v", got, want)
	}
}

func TestRuntimeConnectivityWithoutMixtureAdvertisesNoTracks(t *testing.T) {
	target := targetDefinition{
		Contract: targetContract{
			ExecutionProfile:  targetProfileRuntime,
			PolicySnapshot:    policySnapshotRuntime,
			TrackRequirements: runtimeTrackRequirements(),
		},
		RouterAPIURL:          "http://router.test",
		EnvoyURL:              "http://envoy.test",
		BackendTopologyDigest: catalogTopologyDigest,
	}
	if got := availableTargetTracks(target); len(got) != 0 {
		t.Fatalf("unbound runtime connectivity advertised tracks: %v", got)
	}
}

func TestMixtureContractRejectsEmptyDecisionArmBoundary(t *testing.T) {
	mixture := catalogTestMixtureSnapshot(
		[]ModelArm{catalogTestArm("fast", []string{"text"}), catalogTestArm("strong", []string{"text"})},
		catalogTopologyDigest,
	).Mixture
	mixture.Decisions[0].ArmIDs = []string{}
	if err := validateMixtureContract(&mixture); err == nil {
		t.Fatal("mixture contract accepted a decision with no eligible arm")
	}
}

func TestTargetExecutorCompatibilityUsesThePublishedContract(t *testing.T) {
	target := CatalogTarget{
		ID: "provider-owned-target", Kind: "provider-runtime", Modes: []Mode{ModeReplay, ModeLive},
		AcceptedExecutors: map[Mode][]string{
			ModeReplay: {"provider-replay.v1"},
			ModeLive:   {"provider-live.v2", "provider-live.v1"},
		},
	}
	if !executorTargetMatches("provider-replay.v1", ModeReplay, target) ||
		!executorTargetMatches("provider-live.v1", ModeLive, target) {
		t.Fatal("target rejected an executor declared by its public contract")
	}
	if executorTargetMatches("provider-live.v1", ModeReplay, target) ||
		executorTargetMatches("undeclared.v1", ModeLive, target) {
		t.Fatal("target accepted an executor outside the requested mode contract")
	}
	target.AcceptedExecutors[ModeLive] = []string{"provider-live.v1", "provider-live.v1"}
	if executorTargetMatches("provider-replay.v1", ModeReplay, target) {
		t.Fatal("target with a malformed executor contract remained executable")
	}
}

func TestCreateCollectionsRejectDuplicatesAndFreezeCatalogOrder(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	for _, mutate := range []func(*CreateRunRequest){
		func(request *CreateRunRequest) { request.SuiteIDs = []string{"evaluation-smoke", "evaluation-smoke"} },
		func(request *CreateRunRequest) { request.TrackIDs = []TrackID{"routing", "routing"} },
	} {
		request := validCreateRequest()
		request.ClientRequestID = newTestClientRequestID()
		mutate(&request)
		if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); !errors.Is(err, ErrInvalid) {
			t.Fatalf("duplicate create collection error=%v, want ErrInvalid", err)
		}
	}
	invalidBaseline := validCreateRequest()
	invalidBaseline.ClientRequestID = newTestClientRequestID()
	invalidBaseline.BaselineRunID = "portable-but-not-a-uuid"
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), invalidBaseline); !errors.Is(err, ErrInvalid) {
		t.Fatalf("non-UUID baseline error=%v, want ErrInvalid", err)
	}

	registry, err := NewRegistry(
		"http://router.test", "http://envoy.test",
		RegistryOptions{Mixtures: []MixtureTargetSnapshot{catalogTestMixtureSnapshot(
			[]ModelArm{catalogTestArm("deep", []string{"text"}), catalogTestArm("fast", []string{"text"})},
			catalogTopologyDigest,
		)}},
	)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	request := CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "canonical collections",
		SuiteIDs: []string{"live-capacity", "live-mom-core"},
		TrackIDs: []TrackID{"capacity", "routing"}, Mode: ModeLive, TargetID: mixtureTargetID("default"),
		ChangeProfile: "recipe", SampleLimit: 4, Concurrency: 2, Seed: 17,
		CapacitySLO:          testCapacitySLO(1),
		CapacityLoadProtocol: defaultCapacityLoadProtocol(2),
	}
	canonical, _, err := service.validateCreateRequest(registry, request)
	if err != nil {
		t.Fatalf("validateCreateRequest: %v", err)
	}
	if !reflect.DeepEqual(canonical.SuiteIDs, []string{"live-mom-core", "live-capacity"}) ||
		!reflect.DeepEqual(canonical.TrackIDs, []TrackID{"routing", "capacity"}) {
		t.Fatalf("canonical collections suites=%v tracks=%v", canonical.SuiteIDs, canonical.TrackIDs)
	}
}

func TestBuiltinLiveSuiteEvidenceCeilingsMatchTheirServerMethods(t *testing.T) {
	want := map[string]EvidenceLevel{
		"live-mom-core": "E0", "live-agent-tasks": "E5", "live-fault-recovery": "E5", "live-multimodal": "E0",
		"live-hard-policy": "E4", "live-production-experiment": "E5", "live-capacity": "E0",
	}
	for _, suite := range builtinSuites() {
		if expected, live := want[suite.ID]; live && suite.EvidenceLevel != expected {
			t.Fatalf("live suite %q evidence=%q, want %q", suite.ID, suite.EvidenceLevel, expected)
		}
	}
}

func catalogTestMixtureSnapshot(arms []ModelArm, topologyDigest string) MixtureTargetSnapshot {
	return catalogTestNamedMixtureSnapshot("default", []string{"MoM"}, arms, topologyDigest)
}

func catalogTestNamedMixtureSnapshot(recipeName string, aliases []string, arms []ModelArm, topologyDigest string) MixtureTargetSnapshot {
	arms = copyModelArms(arms)
	sort.Slice(arms, func(i, j int) bool { return arms[i].Model < arms[j].Model })
	armIDs := make([]string, len(arms))
	for index, arm := range arms {
		armIDs[index] = arm.ID
	}
	sort.Strings(armIDs)
	aliases = append([]string(nil), aliases...)
	recipeDigest := digestString("catalog-recipe:" + recipeName)
	poolDigest := modelPoolSnapshotDigest(arms)
	selectorPolicyDigest := digestString("catalog-selector-policy:" + recipeName)
	selectorDigest := selectorSnapshotDigest(selectorPolicyDigest, []SupportModel{})
	adaptationDigest := digestString("catalog-adaptation:" + recipeName)
	id := "mom-" + strings.TrimPrefix(digestString(recipeName), "sha256:")
	decisions := []MixtureDecisionBinding{}
	if len(armIDs) > 0 {
		decisions = append(decisions, MixtureDecisionBinding{Name: "route", Algorithm: "static", ArmIDs: armIDs})
	}
	snapshot := MixtureTargetSnapshot{
		Mixture: ManifestMixture{
			SchemaVersion: SchemaVersion, ID: id, EntrypointModel: aliases[0], Aliases: aliases,
			RecipeName: recipeName, RecipeDigest: recipeDigest, PoolDigest: poolDigest,
			SelectorPolicyDigest: selectorPolicyDigest, SelectorDigest: selectorDigest, AdaptationDigest: adaptationDigest,
			BindingDigest: digestString("catalog-binding:" + recipeName),
			ModelArms:     arms, SupportModels: []SupportModel{},
			Decisions: decisions,
		},
		BackendTopologyDigest: topologyDigest,
		Ready:                 digestPattern.MatchString(topologyDigest),
	}
	mustFreezeTestRoutingRecipePlan(&snapshot.Mixture)
	return snapshot
}

func TestCreateRunAllowlistAndCanonicalManifest(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), request)
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
	if !reflect.DeepEqual(manifest.SuiteExecutors, map[string]string{"evaluation-smoke": "fixture-replay.v1"}) {
		t.Fatalf("manifest suite executor identities=%v", manifest.SuiteExecutors)
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
		{name: "name whitespace", mutate: func(r *CreateRunRequest) { r.Name = " " + r.Name }},
		{name: "client request whitespace", mutate: func(r *CreateRunRequest) { r.ClientRequestID += " " }},
		{name: "description whitespace", mutate: func(r *CreateRunRequest) { r.Description += " " }},
		{name: "target whitespace", mutate: func(r *CreateRunRequest) { r.TargetID += " " }},
		{name: "change profile whitespace", mutate: func(r *CreateRunRequest) { r.ChangeProfile += " " }},
		{name: "baseline whitespace", mutate: func(r *CreateRunRequest) { r.BaselineRunID = " " }},
		{name: "suite whitespace", mutate: func(r *CreateRunRequest) { r.SuiteIDs[0] += " " }},
		{name: "track whitespace", mutate: func(r *CreateRunRequest) { r.TrackIDs[0] += " " }},
		{name: "target mode", mutate: func(r *CreateRunRequest) { r.Mode, r.TargetID = ModeLive, "fixture" }},
		{name: "suite requires target capabilities", mutate: func(r *CreateRunRequest) {
			r.Mode, r.TargetID = ModeLive, mixtureTargetID("default")
			r.SuiteIDs, r.TrackIDs = []string{"live-multimodal"}, []TrackID{"multimodal"}
		}},
		{name: "negative seed", mutate: func(r *CreateRunRequest) { r.Seed = -1 }},
		{name: "oversized seed", mutate: func(r *CreateRunRequest) { r.Seed = 1 << 32 }},
		{name: "track outside suite", mutate: func(r *CreateRunRequest) {
			r.Mode, r.TargetID = ModeLive, mixtureTargetID("default")
			r.SuiteIDs, r.TrackIDs = []string{"live-mom-core"}, []TrackID{"capacity"}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			invalid := validCreateRequest()
			test.mutate(&invalid)
			if _, invalidErr := service.CreateRunAs(context.Background(), SystemActor(), invalid); !errors.Is(invalidErr, ErrInvalid) {
				t.Fatalf("CreateRun error=%v, want ErrInvalid", invalidErr)
			}
		})
	}
	runs, listErr := service.store.ListRuns()
	if listErr != nil || len(runs) != 1 {
		t.Fatalf("invalid requests changed store: runs=%d err=%v", len(runs), listErr)
	}
}

func TestCreateRunClientRequestIDIsIdempotent(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.ClientRequestID = "4d0b4f2c-1fc5-40b0-b04e-876ad9d4d8e2"

	first, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("first CreateRun: %v", err)
	}
	second, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("replayed CreateRun: %v", err)
	}
	if first.ID != second.ID || second.ClientRequestID != request.ClientRequestID {
		t.Fatalf("idempotent create returned distinct runs: first=%+v second=%+v", first, second)
	}
	runs, err := service.store.ListRuns()
	if err != nil {
		t.Fatalf("ListRuns: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("ListRuns count=%d, want 1", len(runs))
	}

	changed := request
	changed.Name = "Different experiment"
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), changed); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed idempotent replay error=%v, want ErrConflict", err)
	}
}

func TestCreateRunRejectsInvalidClientRequestID(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.ClientRequestID = "retry-me"
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); !errors.Is(err, ErrInvalid) {
		t.Fatalf("CreateRun error=%v, want ErrInvalid", err)
	}
}

func TestCreateRunClientRequestIDCollapsesConcurrentRetries(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.ClientRequestID = "06f3556a-6769-4b33-b90f-4cf4e2b79a31"

	const attempts = 16
	ids := make(chan string, attempts)
	errs := make(chan error, attempts)
	var workers sync.WaitGroup
	for range attempts {
		workers.Add(1)
		go func() {
			defer workers.Done()
			run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
			if err != nil {
				errs <- err
				return
			}
			ids <- run.ID
		}()
	}
	workers.Wait()
	close(ids)
	close(errs)
	for err := range errs {
		t.Errorf("concurrent CreateRun: %v", err)
	}
	var first string
	for id := range ids {
		if first == "" {
			first = id
		}
		if id != first {
			t.Fatalf("concurrent retry returned run %q, want %q", id, first)
		}
	}
	runs, err := service.store.ListRuns()
	if err != nil || len(runs) != 1 {
		t.Fatalf("ListRuns count=%d err=%v, want one durable run", len(runs), err)
	}
}

func TestCatalogAndManifestRefreshFromTheSameCurrentConfigBytes(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	findRuntime := func(catalog Catalog) CatalogTarget {
		for _, target := range catalog.Targets {
			if target.ID == mixtureTargetID("default") {
				return target
			}
		}
		t.Fatal("default mixture target missing")
		return CatalogTarget{}
	}
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("initial Catalog: %v", err)
	}
	if got := findRuntime(catalog).TrackIDs; len(got) != 0 {
		t.Fatalf("initial runtime tracks=%v", got)
	}
	configBytes := []byte(modelArmTestYAML)
	if writeErr := os.WriteFile(filepath.Join(root, "config.yaml"), configBytes, 0o600); writeErr != nil {
		t.Fatalf("deploy updated config: %v", writeErr)
	}
	catalog, err = service.Catalog()
	if err != nil {
		t.Fatalf("refreshed Catalog: %v", err)
	}
	if got := findRuntime(catalog).TrackIDs; !reflect.DeepEqual(got, []TrackID{"routing", "model_pool", "joint", "multimodal", "capacity"}) {
		t.Fatalf("refreshed runtime tracks=%v", got)
	}
	run, err := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "multimodal snapshot", SuiteIDs: []string{"live-multimodal"}, TrackIDs: []TrackID{"multimodal"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "agent_multimodal",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	var manifest RunManifest
	if err := readJSON(filepath.Join(root, "runs", run.ID, manifestFileName), &manifest); err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Target.Mixture == nil || len(manifest.Target.Mixture.ModelArms) != 2 || manifest.ConfigDigest != digestBytes(configBytes) ||
		manifest.Target.Mixture.SupportModels == nil ||
		!digestPattern.MatchString(manifest.PolicySnapshotDigest) ||
		!reflect.DeepEqual(manifest.SuiteRevisions, map[string]string{"live-multimodal": "executor-v1"}) {
		t.Fatalf("manifest was not frozen from deployed bytes: %+v", manifest)
	}
	manifestBytes, readErr := os.ReadFile(filepath.Join(root, "runs", run.ID, manifestFileName))
	if readErr != nil || strings.Contains(string(manifestBytes), `"support_models":null`) {
		t.Fatalf("manifest did not serialize the empty support-model boundary as an array: err=%v body=%s", readErr, manifestBytes)
	}
	for _, suite := range catalog.Suites {
		if suite.ID == "live-model-pool" || suite.ID == "live-joint" {
			t.Fatalf("catalog retained unsupported placeholder suite %q", suite.ID)
		}
	}
}

func TestCatalogFailsClosedWhenCurrentConfigIsInvalid(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	if err := os.WriteFile(filepath.Join(root, "config.yaml"), []byte("not: [valid\n"), 0o600); err != nil {
		t.Fatalf("replace config: %v", err)
	}
	if _, err := service.Catalog(); err == nil {
		t.Fatal("Catalog returned a stale startup snapshot after current config became invalid")
	}
}

func TestBaselineCandidateCreationRequiresAComparableCohort(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	baseline, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create baseline: %v", err)
	}
	baseline = completeTestRun(t, service, baseline)
	service.codeRevision = strings.Repeat("b", 40)
	valid := validCreateRequest()
	valid.Name = "candidate"
	valid.BaselineRunID = baseline.ID
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), valid); err != nil {
		t.Fatalf("comparable candidate rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*CreateRunRequest)
	}{
		{name: "mode and target", mutate: func(r *CreateRunRequest) { r.Mode, r.TargetID = ModeLive, mixtureTargetID("default") }},
		{name: "suite", mutate: func(r *CreateRunRequest) { r.SuiteIDs = []string{"live-mom-core"} }},
		{name: "track", mutate: func(r *CreateRunRequest) { r.TrackIDs = []TrackID{"joint"} }},
		{name: "sample limit", mutate: func(r *CreateRunRequest) { r.SampleLimit++ }},
		{name: "concurrency", mutate: func(r *CreateRunRequest) { r.Concurrency++ }},
		{name: "seed", mutate: func(r *CreateRunRequest) { r.Seed++ }},
		{name: "change profile", mutate: func(r *CreateRunRequest) { r.ChangeProfile = "recipe" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := valid
			candidate.ClientRequestID = newTestClientRequestID()
			candidate.SuiteIDs = append([]string(nil), valid.SuiteIDs...)
			candidate.TrackIDs = append([]TrackID(nil), valid.TrackIDs...)
			test.mutate(&candidate)
			if _, err := service.CreateRunAs(context.Background(), SystemActor(), candidate); !errors.Is(err, ErrInvalid) {
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
	valid.ClientRequestID = newTestClientRequestID()
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), valid); err != nil {
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
				ClientRequestID: newTestClientRequestID(),
				Name:            "live policy baseline", SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
				Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: test.profile,
				SampleLimit: 4, Concurrency: 1, Seed: 17,
			}
			baseline, err := service.CreateRunAs(context.Background(), SystemActor(), request)
			if err != nil {
				t.Fatalf("create baseline: %v", err)
			}
			baseline = completeTestRun(t, service, baseline)
			changed := strings.Replace(modelArmTestYAML, "routing:\n  modelCards:", "routing:\n  strategy: confidence\n  modelCards:", 1)
			if writeErr := os.WriteFile(configPath, []byte(changed), 0o600); writeErr != nil {
				t.Fatalf("write candidate config: %v", writeErr)
			}
			request.Name = "live policy candidate"
			request.ClientRequestID = newTestClientRequestID()
			request.BaselineRunID = baseline.ID
			_, err = service.CreateRunAs(context.Background(), SystemActor(), request)
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("frozen policy change error=%v, want ErrInvalid", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("allowed policy treatment rejected: %v", err)
			}
		})
	}
}

func TestLiveSelectorCandidateRequiresOnlyTheSelectorFactor(t *testing.T) {
	tests := []struct {
		name       string
		profile    ChangeProfile
		mixedDelta bool
		wantErr    bool
	}{
		{name: "selector", profile: "selector"},
		{name: "recipe cannot claim selector", profile: "recipe", wantErr: true},
		{name: "pool cannot claim selector", profile: "model_pool", wantErr: true},
		{name: "schema cannot claim selector", profile: "schema_adapter", wantErr: true},
		{name: "selector rejects mixed recipe", profile: "selector", mixedDelta: true, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			configPath := filepath.Join(root, "config.yaml")
			if err := os.WriteFile(configPath, []byte(multiRecipeMixtureTestYAML), 0o600); err != nil {
				t.Fatalf("write baseline config: %v", err)
			}
			request := CreateRunRequest{
				ClientRequestID: newTestClientRequestID(), Name: "selector baseline",
				SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
				Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: test.profile,
				SampleLimit: 4, Concurrency: 1, Seed: 17,
			}
			baseline, err := service.CreateRunAs(context.Background(), SystemActor(), request)
			if err != nil {
				t.Fatalf("create baseline: %v", err)
			}
			baselineManifest, _, err := service.readDurableManifest(baseline.ID)
			if err != nil {
				t.Fatalf("read baseline manifest: %v", err)
			}
			baseline = completeTestRun(t, service, baseline)
			changed := strings.Replace(
				multiRecipeMixtureTestYAML,
				"    - name: selector\n      backend_refs:",
				"    - name: selector\n      provider_model_id: selector-runtime-v2\n      backend_refs:",
				1,
			)
			if test.mixedDelta {
				changed = strings.Replace(changed, "routing:\n  modelCards:", "routing:\n  strategy: confidence\n  modelCards:", 1)
			}
			if writeCandidateErr := os.WriteFile(configPath, []byte(changed), 0o600); writeCandidateErr != nil {
				t.Fatalf("write candidate config: %v", writeCandidateErr)
			}
			request.ClientRequestID = newTestClientRequestID()
			request.Name = "selector candidate"
			request.BaselineRunID = baseline.ID
			candidate, err := service.CreateRunAs(context.Background(), SystemActor(), request)
			if test.wantErr {
				if !errors.Is(err, ErrInvalid) {
					t.Fatalf("candidate error=%v, want ErrInvalid", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("selector treatment rejected: %v", err)
			}
			candidateManifest, _, err := service.readDurableManifest(candidate.ID)
			if err != nil {
				t.Fatalf("read candidate manifest: %v", err)
			}
			if baselineManifest.Target.Mixture == nil || candidateManifest.Target.Mixture == nil ||
				baselineManifest.Target.Mixture.SelectorDigest == candidateManifest.Target.Mixture.SelectorDigest ||
				baselineManifest.PolicySnapshotDigest != candidateManifest.PolicySnapshotDigest ||
				baselineManifest.Target.Mixture.BindingDigest != candidateManifest.Target.Mixture.BindingDigest ||
				baselineManifest.Target.Mixture.PoolDigest != candidateManifest.Target.Mixture.PoolDigest ||
				baselineManifest.Target.BackendTopologyDigest != candidateManifest.Target.BackendTopologyDigest {
				t.Fatalf("selector treatment crossed an orthogonal factor: baseline=%#v candidate=%#v", baselineManifest.Target, candidateManifest.Target)
			}
		})
	}
}

func TestLiveOnlineAdaptationCandidateRequiresOnlyAdaptationFactor(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write baseline config: %v", err)
	}
	request := CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "adaptation baseline",
		SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "online_adaptation",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	baseline, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("create baseline: %v", err)
	}
	baseline = completeTestRun(t, service, baseline)
	changed := strings.Replace(
		modelArmTestYAML,
		"      rules: {}\n      modelRefs:",
		"      rules: {}\n      adaptations: {mode: bypass}\n      modelRefs:",
		1,
	)
	if err := os.WriteFile(configPath, []byte(changed), 0o600); err != nil {
		t.Fatalf("write candidate config: %v", err)
	}
	request.ClientRequestID = newTestClientRequestID()
	request.Name = "adaptation candidate"
	request.BaselineRunID = baseline.ID
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); err != nil {
		t.Fatalf("online adaptation treatment rejected: %v", err)
	}
}
