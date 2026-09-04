package evaluationplane

import (
	"fmt"
	"net/url"
	"strings"
	"time"
)

var allTrackIDs = []TrackID{
	"routing", "model_pool", "joint", "agentic",
	"multimodal", "preference", "safety", "capacity",
}

type targetDefinition struct {
	Public                CatalogTarget
	RouterAPIURL          string
	EnvoyURL              string
	RouterAPIKey          *SecretRef
	EnvoyAPIKey           *SecretRef
	ModelArms             []ModelArm
	BackendTopologyDigest string
}

type Registry struct {
	tracks         map[TrackID]CatalogTrack
	suites         map[string]CatalogSuite
	targets        map[string]targetDefinition
	changeProfiles map[ChangeProfile]CatalogChangeProfile
}

type RegistryOptions struct {
	RouterAPIKey          *SecretRef
	EnvoyAPIKey           *SecretRef
	ModelArms             []ModelArm
	BackendTopologyDigest string
	RouterAuthRequired    bool
}

func NewRegistry(routerAPIURL, envoyURL string, registryOptions ...RegistryOptions) (*Registry, error) {
	options := RegistryOptions{}
	if len(registryOptions) > 0 {
		options = registryOptions[0]
	}
	if len(registryOptions) > 1 {
		return nil, fmt.Errorf("only one registry options value is accepted")
	}
	if err := validateTargetContract(options.RouterAPIKey, options.EnvoyAPIKey, options.ModelArms, options.BackendTopologyDigest); err != nil {
		return nil, err
	}
	registry := &Registry{
		tracks:         make(map[TrackID]CatalogTrack),
		suites:         make(map[string]CatalogSuite),
		targets:        make(map[string]targetDefinition),
		changeProfiles: make(map[ChangeProfile]CatalogChangeProfile),
	}
	for _, track := range builtinTracks() {
		registry.tracks[track.ID] = track
	}
	for _, suite := range builtinSuites() {
		registry.suites[suite.ID] = suite
	}
	for _, profile := range builtinChangeProfiles() {
		registry.changeProfiles[profile.ID] = profile
	}
	healthy := true
	registry.targets["fixture"] = targetDefinition{Public: CatalogTarget{
		ID: "fixture", Name: "Built-in replay fixture", Kind: "builtin-fixture",
		Description: "Deterministic evidence for validating the complete evaluation plane.",
		TrackIDs:    append([]TrackID(nil), allTrackIDs...), Modes: []Mode{ModeReplay},
		EvidenceLevel: "E0", Healthy: &healthy,
		Labels: map[string]string{"execution": "local", "network": "none"},
	}}

	routerAPIURL, err := normalizeServerURL(routerAPIURL)
	if err != nil {
		return nil, fmt.Errorf("router evaluation target: %w", err)
	}
	envoyURL, err = normalizeServerURL(envoyURL)
	if err != nil {
		return nil, fmt.Errorf("envoy evaluation target: %w", err)
	}
	runtimeRouterURL := routerAPIURL
	if options.RouterAuthRequired {
		runtimeRouterURL = ""
	}
	runtimeTrackIDs := runtimeTracks(runtimeRouterURL, envoyURL, options.ModelArms, options.BackendTopologyDigest)
	runtimeHealthy := len(runtimeTrackIDs) > 0
	runtimeLabels := map[string]string{
		"capabilities": "manifest-dependent",
		"credentials":  "environment-only",
		"direct_arms":  "unavailable",
		"model_arms":   "server-owned",
	}
	if options.RouterAuthRequired {
		runtimeLabels["router_auth"] = "dedicated-evaluation-credential-unavailable"
	}
	registry.targets["runtime"] = targetDefinition{
		Public: CatalogTarget{
			ID: "runtime", Name: "Active vLLM-SR runtime", Kind: "runtime",
			Description: "Capabilities derived from server-owned endpoints; model-pool and joint evaluation require an attested direct-arm target seam.",
			TrackIDs:    runtimeTrackIDs, Modes: []Mode{ModeLive}, Healthy: &runtimeHealthy,
			Labels: runtimeLabels,
		},
		RouterAPIURL:          runtimeRouterURL,
		EnvoyURL:              envoyURL,
		RouterAPIKey:          copySecretRef(options.RouterAPIKey),
		EnvoyAPIKey:           copySecretRef(options.EnvoyAPIKey),
		ModelArms:             copyModelArms(options.ModelArms),
		BackendTopologyDigest: options.BackendTopologyDigest,
	}
	return registry, nil
}

// runtimeTracks advertises only evidence the current server-owned endpoints
// can actually produce. ModelArms are snapshot metadata, not an attested
// direct-arm execution seam, so they cannot enable model-pool or joint tracks.
func runtimeTracks(routerAPIURL, envoyURL string, modelArms []ModelArm, backendTopologyDigest string) []TrackID {
	tracks := make([]TrackID, 0, 3)
	if !digestPattern.MatchString(backendTopologyDigest) {
		return tracks
	}
	if routerAPIURL != "" {
		tracks = append(tracks, "routing")
	}
	if envoyURL != "" {
		if hasMultimodalArm(modelArms) {
			tracks = append(tracks, "multimodal")
		}
		tracks = append(tracks, "capacity")
	}
	return tracks
}

func normalizeServerURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", nil
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", fmt.Errorf("must be an absolute http(s) URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("cannot contain credentials, query, or fragment")
	}
	return strings.TrimSuffix(parsed.String(), "/"), nil
}

func (r *Registry) Catalog() Catalog {
	changeProfiles := make([]CatalogChangeProfile, 0, len(r.changeProfiles))
	for _, definition := range builtinChangeProfiles() {
		if profile, ok := r.changeProfiles[definition.ID]; ok {
			changeProfiles = append(changeProfiles, profile)
		}
	}
	tracks := make([]CatalogTrack, 0, len(r.tracks))
	for _, id := range allTrackIDs {
		if track, ok := r.tracks[id]; ok {
			tracks = append(tracks, track)
		}
	}
	suites := make([]CatalogSuite, 0, len(r.suites))
	for _, definition := range builtinSuites() {
		if suite, ok := r.suites[definition.ID]; ok {
			suites = append(suites, suite)
		}
	}
	targets := make([]CatalogTarget, 0, len(r.targets))
	for _, id := range []string{"fixture", "runtime"} {
		if target, ok := r.targets[id]; ok {
			targets = append(targets, target.Public)
		}
	}
	return Catalog{
		SchemaVersion: SchemaVersion, GateContractVersion: GateContractVersion,
		GeneratedAt: time.Now().UTC(), ChangeProfiles: changeProfiles,
		Tracks: tracks, Suites: suites, Targets: targets,
	}
}

func (r *Registry) target(id string) (targetDefinition, bool) {
	target, ok := r.targets[id]
	return target, ok
}

func (r *Registry) suite(id string) (CatalogSuite, bool) {
	suite, ok := r.suites[id]
	return suite, ok
}

func (r *Registry) track(id TrackID) (CatalogTrack, bool) {
	track, ok := r.tracks[id]
	return track, ok
}

func (r *Registry) changeProfile(id ChangeProfile) (CatalogChangeProfile, bool) {
	profile, ok := r.changeProfiles[id]
	return profile, ok
}

func builtinChangeProfiles() []CatalogChangeProfile {
	return []CatalogChangeProfile{
		{ID: "schema_adapter", Name: "Schema / adapter", Description: "Strict schema and adapter parity changes."},
		{ID: "recipe", Name: "Routing recipe", Description: "Recipe signal, decision, algorithm, and policy changes."},
		{ID: "selector", Name: "Selector / binding", Description: "Selector, projection, classifier, and binding changes."},
		{ID: "model_pool", Name: "Model pool", Description: "Logical arm composition, capability, quality, and price changes."},
		{ID: "runtime_capacity", Name: "Runtime / capacity", Description: "Serving runtime, placement, capacity, and transport changes."},
		{ID: "agent_multimodal", Name: "Agent / multimodal", Description: "Agent trajectory, tool, state, and multimodal changes."},
		{ID: "online_adaptation", Name: "Online adaptation", Description: "Online assignment, preference, feedback, and adaptive policy changes."},
	}
}

func changeProfileRank(id ChangeProfile) int {
	for index, profile := range builtinChangeProfiles() {
		if profile.ID == id {
			return index
		}
	}
	return len(builtinChangeProfiles())
}

func validChangeProfile(id ChangeProfile) bool {
	return changeProfileRank(id) < len(builtinChangeProfiles())
}

func builtinTracks() []CatalogTrack {
	return []CatalogTrack{
		{ID: "routing", Name: "Routing", Description: "Recipe decisions, coverage, abstention, fallback, and oracle regret.", Modes: []Mode{ModeReplay, ModeLive}, Metrics: []string{"routing.coverage", "routing.accuracy", "routing.abstention_rate", "routing.fallback_rate", "routing.latency_p95_ms"}, EvidenceLevels: []EvidenceLevel{"E0", "E3"}},
		{ID: "model_pool", Name: "Model pool", Description: "Arm quality, complementarity, unique wins, and pool oracle quality.", Modes: []Mode{ModeReplay, ModeLive}, Metrics: []string{"model_pool.best_single_quality", "model_pool.oracle_quality", "model_pool.oracle_gain", "model_pool.unique_win_rate", "model_pool.selection_entropy_bits", "model_pool.selection_arm_coverage"}, EvidenceLevels: []EvidenceLevel{"E0", "E4"}},
		{ID: "joint", Name: "Routing + pool", Description: "Realized system utility, oracle regret, latency, reliability, and cost.", Modes: []Mode{ModeReplay, ModeLive}, Metrics: []string{"joint.realized_quality", "joint.oracle_regret", "joint.normalized_regret", "joint.reliability"}, EvidenceLevels: []EvidenceLevel{"E0", "E5"}},
		{ID: "agentic", Name: "Agentic", Description: "Trajectory success, tool validity, state continuity, and recovery.", Modes: []Mode{ModeReplay}, Metrics: []string{"agentic.success_rate", "agentic.invalid_tool_rate"}, EvidenceLevels: []EvidenceLevel{"E0"}},
		{ID: "multimodal", Name: "Multimodal", Description: "Capability-aware routing, grounding quality, and privacy signals.", Modes: []Mode{ModeReplay, ModeLive}, Metrics: []string{"multimodal.support_rate", "multimodal.quality"}, EvidenceLevels: []EvidenceLevel{"E0", "E5"}},
		{ID: "preference", Name: "Preference", Description: "Offline preference agreement and propensity-qualified online evidence.", Modes: []Mode{ModeReplay}, Metrics: []string{"preference.agreement", "preference.propensity_coverage"}, EvidenceLevels: []EvidenceLevel{"E0"}},
		{ID: "safety", Name: "Safety", Description: "Policy adherence, blocking correctness, privacy, and unsafe regressions.", Modes: []Mode{ModeReplay}, Metrics: []string{"safety.violation_rate", "safety.violation_upper_95", "safety.block_accuracy"}, EvidenceLevels: []EvidenceLevel{"E0"}},
		{ID: "capacity", Name: "Capacity", Description: "Throughput, tail latency, success envelope, GPU efficiency, and TCO.", Modes: []Mode{ModeReplay, ModeLive}, Metrics: []string{"capacity.throughput_rps", "capacity.latency_p95_ms", "capacity.success_rate", "capacity.cost_per_successful_request"}, EvidenceLevels: []EvidenceLevel{"E0", "E5"}},
	}
}

func builtinSuites() []CatalogSuite {
	return []CatalogSuite{
		{ID: "evaluation-smoke", Name: "Evaluation smoke", Description: "Deterministic all-track vertical slice.", TrackIDs: append([]TrackID(nil), allTrackIDs...), Modes: []Mode{ModeReplay}, EvidenceLevel: "E0", CaseCount: 4, Revision: "builtin-v1", Tags: []string{"smoke", "deterministic"}},
		{ID: "live-routing-core", Name: "Live routing core", Description: "Diagnostic routing smoke using bounded live probes; no promotion-grade policy claim.", TrackIDs: []TrackID{"routing"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E0", Revision: "executor-v1"},
		{ID: "live-model-pool", Name: "Live model pool", Description: "Requires an attested server-owned direct-arm matrix target; unavailable on the generic runtime target.", TrackIDs: []TrackID{"model_pool"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E0", Revision: "executor-v1"},
		{ID: "live-joint", Name: "Live routing + pool", Description: "Requires attested route correlation and direct-arm execution; unavailable on the generic runtime target.", TrackIDs: []TrackID{"routing", "model_pool", "joint"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E0", Revision: "executor-v1"},
		{ID: "live-multimodal", Name: "Live multimodal", Description: "Diagnostic single-probe multimodal smoke; no grounding, privacy, or robustness claim.", TrackIDs: []TrackID{"multimodal"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E0", Revision: "executor-v1"},
		{ID: "live-capacity", Name: "Live capacity", Description: "Diagnostic bounded concurrency smoke without warmup, repeats, duration, or a declared SLO.", TrackIDs: []TrackID{"capacity"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E0", Revision: "executor-v1"},
	}
}

func containsMode(modes []Mode, want Mode) bool {
	for _, mode := range modes {
		if mode == want {
			return true
		}
	}
	return false
}

func containsTrack(tracks []TrackID, want TrackID) bool {
	for _, track := range tracks {
		if track == want {
			return true
		}
	}
	return false
}
