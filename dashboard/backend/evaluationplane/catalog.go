package evaluationplane

import (
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"
)

var allTrackIDs = []TrackID{
	"routing", "model_pool", "joint", "agentic",
	"multimodal", "preference", "safety", "capacity",
}

type targetDefinition struct {
	Public                     CatalogTarget
	Contract                   targetContract
	RouterAPIURL               string
	EnvoyURL                   string
	RouterAPIKey               *SecretRef
	EnvoyAPIKey                *SecretRef
	AgentTaskLedger            *ServiceEndpoint
	FaultRecoveryLedger        *ServiceEndpoint
	HardPolicyLedger           *ServiceEndpoint
	ProductionExperimentLedger *ServiceEndpoint
	Mixture                    *ManifestMixture
	ConfigDigest               string
	BackendTopologyDigest      string
	Features                   []targetFeature
}

type Registry struct {
	tracks         map[TrackID]CatalogTrack
	suites         map[string]CatalogSuite
	suiteOrder     []string
	executors      map[string]executorContract
	targets        map[string]targetDefinition
	targetOrder    []string
	changeProfiles map[ChangeProfile]CatalogChangeProfile
}

type RegistryOptions struct {
	RouterAPIKey               *SecretRef
	EnvoyAPIKey                *SecretRef
	AgentTaskLedger            *ServiceEndpoint
	FaultRecoveryLedger        *ServiceEndpoint
	HardPolicyLedger           *ServiceEndpoint
	ProductionExperimentLedger *ServiceEndpoint
	Mixtures                   []MixtureTargetSnapshot
	DeploymentTargets          []DeploymentTargetSnapshot
	DefaultConfigDigest        string
	RouterAuthRequired         bool
	InstalledSuites            []CatalogSuite
}

func NewRegistry(routerAPIURL, envoyURL string, registryOptions ...RegistryOptions) (*Registry, error) {
	if err := ValidateMetricAnalysisCatalog(); err != nil {
		return nil, fmt.Errorf("metric analysis catalog: %w", err)
	}
	if err := ValidateResearchBenchmarkInventory(); err != nil {
		return nil, fmt.Errorf("research benchmark inventory: %w", err)
	}
	options, err := resolveRegistryOptions(registryOptions)
	if err != nil {
		return nil, err
	}
	if err := validateRegistryOptions(options); err != nil {
		return nil, err
	}
	if err := validateRegistryOrigins(routerAPIURL, envoyURL, options); err != nil {
		return nil, err
	}
	registry := emptyRegistry()
	if err := registry.registerCatalogDefinitions(options); err != nil {
		return nil, err
	}
	if err := registry.registerRecordedTargets(len(options.InstalledSuites) > 0); err != nil {
		return nil, err
	}
	if err := registry.registerMixtureTargets(routerAPIURL, envoyURL, options); err != nil {
		return nil, err
	}
	return registry, nil
}

func validateServerOrigin(raw string) error {
	if raw == "" {
		return nil
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("must be an absolute http(s) URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return fmt.Errorf("cannot contain credentials, query, or fragment")
	}
	if parsed.Path != "" || parsed.RawPath != "" || parsed.String() != raw {
		return fmt.Errorf("must be an exact canonical origin without whitespace, a trailing slash, or an API path")
	}
	return nil
}

func serverOriginKey(raw string) (string, error) {
	if err := validateServerOrigin(raw); err != nil {
		return "", err
	}
	if raw == "" {
		return "", fmt.Errorf("origin is required")
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("parse origin: %w", err)
	}
	scheme := strings.ToLower(parsed.Scheme)
	port := parsed.Port()
	if port == "" {
		if scheme == "http" {
			port = "80"
		} else {
			port = "443"
		}
	}
	return scheme + "://" + net.JoinHostPort(strings.ToLower(parsed.Hostname()), port), nil
}

func serverOriginsDistinct(left, right string) (bool, error) {
	if left == "" || right == "" {
		return false, nil
	}
	leftKey, err := serverOriginKey(left)
	if err != nil {
		return false, err
	}
	rightKey, err := serverOriginKey(right)
	if err != nil {
		return false, err
	}
	return leftKey != rightKey, nil
}

func (r *Registry) Catalog() Catalog {
	changeProfiles := make([]CatalogChangeProfile, 0, len(r.changeProfiles))
	for _, definition := range builtinChangeProfiles() {
		if profile, ok := r.changeProfiles[definition.ID]; ok {
			changeProfiles = append(changeProfiles, copyCatalogChangeProfile(profile))
		}
	}
	tracks := make([]CatalogTrack, 0, len(r.tracks))
	for _, id := range allTrackIDs {
		if track, ok := r.tracks[id]; ok {
			tracks = append(tracks, track)
		}
	}
	suites := make([]CatalogSuite, 0, len(r.suites))
	for _, id := range r.suiteOrder {
		if suite, ok := r.suites[id]; ok {
			suites = append(suites, copyCatalogSuite(suite))
		}
	}
	targets := make([]CatalogTarget, 0, len(r.targets))
	for _, id := range r.targetOrder {
		if target, ok := r.targets[id]; ok {
			targets = append(targets, copyCatalogTarget(target.Public))
		}
	}
	return Catalog{
		SchemaVersion: SchemaVersion, GateContractVersion: GateContractVersion,
		GeneratedAt: time.Now().UTC(), ChangeProfiles: changeProfiles,
		Tracks: tracks, Suites: suites, Targets: targets,
	}
}

func copyCatalogTarget(target CatalogTarget) CatalogTarget {
	if target.TrackIDs != nil {
		target.TrackIDs = append(make([]TrackID, 0, len(target.TrackIDs)), target.TrackIDs...)
	}
	target.Modes = append([]Mode(nil), target.Modes...)
	target.AcceptedExecutors = copyAcceptedExecutors(target.AcceptedExecutors)
	target.Labels = copyCatalogLabels(target.Labels)
	target.Mixture = copyCatalogMixture(target.Mixture)
	if target.Healthy != nil {
		healthy := *target.Healthy
		target.Healthy = &healthy
	}
	return target
}

func copyCatalogChangeProfile(profile CatalogChangeProfile) CatalogChangeProfile {
	profile.CampaignSlots = append([]CatalogCampaignSlot(nil), profile.CampaignSlots...)
	for index := range profile.CampaignSlots {
		profile.CampaignSlots[index].AcceptedExecutorIDs = append(
			[]string(nil), profile.CampaignSlots[index].AcceptedExecutorIDs...,
		)
	}
	return profile
}

func catalogMixtureFromManifest(mixture *ManifestMixture) *CatalogMixture {
	if mixture == nil {
		return nil
	}
	return &CatalogMixture{
		ID: mixture.ID, EntrypointModel: mixture.EntrypointModel,
		Aliases:    append([]string(nil), mixture.Aliases...),
		RecipeName: mixture.RecipeName, RecipeDescription: mixture.RecipeDescription,
		RecipeDigest: mixture.RecipeDigest, PoolDigest: mixture.PoolDigest,
		SelectorPolicyDigest: mixture.SelectorPolicyDigest, SelectorDigest: mixture.SelectorDigest,
		AdaptationDigest: mixture.AdaptationDigest, BindingDigest: mixture.BindingDigest, ModelArms: copyModelArms(mixture.ModelArms),
		SupportModels:     copySupportModels(mixture.SupportModels),
		FallbackArmID:     mixture.FallbackArmID,
		Decisions:         copyMixtureDecisions(mixture.Decisions),
		RoutingRecipePlan: copyRoutingRecipePlan(mixture.RoutingRecipePlan),
	}
}

func manifestMixtureFromCatalog(mixture *CatalogMixture) *ManifestMixture {
	if mixture == nil {
		return nil
	}
	return &ManifestMixture{
		SchemaVersion: SchemaVersion,
		ID:            mixture.ID, EntrypointModel: mixture.EntrypointModel,
		Aliases:    append([]string(nil), mixture.Aliases...),
		RecipeName: mixture.RecipeName, RecipeDescription: mixture.RecipeDescription,
		RecipeDigest: mixture.RecipeDigest, PoolDigest: mixture.PoolDigest,
		SelectorPolicyDigest: mixture.SelectorPolicyDigest, SelectorDigest: mixture.SelectorDigest,
		AdaptationDigest: mixture.AdaptationDigest, BindingDigest: mixture.BindingDigest, ModelArms: copyModelArms(mixture.ModelArms),
		SupportModels:     copySupportModels(mixture.SupportModels),
		FallbackArmID:     mixture.FallbackArmID,
		Decisions:         copyMixtureDecisions(mixture.Decisions),
		RoutingRecipePlan: copyRoutingRecipePlan(mixture.RoutingRecipePlan),
	}
}

func copyCatalogMixture(mixture *CatalogMixture) *CatalogMixture {
	if mixture == nil {
		return nil
	}
	copy := *mixture
	copy.Aliases = append([]string(nil), mixture.Aliases...)
	copy.ModelArms = copyModelArms(mixture.ModelArms)
	copy.SupportModels = copySupportModels(mixture.SupportModels)
	copy.Decisions = copyMixtureDecisions(mixture.Decisions)
	copy.RoutingRecipePlan = copyRoutingRecipePlan(mixture.RoutingRecipePlan)
	return &copy
}

func copyManifestMixture(mixture *ManifestMixture) *ManifestMixture {
	if mixture == nil {
		return nil
	}
	copy := *mixture
	copy.Aliases = append([]string(nil), mixture.Aliases...)
	copy.ModelArms = copyModelArms(mixture.ModelArms)
	copy.SupportModels = copySupportModels(mixture.SupportModels)
	copy.Decisions = copyMixtureDecisions(mixture.Decisions)
	copy.RoutingRecipePlan = copyRoutingRecipePlan(mixture.RoutingRecipePlan)
	return &copy
}

func copyRoutingRecipePlan(plan RoutingRecipePlan) RoutingRecipePlan {
	plan.ArmIDs = append([]string{}, plan.ArmIDs...)
	plan.Signals = append([]RoutingRecipeInputSpec{}, plan.Signals...)
	plan.Projections = append([]RoutingRecipeProjectionSpec{}, plan.Projections...)
	plan.TopK = append([]int{}, plan.TopK...)
	return plan
}

func copySupportModels(models []SupportModel) []SupportModel {
	result := make([]SupportModel, len(models))
	for index, model := range models {
		result[index] = model
		result[index].RuntimeRevision = copyStringPointer(model.RuntimeRevision)
	}
	return result
}

func copyMixtureDecisions(decisions []MixtureDecisionBinding) []MixtureDecisionBinding {
	result := make([]MixtureDecisionBinding, len(decisions))
	for index, decision := range decisions {
		result[index] = decision
		result[index].ArmIDs = append([]string(nil), decision.ArmIDs...)
	}
	return result
}

func copyCatalogSuite(suite CatalogSuite) CatalogSuite {
	if suite.CampaignProtocol != nil {
		protocol := *suite.CampaignProtocol
		suite.CampaignProtocol = &protocol
	}
	executors := suite.Executors
	suite.Executors = make(map[Mode]string, len(suite.Executors))
	for mode, executorID := range executors {
		suite.Executors[mode] = executorID
	}
	suite.TrackIDs = append([]TrackID(nil), suite.TrackIDs...)
	suite.Modes = append([]Mode(nil), suite.Modes...)
	// The wire contract uses an empty array, never null, for an intentionally
	// untagged suite. Preserve that distinction while returning a defensive copy.
	suite.Tags = append([]string{}, suite.Tags...)
	suite.Methods = append([]CatalogMethod(nil), suite.Methods...)
	for index := range suite.Methods {
		suite.Methods[index].QualifiedGateIDs = append([]string{}, suite.Methods[index].QualifiedGateIDs...)
	}
	return suite
}

func copyAcceptedExecutors(source map[Mode][]string) map[Mode][]string {
	result := make(map[Mode][]string, len(source))
	for mode, executors := range source {
		result[mode] = append([]string(nil), executors...)
	}
	return result
}

func copyCatalogLabels(source map[string]string) map[string]string {
	if source == nil {
		return nil
	}
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func (r *Registry) target(id string) (targetDefinition, bool) {
	target, ok := r.targets[id]
	return copyTargetDefinition(target), ok
}

func (r *Registry) suite(id string) (CatalogSuite, bool) {
	suite, ok := r.suites[id]
	return copyCatalogSuite(suite), ok
}

func (r *Registry) track(id TrackID) (CatalogTrack, bool) {
	track, ok := r.tracks[id]
	return track, ok
}

func (r *Registry) changeProfile(id ChangeProfile) (CatalogChangeProfile, bool) {
	profile, ok := r.changeProfiles[id]
	return copyCatalogChangeProfile(profile), ok
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
		builtinTrack("routing", "Routing", "Decision quality, coverage, abstention, fallbacks, and missed best-model opportunities.", []EvidenceLevel{"E0", "E3", "E4"}),
		builtinTrack("model_pool", "Model pool", "Quality and reliability of each model, complementary strengths, unique wins, and the best possible pool outcome.", []EvidenceLevel{"E0", "E4"}),
		builtinTrack("joint", "Routing and model pool", "End-to-end quality, reliability, latency, and cost, including the gap from the best available model.", []EvidenceLevel{"E0", "E5"}),
		builtinTrack("agentic", "Agent tasks", "Task completion, tool-use policy, state and privacy, recovery from failures, latency, and cost.", []EvidenceLevel{"E0", "E5"}),
		builtinTrack("multimodal", "Multimodal", "Input capability matching, grounded response quality, reliability, and privacy for text and non-text requests.", []EvidenceLevel{"E0", "E4", "E5"}),
		builtinTrack("preference", "Preference", "Offline preference agreement and statistically valid online preference outcomes.", []EvidenceLevel{"E0", "E4", "E5"}),
		builtinTrack("safety", "Safety", "Policy adherence, correct blocking behavior, privacy, and unsafe regressions.", []EvidenceLevel{"E0", "E3", "E4"}),
		builtinTrack("capacity", "Capacity", "Throughput, tail latency, error bounds, stability, service-objective headroom, and test cost across repeated load levels.", []EvidenceLevel{"E0", "E5"}),
	}
}

func builtinTrack(id TrackID, name, description string, evidenceLevels []EvidenceLevel) CatalogTrack {
	return CatalogTrack{
		ID: id, Name: name, Description: description,
		Modes: []Mode{ModeReplay, ModeLive}, Metrics: StaticMetricAnalysisIDsForTrack(id),
		EvidenceLevels: append([]EvidenceLevel(nil), evidenceLevels...),
	}
}

func builtinSuites() []CatalogSuite {
	return []CatalogSuite{
		{ID: "evaluation-smoke", Name: "Evaluation setup check", Description: "A small deterministic workload that verifies every evaluation area is connected and reportable.", Executors: map[Mode]string{ModeReplay: fixtureReplayExecutorID}, TrackIDs: append([]TrackID(nil), allTrackIDs...), Modes: []Mode{ModeReplay}, EvidenceLevel: "E0", CaseCount: 4, Revision: "builtin-v1", Tags: []string{"smoke", "deterministic"}, Methods: fixtureCatalogMethods()},
		{ID: "live-mom-core", Name: "Routing and model-pool setup check", Description: "A small hidden-answer workload for diagnosing routing, model-pool, and end-to-end execution. It is not large or diverse enough to support a release comparison.", Executors: map[Mode]string{ModeReplay: momReplayExecutorID, ModeLive: liveRuntimeExecutorID}, TrackIDs: []TrackID{"routing", "model_pool", "joint"}, Modes: []Mode{ModeReplay, ModeLive}, EvidenceLevel: "E0", CaseCount: 64, Revision: "mom-diagnostic-cohort-v2", Tags: []string{"smoke", "mom", "hidden-label", "diagnostic-only"}, Methods: []CatalogMethod{
			configuredCatalogMethod("routing.live-diagnostic.v1", "routing", nil, CatalogMethodEvidenceSourceLiveRuntime),
			configuredCatalogMethod("model-pool.live-dense.v1", "model_pool", nil, CatalogMethodEvidenceSourceLiveRuntime),
			configuredCatalogMethod("joint.live-routed-outcome.v1", "joint", nil, CatalogMethodEvidenceSourceLiveRuntime),
		}},
		{ID: "live-agent-tasks", Name: "Agent task evaluation", Description: "Repeated tool-use and reasoning tasks with complete provider results. Measures task completion, tool-policy compliance, reliability, latency, and cost; it does not invoke tools itself or claim parity with external agent benchmarks.", Executors: map[Mode]string{ModeLive: liveRuntimeExecutorID}, TrackIDs: []TrackID{"agentic"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E5", Revision: "executor-v1", Tags: []string{}, Methods: []CatalogMethod{dataRequiredCatalogMethod("live-agent-task.v1", "agentic", nil, CatalogMethodEvidenceSourceLiveRuntime, "Connect a managed agent-task results source that includes every repeated attempt, the required tool policy for each task, and provider-confirmed outcomes for the selected Mixture.")}},
		{ID: "live-fault-recovery", Name: "Agent fault-recovery evaluation", Description: "Matched baseline and injected-failure tasks that measure recovery, state continuity, side effects, retries, and latency.", Executors: map[Mode]string{ModeLive: liveRuntimeExecutorID}, TrackIDs: []TrackID{"agentic"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E5", Revision: "executor-v1", Tags: []string{}, Methods: []CatalogMethod{dataRequiredCatalogMethod("live-fault-recovery.v1", "agentic", []string{"G6"}, CatalogMethodEvidenceSourceLiveRuntime, "Connect a managed fault-recovery results source with complete matched baseline and injected-failure attempts at the same task step.")}},
		{ID: "live-multimodal", Name: "Multimodal response evaluation", Description: "Text and non-text requests graded for supported input handling, response quality, reliability, and latency.", Executors: map[Mode]string{ModeLive: liveRuntimeExecutorID}, TrackIDs: []TrackID{"multimodal"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E0", Revision: "executor-v1", Tags: []string{}, Methods: []CatalogMethod{configuredCatalogMethod("multimodal.live-chat.v1", "multimodal", nil, CatalogMethodEvidenceSourceLiveRuntime)}},
		{ID: "live-hard-policy", Name: "Policy enforcement evaluation", Description: "Live policy and adversarial cases that verify required rules are enforced by the selected system configuration.", Executors: map[Mode]string{ModeLive: liveRuntimeExecutorID}, TrackIDs: []TrackID{"safety"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E4", Revision: "executor-v1", Tags: []string{}, Methods: []CatalogMethod{dataRequiredCatalogMethod("policy.hard-enforcement.v1", "safety", []string{"G2"}, CatalogMethodEvidenceSourceLiveRuntime, "Connect managed policy-test results with the evaluated rules, enforcement points, and complete outcomes for the selected configuration.")}},
		{ID: "live-production-experiment", Name: "Guarded production evaluation", Description: "Evaluates connected production experiment results for assignment balance, exposure controls, risk, stop conditions, rollback readiness, and preference lift between baseline and candidate.", Executors: map[Mode]string{ModeLive: liveRuntimeExecutorID}, TrackIDs: []TrackID{"preference"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E5", Revision: "executor-v1", Tags: []string{}, Methods: []CatalogMethod{
			dataRequiredCatalogMethod("production.experiment-controls.v1", "preference", []string{"G8"}, CatalogMethodEvidenceSourceLiveProduction, "Connect managed production experiment results with complete assignment, exposure, and safety-control data."),
			dataRequiredCatalogMethod("production.preference-lift.v1", "preference", []string{"G9"}, CatalogMethodEvidenceSourceLiveProduction, "Connect managed production experiment results with complete preference outcomes and the recorded assignment probability for each policy."),
		}},
		{ID: "live-capacity", Name: "Capacity setup check", Description: "A short repeated closed-loop workload for checking load execution, telemetry, and report generation. It is diagnostic and does not support a release capacity decision.", Executors: map[Mode]string{ModeLive: liveRuntimeExecutorID}, TrackIDs: []TrackID{"capacity"}, Modes: []Mode{ModeLive}, EvidenceLevel: "E0", Revision: "executor-v1", Tags: []string{"smoke", "diagnostic-only"}, Methods: []CatalogMethod{configuredCatalogMethod("capacity.slo-envelope.v1", "capacity", nil, CatalogMethodEvidenceSourceLiveRuntime)}},
	}
}

func fixtureCatalogMethods() []CatalogMethod {
	methods := make([]CatalogMethod, 0, len(allTrackIDs))
	for _, trackID := range allTrackIDs {
		methods = append(methods, configuredCatalogMethod("fixture."+string(trackID)+".v1", trackID, nil, CatalogMethodEvidenceSourceDiagnosticFixture))
	}
	return methods
}

func configuredCatalogMethod(id string, trackID TrackID, gateIDs []string, source CatalogMethodEvidenceSource) CatalogMethod {
	return CatalogMethod{ID: id, TrackID: trackID, QualifiedGateIDs: append([]string{}, gateIDs...), EvidenceSource: source, Status: "configured"}
}

func dataRequiredCatalogMethod(id string, trackID TrackID, gateIDs []string, source CatalogMethodEvidenceSource, reason string) CatalogMethod {
	return CatalogMethod{ID: id, TrackID: trackID, QualifiedGateIDs: append([]string{}, gateIDs...), EvidenceSource: source, Status: "data_required", Reason: reason}
}

func builtinSuitesFor(options RegistryOptions) []CatalogSuite {
	suites := builtinSuites()
	ready := map[string]bool{
		"live-agent-task.v1":                options.AgentTaskLedger != nil,
		"live-fault-recovery.v1":            options.FaultRecoveryLedger != nil,
		"policy.hard-enforcement.v1":        options.HardPolicyLedger != nil,
		"production.experiment-controls.v1": options.ProductionExperimentLedger != nil,
		"production.preference-lift.v1":     options.ProductionExperimentLedger != nil,
	}
	for suiteIndex := range suites {
		for methodIndex := range suites[suiteIndex].Methods {
			if ready[suites[suiteIndex].Methods[methodIndex].ID] {
				suites[suiteIndex].Methods[methodIndex].Status = "configured"
				suites[suiteIndex].Methods[methodIndex].Reason = ""
			}
		}
	}
	return suites
}

func validNormalizedSuiteExecutors(suite CatalogSuite, executors map[string]executorContract) bool {
	expectedModes := []Mode{ModeReplay}
	if len(normalizedSuiteLiveMethodTracks(suite)) > 0 {
		expectedModes = append(expectedModes, ModeLive)
	}
	if len(suite.Modes) != len(expectedModes) || len(suite.Executors) != len(expectedModes) {
		return false
	}
	for index, mode := range expectedModes {
		if suite.Modes[index] != mode {
			return false
		}
		executorID := suite.Executors[mode]
		if mode == ModeReplay && executorID != normalizedSuiteExecutorID ||
			mode == ModeLive && executorID != normalizedSuiteLiveExecutorID {
			return false
		}
		executor, registered := executors[executorID]
		if !registered || executor.Mode != mode || !executor.NormalizedSuite ||
			(mode == ModeReplay) != executor.RecordedNormalizedSource {
			return false
		}
	}
	return true
}

func suiteExecutorForMode(suite CatalogSuite, mode Mode) (string, bool) {
	executor, ok := suite.Executors[mode]
	return executor, ok && portableIDPattern.MatchString(executor) && containsMode(suite.Modes, mode)
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
