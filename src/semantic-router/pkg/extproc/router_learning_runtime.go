package extproc

import (
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type routerLearningRuntime struct {
	leaseMu         sync.Mutex
	leaseRefs       sync.WaitGroup
	retired         bool
	shared          *routerLearningSharedState
	config          *config.RouterConfig
	replayRecorder  *routerreplay.Recorder
	replayRecorders map[string]*routerreplay.Recorder
}

// routerLearningSharedState survives hot reloads. Experience and idempotency
// are process-scoped learning state, while config and recorder bindings remain
// generation-scoped on routerLearningRuntime.
type routerLearningSharedState struct {
	mu              sync.Mutex
	experience      map[string]*routerLearningModelExperience
	idempotencyKeys map[string]*outcomeIdempotencyClaim
}

type outcomeIdempotencyClaim struct {
	done        chan struct{}
	committedAt time.Time
}

func (rt *routerLearningRuntime) resolveOutcomeDecisionContext(outcome *routerruntime.RouterOutcome) (string, int) {
	if outcome == nil {
		return "", 0
	}
	decision := strings.TrimSpace(outcome.Metadata["decision"])
	recipe := config.RecipeName(strings.TrimSpace(outcome.Metadata["recipe"]))
	tier := outcomeDecisionTier(outcome)
	if decision != "" && tier != 0 {
		return outcomeDecisionKey(recipe, decision), tier
	}
	record, ok := rt.replayRecord(outcome.ReplayID)
	if !ok {
		return outcomeDecisionKey(recipe, decision), tier
	}
	if decision == "" {
		decision = replayRecordDecision(record)
	}
	if recipe == "" {
		recipe = config.RecipeName(strings.TrimSpace(record.Recipe))
	}
	if tier == 0 {
		tier = replayRecordDecisionTier(record)
	}
	return outcomeDecisionKey(recipe, decision), tier
}

func replayRecordDecision(record routerreplay.RoutingRecord) string {
	if record.RouteDiagnostics != nil {
		if decision := strings.TrimSpace(record.RouteDiagnostics.Decision); decision != "" {
			return decision
		}
	}
	return strings.TrimSpace(record.Decision)
}

func replayRecordDecisionTier(record routerreplay.RoutingRecord) int {
	if record.RouteDiagnostics != nil && record.RouteDiagnostics.DecisionTier != 0 {
		return record.RouteDiagnostics.DecisionTier
	}
	return record.DecisionTier
}

func outcomeDecisionKey(recipe config.RecipeName, decision string) string {
	if decision == "" {
		return ""
	}
	return config.RoutingDecisionKey(recipe, decision)
}

func (rt *routerLearningRuntime) replayRecord(replayID string) (routerreplay.RoutingRecord, bool) {
	if rt == nil || strings.TrimSpace(replayID) == "" {
		return routerreplay.RoutingRecord{}, false
	}
	if rt.replayRecorder != nil {
		if record, found := rt.replayRecorder.GetRecord(replayID); found {
			return record, true
		}
	}
	seen := map[*routerreplay.Recorder]struct{}{}
	if rt.replayRecorder != nil {
		seen[rt.replayRecorder] = struct{}{}
	}
	for _, recorder := range rt.replayRecorders {
		if recorder == nil {
			continue
		}
		if _, ok := seen[recorder]; ok {
			continue
		}
		seen[recorder] = struct{}{}
		if record, found := recorder.GetRecord(replayID); found {
			return record, true
		}
	}
	return routerreplay.RoutingRecord{}, false
}

func outcomeDecisionTier(outcome *routerruntime.RouterOutcome) int {
	if outcome == nil || outcome.Metadata == nil {
		return 0
	}
	value := strings.TrimSpace(outcome.Metadata["decision_tier"])
	if value == "" {
		return 0
	}
	tier, err := strconv.Atoi(value)
	if err != nil || tier < 0 {
		return 0
	}
	return tier
}

func (rt *routerLearningRuntime) appendReplayOutcome(outcome *routerruntime.RouterOutcome) bool {
	if rt == nil || outcome == nil || strings.TrimSpace(outcome.ReplayID) == "" {
		return false
	}
	replayOutcome := routerReplayOutcome(outcome)
	if rt.replayRecorder != nil && rt.tryAppendReplayOutcome(rt.replayRecorder, outcome.ReplayID, replayOutcome) {
		return true
	}
	seen := map[*routerreplay.Recorder]struct{}{}
	if rt.replayRecorder != nil {
		seen[rt.replayRecorder] = struct{}{}
	}
	for _, recorder := range rt.replayRecorders {
		if recorder == nil {
			continue
		}
		if _, ok := seen[recorder]; ok {
			continue
		}
		seen[recorder] = struct{}{}
		if rt.tryAppendReplayOutcome(recorder, outcome.ReplayID, replayOutcome) {
			return true
		}
	}
	return false
}

func (rt *routerLearningRuntime) tryAppendReplayOutcome(
	recorder *routerreplay.Recorder,
	replayID string,
	outcome routerreplay.Outcome,
) bool {
	if recorder == nil || strings.TrimSpace(replayID) == "" {
		return false
	}
	if _, found := recorder.GetRecord(replayID); !found {
		return false
	}
	return recorder.AppendOutcome(replayID, outcome) == nil
}

func routerReplayOutcome(outcome *routerruntime.RouterOutcome) routerreplay.Outcome {
	return routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    string(outcome.Source),
		Target:    string(outcome.Target),
		TargetRef: strings.TrimSpace(outcome.TargetRef),
		Verdict:   string(outcome.Verdict),
		Reason:    strings.TrimSpace(outcome.Reason),
		Score:     outcome.Score,
		Metadata:  cloneLearningOutcomeMetadata(outcome.Metadata),
	}
}

func cloneLearningOutcomeMetadata(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

type routerLearningModelExperience struct {
	QualitySeed             float64
	SeedWeight              float64
	GoodFitCount            int
	UnderpoweredCount       int
	OverprovisionedCount    int
	FailedCount             int
	LatencyEWMA             float64
	CacheHitEWMA            float64
	CacheWriteEWMA          float64
	InputCostMultiplierEWMA float64
	LastUpdated             time.Time
}

func routerOutcomeVerdict(verdict routerruntime.RouterOutcomeVerdict) (routerLearningOutcomeVerdict, bool) {
	switch verdict {
	case routerruntime.RouterOutcomeVerdictGoodFit:
		return routerLearningOutcomeGoodFit, true
	case routerruntime.RouterOutcomeVerdictUnderpowered:
		return routerLearningOutcomeUnderpowered, true
	case routerruntime.RouterOutcomeVerdictOverprovisioned:
		return routerLearningOutcomeOverprovisioned, true
	case routerruntime.RouterOutcomeVerdictFailed:
		return routerLearningOutcomeFailed, true
	default:
		return "", false
	}
}

func newRouterLearningRuntime(
	cfg *config.RouterConfig,
	replayRecorder *routerreplay.Recorder,
	replayRecorders map[string]*routerreplay.Recorder,
) *routerLearningRuntime {
	return &routerLearningRuntime{
		shared: &routerLearningSharedState{
			experience:      map[string]*routerLearningModelExperience{},
			idempotencyKeys: map[string]*outcomeIdempotencyClaim{},
		},
		config:          cfg,
		replayRecorder:  replayRecorder,
		replayRecorders: replayRecorders,
	}
}

func inheritRouterLearningState(previous, next *OpenAIRouter) {
	if previous == nil || next == nil {
		return
	}
	previousRuntime := previous.routerLearningRuntimeState()
	nextRuntime := next.routerLearningRuntimeState()
	if previousRuntime == nil || nextRuntime == nil || previousRuntime.shared == nil {
		return
	}
	nextRuntime.shared = previousRuntime.shared
}

// AcquireLease keeps the runtime and its replay store alive for one
// management request. The registry acquires this lease while it still holds
// its read lock, so a concurrent reload cannot publish a replacement between
// pointer lookup and reference acquisition.
func (rt *routerLearningRuntime) AcquireLease() (func(), bool) {
	if rt == nil {
		return nil, false
	}
	rt.leaseMu.Lock()
	if rt.retired {
		rt.leaseMu.Unlock()
		return nil, false
	}
	rt.leaseRefs.Add(1)
	rt.leaseMu.Unlock()

	var once sync.Once
	return func() {
		once.Do(rt.leaseRefs.Done)
	}, true
}

// RetireAndWait prevents new management leases and waits for requests that
// already acquired this runtime before its replay store is closed.
func (rt *routerLearningRuntime) RetireAndWait() {
	if rt == nil {
		return
	}
	rt.leaseMu.Lock()
	rt.retired = true
	rt.leaseMu.Unlock()
	rt.leaseRefs.Wait()
}

func (r *OpenAIRouter) routerLearningRuntimeState() *routerLearningRuntime {
	if r == nil {
		return nil
	}
	r.routerLearningMu.Lock()
	defer r.routerLearningMu.Unlock()
	if r.routerLearningRuntime == nil {
		r.routerLearningRuntime = newRouterLearningRuntime(r.Config, r.ReplayRecorder, r.ReplayRecorders)
	}
	return r.routerLearningRuntime
}

func (rt *routerLearningRuntime) recordModelExperience(
	decisionName string,
	decisionTier int,
	model string,
	verdict routerLearningOutcomeVerdict,
	score float64,
) {
	if rt == nil || strings.TrimSpace(model) == "" {
		return
	}
	rt.shared.mu.Lock()
	defer rt.shared.mu.Unlock()
	rt.recordModelExperienceLocked(decisionName, decisionTier, model, verdict, score)
	if strings.TrimSpace(decisionName) != "" {
		rt.recordModelExperienceLocked("", decisionTier, model, verdict, score)
	}
	if decisionTier != 0 {
		rt.recordModelExperienceLocked("", 0, model, verdict, score)
	}
}

func (rt *routerLearningRuntime) recordModelExperienceLocked(
	decisionName string,
	decisionTier int,
	model string,
	verdict routerLearningOutcomeVerdict,
	score float64,
) {
	key := modelExperienceKey(decisionName, decisionTier, model)
	exp := rt.shared.experience[key]
	if exp == nil {
		exp = &routerLearningModelExperience{
			QualitySeed: 0.5,
			SeedWeight:  2,
		}
		rt.shared.experience[key] = exp
	}
	switch verdict {
	case routerLearningOutcomeGoodFit:
		exp.GoodFitCount += outcomeCount(score)
	case routerLearningOutcomeUnderpowered:
		exp.UnderpoweredCount += outcomeCount(score)
	case routerLearningOutcomeOverprovisioned:
		exp.OverprovisionedCount += outcomeCount(score)
	case routerLearningOutcomeFailed:
		exp.FailedCount += outcomeCount(score)
	}
	exp.LastUpdated = time.Now()
}

func outcomeCount(score float64) int {
	if score <= 0 {
		return 1
	}
	if score < 1 {
		return 1
	}
	return int(score)
}

func (rt *routerLearningRuntime) experienceSnapshot(decisionName string, decisionTier int, model string) routerLearningModelExperience {
	if rt == nil || strings.TrimSpace(model) == "" {
		return defaultRouterLearningModelExperience()
	}
	rt.shared.mu.Lock()
	defer rt.shared.mu.Unlock()
	for _, key := range []string{
		modelExperienceKey(decisionName, decisionTier, model),
		modelExperienceKey("", decisionTier, model),
		modelExperienceKey("", 0, model),
	} {
		if exp := rt.shared.experience[key]; exp != nil {
			return *exp
		}
	}
	return defaultRouterLearningModelExperience()
}

func defaultRouterLearningModelExperience() routerLearningModelExperience {
	return routerLearningModelExperience{
		QualitySeed: 0.5,
		SeedWeight:  2,
	}
}

func modelExperienceKey(decisionName string, decisionTier int, model string) string {
	decisionName = strings.TrimSpace(decisionName)
	model = strings.TrimSpace(model)
	if decisionName == "" {
		decisionName = "_global"
	}
	return decisionName + "|" + strconv.Itoa(decisionTier) + "|" + model
}
