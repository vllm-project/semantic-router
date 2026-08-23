package extproc

import (
	"context"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type routerLearningRuntime struct {
	leaseMu           sync.Mutex
	leaseRefs         sync.WaitGroup
	retired           bool
	shared            *routerLearningSharedState
	config            *config.RouterConfig
	replayRecorder    *routerreplay.Recorder
	replayRecorders   map[string]*routerreplay.Recorder
	outcomeProjection OutcomeLearningProjectionRuntime
}

// routerLearningSharedState retains best-effort runtime telemetry for
// standalone routing. Managed outcome evidence is read from the globally
// revisioned projection and never committed to this process-local map.
type routerLearningSharedState struct {
	mu         sync.Mutex
	experience map[string]*routerLearningModelExperience
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

func newRouterLearningRuntime(
	cfg *config.RouterConfig,
	replayRecorder *routerreplay.Recorder,
	replayRecorders map[string]*routerreplay.Recorder,
) *routerLearningRuntime {
	return &routerLearningRuntime{
		shared: &routerLearningSharedState{
			experience: map[string]*routerLearningModelExperience{},
		},
		config:          cfg,
		replayRecorder:  replayRecorder,
		replayRecorders: replayRecorders,
	}
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
		r.routerLearningRuntime.outcomeProjection = r.OutcomeProjection
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
		rt.recordModelExperienceLocked(modelExperienceFallbackDecision(decisionName), decisionTier, model, verdict, score)
	}
	if decisionTier != 0 {
		rt.recordModelExperienceLocked(modelExperienceFallbackDecision(decisionName), 0, model, verdict, score)
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
	fallbackDecision := modelExperienceFallbackDecision(decisionName)
	for _, key := range []string{
		modelExperienceKey(decisionName, decisionTier, model),
		modelExperienceKey(fallbackDecision, decisionTier, model),
		modelExperienceKey(fallbackDecision, 0, model),
	} {
		if exp := rt.shared.experience[key]; exp != nil {
			return *exp
		}
	}
	return defaultRouterLearningModelExperience()
}

// experienceSnapshotForRequest overlays only the durable, globally published
// outcome counters in managed mode. A request never treats process-local
// feedback state as authoritative, and a restarted replica reconstructs the
// same view from PostgreSQL through the revisioned Valkey projection.
func (rt *routerLearningRuntime) experienceSnapshotForRequest(
	request *RequestContext,
	decisionName string,
	decisionTier int,
	model string,
) routerLearningModelExperience {
	if rt == nil || rt.outcomeProjection == nil || request == nil ||
		request.InferenceAccess == nil || request.InferenceAccess.tenant.NamespaceID == "" {
		return rt.experienceSnapshot(decisionName, decisionTier, model)
	}
	result := defaultRouterLearningModelExperience()
	base := context.Background()
	if request.TraceContext != nil {
		base = request.TraceContext
	}
	ctx, cancel := context.WithTimeout(base, 100*time.Millisecond)
	defer cancel()
	projection, err := rt.outcomeProjection.Read(ctx, request.InferenceAccess.tenant.NamespaceID)
	if err != nil {
		return result
	}
	for _, entry := range projection.Entries {
		if !projectionEntryMatches(entry, decisionName, decisionTier, model) {
			continue
		}
		result.GoodFitCount = saturatingCount(result.GoodFitCount, entry.GoodFitCount)
		result.UnderpoweredCount = saturatingCount(result.UnderpoweredCount, entry.UnderpoweredCount)
		result.OverprovisionedCount = saturatingCount(result.OverprovisionedCount, entry.OverprovisionedCount)
		result.FailedCount = saturatingCount(result.FailedCount, entry.FailedCount)
	}
	return result
}

func projectionEntryMatches(
	entry outcomefeedback.ProjectionEntry,
	decisionName string,
	decisionTier int,
	model string,
) bool {
	if strings.TrimSpace(model) == "" ||
		(entry.ModelName != model && entry.ModelID != model) {
		return false
	}
	if decisionTier != 0 && entry.DecisionTier != decisionTier {
		return false
	}
	exact := config.RoutingDecisionKey(config.RecipeName(entry.RecipeName), entry.DecisionName)
	if decisionName == exact || decisionName == entry.DecisionName {
		return true
	}
	fallback := modelExperienceFallbackDecision(decisionName)
	return fallback != "" && fallback == modelExperienceFallbackDecision(exact)
}

func saturatingCount(current int, addition int64) int {
	if addition <= 0 {
		return current
	}
	maximum := int64(math.MaxInt)
	if int64(current) >= maximum-addition {
		return math.MaxInt
	}
	return current + int(addition)
}

// modelExperienceFallbackDecision keeps tier-wide and global experience inside
// the same routing namespace as an exact decision key. RoutingNamespaceKey
// escapes local names, so the final "::" delimiter unambiguously separates the
// runtime scope from the decision. Unscoped default-profile keys retain their
// existing process-wide fallback behavior.
func modelExperienceFallbackDecision(decisionName string) string {
	decisionName = strings.TrimSpace(decisionName)
	separator := strings.LastIndex(decisionName, "::")
	if separator <= 0 {
		return ""
	}
	return decisionName[:separator] + "::_global"
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
