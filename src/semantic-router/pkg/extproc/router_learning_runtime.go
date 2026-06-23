package extproc

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type routerLearningRuntime struct {
	mu              sync.Mutex
	config          *config.RouterConfig
	replayRecorder  *routerreplay.Recorder
	replayRecorders map[string]*routerreplay.Recorder
	experience      map[string]*routerLearningModelExperience
}

func (rt *routerLearningRuntime) UpdateOutcome(
	_ context.Context,
	outcome *routerruntime.RouterOutcome,
) routerruntime.RouterOutcomeResult {
	result := routerruntime.RouterOutcomeResult{}
	if rt == nil || outcome == nil || outcome.Target != routerruntime.RouterOutcomeTargetModel {
		result.Recorded = rt.appendReplayOutcome(outcome)
		return result
	}
	verdict, ok := routerOutcomeVerdict(outcome.Verdict)
	if !ok {
		result.Recorded = rt.appendReplayOutcome(outcome)
		return result
	}
	model := rt.resolveOutcomeModel(outcome)
	if model == "" {
		result.Recorded = rt.appendReplayOutcome(outcome)
		return result
	}
	outcome.TargetRef = model
	result.Recorded = rt.appendReplayOutcome(outcome)
	decisionName, decisionTier := rt.resolveOutcomeDecisionContext(outcome)
	rt.recordModelExperience(
		decisionName,
		decisionTier,
		model,
		verdict,
		outcome.Score,
	)
	result.Updated = 1
	return result
}

func (rt *routerLearningRuntime) resolveOutcomeModel(outcome *routerruntime.RouterOutcome) string {
	if outcome == nil {
		return ""
	}
	if model := strings.TrimSpace(outcome.TargetRef); model != "" {
		return model
	}
	if model := strings.TrimSpace(outcome.Metadata["model"]); model != "" {
		return model
	}
	if model := strings.TrimSpace(outcome.Metadata["selected_model"]); model != "" {
		return model
	}
	record, ok := rt.replayRecord(outcome.ReplayID)
	if !ok {
		return ""
	}
	if model := strings.TrimSpace(record.SelectedModel); model != "" {
		return model
	}
	if record.RouteDiagnostics != nil {
		return strings.TrimSpace(record.RouteDiagnostics.SelectedModel)
	}
	return ""
}

func (rt *routerLearningRuntime) resolveOutcomeDecisionContext(outcome *routerruntime.RouterOutcome) (string, int) {
	if outcome == nil {
		return "", 0
	}
	decision := strings.TrimSpace(outcome.Metadata["decision"])
	tier := outcomeDecisionTier(outcome)
	if decision != "" && tier != 0 {
		return decision, tier
	}
	record, ok := rt.replayRecord(outcome.ReplayID)
	if !ok {
		return decision, tier
	}
	if decision == "" {
		decision = strings.TrimSpace(record.Decision)
		if record.RouteDiagnostics != nil && strings.TrimSpace(record.RouteDiagnostics.Decision) != "" {
			decision = strings.TrimSpace(record.RouteDiagnostics.Decision)
		}
	}
	if tier == 0 {
		tier = record.DecisionTier
		if record.RouteDiagnostics != nil && record.RouteDiagnostics.DecisionTier != 0 {
			tier = record.RouteDiagnostics.DecisionTier
		}
	}
	return decision, tier
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
		config:          cfg,
		replayRecorder:  replayRecorder,
		replayRecorders: replayRecorders,
		experience:      map[string]*routerLearningModelExperience{},
	}
}

func (r *OpenAIRouter) routerLearningRuntimeState() *routerLearningRuntime {
	if r == nil {
		return nil
	}
	r.routerLearningMu.Lock()
	defer r.routerLearningMu.Unlock()
	if r.routerLearningRuntime == nil {
		r.routerLearningRuntime = newRouterLearningRuntime(r.Config, r.ReplayRecorder, r.ReplayRecorders)
	} else {
		r.routerLearningRuntime.config = r.Config
		r.routerLearningRuntime.replayRecorder = r.ReplayRecorder
		r.routerLearningRuntime.replayRecorders = r.ReplayRecorders
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
	rt.mu.Lock()
	defer rt.mu.Unlock()
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
	exp := rt.experience[key]
	if exp == nil {
		exp = &routerLearningModelExperience{
			QualitySeed: 0.5,
			SeedWeight:  2,
		}
		rt.experience[key] = exp
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
	rt.mu.Lock()
	defer rt.mu.Unlock()
	for _, key := range []string{
		modelExperienceKey(decisionName, decisionTier, model),
		modelExperienceKey("", decisionTier, model),
		modelExperienceKey("", 0, model),
	} {
		if exp := rt.experience[key]; exp != nil {
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
