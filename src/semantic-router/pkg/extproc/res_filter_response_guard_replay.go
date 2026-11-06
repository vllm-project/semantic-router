package extproc

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// responseGuardStreamingNotEnforced is the enforcement a streamed answer
// gets: none. It exists as a whole for the first time when its bytes are
// already with the client, so nothing the guard decided was applied.
const responseGuardStreamingNotEnforced = "not_enforced_streaming"

// recordRouterReplayResponseGuard appends what the selected decision's
// response_guard read and did to the replay record. The record is written
// while the request is routed, before the model has answered, so this reaches
// it as outcomes, the append-only post-route channel every store implements:
// one per response-stage rule the guard read, under the signal key a
// request-stage rule uses (detected, not_detected or unavailable, with the
// score or the failure code), and one per guard rule it read (matched,
// not_matched or unknown, with the action the rule applied).
func (r *OpenAIRouter) recordRouterReplayResponseGuard(ctx *RequestContext) {
	if ctx == nil || ctx.ResponseGuard == nil || ctx.RouterReplayID == "" {
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}
	for _, outcome := range ctx.ResponseGuard.replayOutcomes(ctx, time.Now().UTC()) {
		if err := recorder.AppendOutcome(ctx.RouterReplayID, outcome); err != nil {
			logging.ComponentErrorEvent("extproc", "router_replay_response_guard_outcome_failed", map[string]interface{}{
				"request_id": ctx.RequestID,
				"replay_id":  ctx.RouterReplayID,
				"target":     outcome.Target,
				"error":      err.Error(),
			})
		}
	}
	r.updateRouterReplayHallucinationStatus(ctx, recorder)
}

func (run *responseGuardRun) replayOutcomes(ctx *RequestContext, now time.Time) []routerreplay.Outcome {
	var outcomes []routerreplay.Outcome
	observed := run.observed()
	for _, key := range observedRuleKeys(observed) {
		signalType, rule, _ := strings.Cut(key, ":")
		outcome := run.replayOutcome(ctx, key, now)
		outcome.Verdict = "not_detected"
		outcome.Metadata["signal"] = signalType
		outcome.Metadata["stage"] = config.SystemOneStageResponse
		if action := run.signalAction(signalType); action != "" && !run.streamed {
			outcome.Metadata["action"] = action
		}
		code, failed := observed.Errors[key]
		score, answered := observed.Confidences[key]
		switch {
		case failed:
			outcome.Verdict, outcome.Reason = "unavailable", code
		case slices.Contains(observed.Matched[signalType], rule):
			outcome.Verdict = "detected"
		}
		outcome.Score = score
		outcome.Metadata["score_available"] = strconv.FormatBool(answered && !failed)
		if spans, ok := observed.Spans[key]; ok {
			outcome.Metadata["spans"] = strconv.Itoa(len(spans))
		}
		outcomes = append(outcomes, outcome)
	}
	for i, rule := range run.rules {
		outcome := run.replayOutcome(ctx, config.DecisionPluginResponseGuard, now)
		outcome.TargetRef = fmt.Sprintf("rules[%d]", i)
		outcome.Verdict = ruleOutcomeName(rule.state)
		outcome.Metadata["warnings"] = strings.Join(rule.codes, ",")
		if rule.action != "" && !run.streamed {
			outcome.Metadata["action"] = rule.action
		}
		outcomes = append(outcomes, outcome)
	}
	return outcomes
}

// observedRuleKeys are the "<type>:<rule>" keys of the response-stage rules
// that answered or failed, in key order; an option's key ("<type>:<rule>:
// <label>") is not a rule's.
func observedRuleKeys(observed *classification.ResponseSignal) []string {
	if observed == nil {
		return nil
	}
	var keys []string
	for key := range observed.Confidences {
		if strings.Count(key, ":") == 1 {
			keys = append(keys, key)
		}
	}
	for key := range observed.Errors {
		if strings.Count(key, ":") == 1 && !slices.Contains(keys, key) {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	return keys
}

func (run *responseGuardRun) replayOutcome(ctx *RequestContext, target string, now time.Time) routerreplay.Outcome {
	outcome := routerreplay.Outcome{Timestamp: now, Source: "router", Target: target, Metadata: map[string]string{}}
	if ctx.VSRSelectedDecisionName != "" {
		outcome.Metadata["decision"] = ctx.VSRSelectedDecisionName
	}
	if run.streamed {
		outcome.Metadata["enforcement"] = responseGuardStreamingNotEnforced
	}
	return outcome
}

// signalCodes are the warning codes of a rule that reads a signal type; a
// rule that reads any other response-stage type writes the guard's code.
var signalCodes = map[string][]string{
	config.SignalTypeJailbreak:     {headers.ResponseWarningJailbreak},
	config.SignalTypeHallucination: {headers.ResponseWarningHallucination, headers.ResponseWarningUnverifiedFactual},
}

// signalAction is the action of the first guard rule that read a signal of
// signalType and applied one.
func (run *responseGuardRun) signalAction(signalType string) string {
	codes, ok := signalCodes[signalType]
	if !ok {
		codes = []string{headers.ResponseWarningGuard}
	}
	for _, rule := range run.rules {
		if rule.action != "" && slices.ContainsFunc(rule.codes, func(code string) bool { return slices.Contains(codes, code) }) {
			return rule.action
		}
	}
	return ""
}

// hasSignalError reports whether a response-stage rule of signalType failed.
func hasSignalError(observed *classification.ResponseSignal, signalType string) bool {
	for key := range observed.Errors {
		if strings.HasPrefix(key, signalType+":") {
			return true
		}
	}
	return false
}

// updateRouterReplayHallucinationStatus writes the hallucination rules'
// verdict on the answer into the replay record, when the guard read them.
func (r *OpenAIRouter) updateRouterReplayHallucinationStatus(ctx *RequestContext, recorder *routerreplay.Recorder) {
	finding := findHallucination(ctx.ResponseGuard.observed())
	if !finding.asked {
		return
	}
	spans := finding.spans
	if !replayResponseContentAllowed(ctx) {
		// Span excerpts are response content too.
		spans = nil
	}
	err := recorder.UpdateHallucinationStatus(
		ctx.RouterReplayID,
		finding.detected,
		float32(finding.probability),
		spans,
		nil,
		routerreplay.HallucinationScore{Available: finding.answered, Kind: "probability"},
	)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_hallucination_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}
