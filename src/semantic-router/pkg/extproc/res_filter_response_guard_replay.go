package extproc

import (
	"fmt"
	"slices"
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
	run := ctx.ResponseGuard
	cfg := classifierConfig(r.classifierForRequest(ctx))
	for _, outcome := range run.replayOutcomes(ctx, cfg, time.Now().UTC()) {
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

func (run *responseGuardRun) replayOutcomes(ctx *RequestContext, cfg *config.RouterConfig, now time.Time) []routerreplay.Outcome {
	var outcomes []routerreplay.Outcome
	signalOutcome := func(signalType, rule string) routerreplay.Outcome {
		outcome := run.replayOutcome(ctx, signalKey(signalType, rule), now)
		outcome.Verdict = "not_detected"
		outcome.Metadata["signal"] = signalType
		outcome.Metadata["direction"] = config.SignalDirectionResponse
		if action := run.signalAction(signalType); action != "" && !run.streamed {
			outcome.Metadata["action"] = action
		}
		return outcome
	}
	if jailbreak := run.jailbreak(); jailbreak != nil && cfg != nil {
		for _, rule := range cfg.ResponseJailbreakRules() {
			outcome := signalOutcome(config.SignalTypeJailbreak, rule.Name)
			outcome.Metadata["threshold"] = strconv.FormatFloat(float64(rule.Threshold), 'f', -1, 32)
			jailbreakReplayVerdict(&outcome, jailbreak, rule.Name)
			outcomes = append(outcomes, outcome)
		}
	}
	if hallucination := run.hallucination(); hallucination != nil && cfg != nil {
		for _, rule := range cfg.HallucinationRules {
			outcome := signalOutcome(config.SignalTypeHallucination, rule.Name)
			hallucinationReplayVerdict(&outcome, hallucination, rule.Name)
			outcomes = append(outcomes, outcome)
		}
	}
	if systemOne := run.systemOne(); systemOne != nil && cfg != nil {
		for _, rule := range cfg.SystemOneRules {
			key := signalKey(config.SignalTypeSystemOne, rule.Name)
			_, answered := systemOne.Confidences[key]
			code, failed := systemOne.Errors[key]
			if !answered && !failed {
				continue
			}
			outcome := signalOutcome(config.SignalTypeSystemOne, rule.Name)
			switch {
			case failed:
				outcome.Verdict, outcome.Reason = "unavailable", code
			case slices.Contains(systemOne.MatchedRules, rule.Name):
				outcome.Verdict = "detected"
			}
			outcome.Score = systemOne.Confidences[key]
			outcome.Metadata["score_available"] = strconv.FormatBool(answered)
			outcomes = append(outcomes, outcome)
		}
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

// signalCodes are the warning codes of a rule that reads a signal type.
var signalCodes = map[string][]string{
	config.SignalTypeJailbreak:     {headers.ResponseWarningJailbreak},
	config.SignalTypeHallucination: {headers.ResponseWarningHallucination, headers.ResponseWarningUnverifiedFactual},
	config.SignalTypeSystemOne:     {headers.ResponseWarningGuard},
}

// signalAction is the action of the first guard rule that read a signal of
// signalType and applied one.
func (run *responseGuardRun) signalAction(signalType string) string {
	for _, rule := range run.rules {
		if rule.action != "" && slices.ContainsFunc(rule.codes, func(code string) bool { return slices.Contains(signalCodes[signalType], code) }) {
			return rule.action
		}
	}
	return ""
}

func jailbreakReplayVerdict(outcome *routerreplay.Outcome, observation *classification.ResponseJailbreakObservation, rule string) {
	key := outcome.Target
	if code, failed := observation.Signal.Errors[key]; failed {
		outcome.Verdict, outcome.Reason = "unavailable", code
		outcome.Metadata["score_available"] = "false"
		return
	}
	if value, available := observation.Signal.Confidences[key]; available {
		outcome.Score = value
		outcome.Metadata["score_available"] = "true"
	} else {
		outcome.Metadata["score_available"] = "false"
	}
	scan := observation.Scan
	if scan != nil && scan.Decision != nil {
		outcome.Metadata["source_label"] = scan.Decision.SourceLabel
		outcome.Metadata["label"] = scan.Decision.Label
	}
	if slices.Contains(observation.Signal.MatchedRules, rule) {
		outcome.Verdict = "detected"
		if scan != nil && scan.Type != "" {
			outcome.Metadata["type"] = scan.Type
		}
	}
}

func hallucinationReplayVerdict(outcome *routerreplay.Outcome, observation *classification.ResponseHallucinationObservation, rule string) {
	if code, failed := observation.Signal.Errors[outcome.Target]; failed {
		outcome.Verdict, outcome.Reason = "unavailable", code
		return
	}
	if evidence := observation.Evidence; evidence != nil {
		outcome.Metadata["spans"] = strconv.Itoa(len(evidence.Spans))
		outcome.Metadata["score_available"] = strconv.FormatBool(evidence.Score.Available)
		if evidence.Score.Available {
			outcome.Score = float64(evidence.Confidence)
			outcome.Metadata["score_kind"] = evidence.Score.Kind
		}
	}
	if slices.Contains(observation.Signal.MatchedRules, rule) {
		outcome.Verdict = "detected"
	}
}

// updateRouterReplayHallucinationStatus writes the hallucination detector's
// verdict on the answer into the replay record, when the guard read it.
func (r *OpenAIRouter) updateRouterReplayHallucinationStatus(ctx *RequestContext, recorder *routerreplay.Recorder) {
	observation := ctx.ResponseGuard.hallucination()
	if observation == nil || observation.Evidence == nil {
		return
	}
	evidence := observation.Evidence
	spans := evidence.Spans
	details := hallucinationSpanDetailsForReplay(evidence.Explained)
	if !replayResponseContentAllowed(ctx) {
		// Detector excerpts and explanations are response content too.
		spans, details = nil, nil
	}
	err := recorder.UpdateHallucinationStatus(
		ctx.RouterReplayID,
		len(observation.Signal.MatchedRules) > 0,
		evidence.Confidence,
		spans,
		details,
		routerreplay.HallucinationScore{Available: evidence.Score.Available, Kind: evidence.Score.Kind},
	)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_hallucination_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}

// hallucinationSpanDetailsForReplay converts span details into the replay
// store's shape. Returns nil when the detector returned plain spans only.
func hallucinationSpanDetailsForReplay(spans []classification.EnhancedHallucinationSpan) []routerreplay.HallucinationSpan {
	if len(spans) == 0 {
		return nil
	}
	details := make([]routerreplay.HallucinationSpan, len(spans))
	for i, span := range spans {
		details[i] = routerreplay.HallucinationSpan{
			Text:                    span.Text,
			Start:                   span.Start,
			End:                     span.End,
			HallucinationConfidence: span.HallucinationConfidence,
			ScoreAvailable:          span.ScoreAvailable,
			Severity:                span.Severity,
			Explanation:             span.Explanation,
		}
	}
	return details
}
