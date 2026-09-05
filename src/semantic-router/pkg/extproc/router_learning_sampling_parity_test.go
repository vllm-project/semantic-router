package extproc

import (
	"encoding/json"
	"os"
	"testing"
	"time"
)

// This adapter executes production helpers, not a second Go implementation.
// The Python contract test supplies frozen/randomized states and posterior draws.
type samplingParityArm struct {
	Model  string
	State  routerLearningModelExperience
	Cost   float64
	Sample float64
}

type samplingParityCase struct {
	Arms     []samplingParityArm
	Base     string
	Scope    string
	Sampling bool
}

type samplingParityEvent struct {
	Verdict   routerLearningOutcomeVerdict
	Weight    float64
	Telemetry *routerLearningTelemetryObservation
}

type samplingParityRequest struct {
	Cases  []samplingParityCase
	Events []samplingParityEvent
}

type samplingParityResult struct {
	Scores []map[string]interface{}
	Winner string
}

func samplingParityScores(input samplingParityCase) samplingParityResult {
	maxCost := 0.0
	for _, arm := range input.Arms {
		if arm.Cost > maxCost {
			maxCost = arm.Cost
		}
	}
	scores := make([]routerLearningCandidateScore, 0, len(input.Arms))
	results := make([]map[string]interface{}, 0, len(input.Arms))
	for _, arm := range input.Arms {
		alpha, beta := 0.0, 0.0
		var sample func(float64, float64) float64
		if input.Sampling {
			sample = func(a, b float64) float64 { alpha, beta = a, b; return arm.Sample }
		}
		score := scoreRoutingSamplingExperience(arm.Model, arm.State,
			routingSamplingCostPenalty(arm.Cost, maxCost, input.Scope), arm.Model == input.Base, sample)
		scores = append(scores, score)
		results = append(results, map[string]interface{}{
			"model": score.model, "score": score.score, "posterior_mean": score.posteriorMean,
			"predicted_quality": score.predictedQuality, "cost_penalty": score.costPenalty,
			"overuse_penalty": score.overusePenalty, "reliability_penalty": score.reliabilityPenalty,
			"latency_adjustment": score.latencyAdjustment, "cache_adjustment": score.cacheAdjustment,
			"cold_start": score.coldStart, "alpha": alpha, "beta": beta,
		})
	}
	sortRoutingSamplingScores(scores)
	winner := routingSamplingWinner(scores, input.Base, input.Scope, input.Sampling)
	return samplingParityResult{Scores: results, Winner: winner.model}
}

func TestRoutingSamplingReplayParity(t *testing.T) {
	inputPath := os.Getenv("VLLM_SR_SAMPLING_PARITY_INPUT")
	if inputPath == "" {
		t.Skip("run tests/test_router_learning_policy_parity.py for cross-language parity")
	}
	data, err := os.ReadFile(inputPath)
	if err != nil {
		t.Fatal(err)
	}
	var request samplingParityRequest
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatal(err)
	}
	results := make([]samplingParityResult, 0, len(request.Cases))
	for _, input := range request.Cases {
		results = append(results, samplingParityScores(input))
	}
	state := routerLearningModelExperience{QualitySeed: 0.5, SeedWeight: 2}
	states := make([]routerLearningModelExperience, 0, len(request.Events))
	for _, event := range request.Events {
		if event.Telemetry != nil {
			applyRouterLearningTelemetry(&state, *event.Telemetry)
		} else {
			applyRouterLearningOutcome(&state, event.Verdict, event.Weight)
		}
		state.LastUpdated = time.Unix(1, 0).UTC()
		states = append(states, state)
	}
	output, err := json.Marshal(struct {
		Cases  []samplingParityResult
		States []routerLearningModelExperience
	}{results, states})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv("VLLM_SR_SAMPLING_PARITY_OUTPUT"), output, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestRoutingSamplingColdStartOverridesMargin(t *testing.T) {
	scores := []routerLearningCandidateScore{
		{model: "base", score: 0.9}, {model: "cold", score: 0.1, coldStart: true},
	}
	if got := routingSamplingWinner(scores, "base", "global", true); got.model != "cold" {
		t.Fatalf("cold start must override the global/base margin: %s", got.model)
	}
	if got := routingSamplingWinner(scores, "base", "global", false); got.model != "base" {
		t.Fatalf("suppressed sampling must preserve the margin: %s", got.model)
	}
}
