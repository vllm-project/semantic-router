package services

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestIntentPolicyMatchHasNoModelProbability(t *testing.T) {
	service := &ClassificationService{}
	signals := &classification.SignalResults{SignalErrorMatches: map[string]bool{"jailbreak:guard": true}}
	result := &decision.DecisionResult{Decision: &config.Decision{Name: "guard"}, Confidence: 1, ConfidenceScored: false}
	response := service.buildIntentResponseFromSignals(signals, result, "guard", 1, 0, IntentRequest{Options: &IntentOptions{ReturnProbabilities: true}}, nil, nil)
	raw, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]interface{}
	if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if decoded["classification"].(map[string]interface{})["confidence"] != nil || decoded["decision_result"].(map[string]interface{})["confidence"] != nil {
		t.Fatalf("policy match became probability: %s", raw)
	}
	if decoded["probabilities_available"] != false || decoded["probabilities"] != nil {
		t.Fatalf("policy match exposed probabilities: %s", raw)
	}
	if decoded["signal_error_matches"].(map[string]interface{})["jailbreak:guard"] != true || response.RoutingDecision != "guard" {
		t.Fatalf("policy match was erased: %s", raw)
	}
}

func TestClassificationReportedZeroAndPlaceholder(t *testing.T) {
	zero, err := json.Marshal(Classification{ConfidenceAvailable: confidenceAvailability(true)})
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]interface{}
	if decodeErr := json.Unmarshal(zero, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if decoded["confidence"] != float64(0) || decoded["confidence_available"] != true {
		t.Fatalf("actual zero lost: %s", zero)
	}
	service := &ClassificationService{}
	response, err := service.ClassifyIntent(context.Background(), IntentRequest{Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if decoded["classification"].(map[string]interface{})["confidence"] != nil || response.RoutingDecision != "placeholder_response" || response.ProbabilitiesAvailable {
		t.Fatalf("placeholder invented confidence: %s", raw)
	}
}

func TestEvalUnavailableJailbreakMetricPreservesPolicyMatch(t *testing.T) {
	available := false
	response := EvalResponse{Metrics: &classification.SignalMetricsCollection{Jailbreak: classification.SignalMetrics{Confidence: 1, ConfidenceAvailable: &available}}, SignalErrorMatches: map[string]bool{"jailbreak:guard": true}}
	raw, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]interface{}
	if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	metric := decoded["metrics"].(map[string]interface{})["jailbreak"].(map[string]interface{})
	if metric["confidence"] != nil || metric["confidence_available"] != false || decoded["signal_error_matches"] == nil {
		t.Fatalf("unavailable metric became probability: %s", raw)
	}
}

func TestUnavailableFactAndFeedbackServicesDoNotReportZeroProbability(t *testing.T) {
	service := &ClassificationService{}
	fact, err := service.ClassifyFactCheck(context.Background(), FactCheckRequest{Text: "a claim"})
	if err != nil {
		t.Fatal(err)
	}
	feedback, err := service.ClassifyUserFeedback(context.Background(), UserFeedbackRequest{Text: "a reply"})
	if err != nil {
		t.Fatal(err)
	}
	for _, response := range []interface{}{fact, feedback} {
		raw, marshalErr := json.Marshal(response)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		var decoded map[string]interface{}
		if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if decoded["confidence"] != nil || decoded["confidence_available"] != false {
			t.Fatalf("unavailable classifier exposed model score: %s", raw)
		}
	}
}
