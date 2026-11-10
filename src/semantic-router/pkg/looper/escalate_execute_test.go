package looper

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func gte(value float64) *float64 { return &value }

// An answer whose confidence cannot be read moves on to the next model, under
// either on_error, and every paid attempt stays in the usage.
func TestEscalateLooperSkipsMissingLogprobsAndEscalates(t *testing.T) {
	for _, onError := range []string{config.EscalateOnErrorLastAttempt, config.EscalateOnErrorFail} {
		t.Run(onError, func(t *testing.T) {
			server, calls := newEscalateLogprobBackend(t, map[string]int{"small": 0, "large": 1}, true)
			defer server.Close()

			response, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
				context.Background(),
				escalateRequest(onError, config.ConfidenceMethodAverageLogprob),
			)
			if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}
			if response.Model != "large" || response.AlgorithmType != config.DecisionAlgorithmEscalate {
				t.Fatalf("response = model %q algorithm %q, want large/escalate", response.Model, response.AlgorithmType)
			}
			if !slices.Equal(*calls, []string{"small", "large"}) {
				t.Fatalf("backend calls = %v, want [small large]", *calls)
			}
			if response.Usage.TotalTokens != 6 {
				t.Fatalf("response usage = %+v, want both paid attempts", response.Usage)
			}
			var completion struct {
				Usage TokenUsage `json:"usage"`
			}
			if err := json.Unmarshal(response.Body, &completion); err != nil {
				t.Fatalf("decode response body: %v", err)
			}
			if completion.Usage.TotalTokens != 6 {
				t.Fatalf("body usage = %+v, want both paid attempts", completion.Usage)
			}
		})
	}
}

// When no answer is accepted, last_attempt serves the last scored answer and
// fail fails the request with the paid evidence and no candidate content.
func TestEscalateLooperOnErrorWhenNoAnswerIsAccepted(t *testing.T) {
	for _, onError := range []string{config.EscalateOnErrorLastAttempt, config.EscalateOnErrorFail} {
		t.Run(onError, func(t *testing.T) {
			server, calls := newEscalateLogprobBackend(t, map[string]int{"small": 1, "large": 1}, true)
			defer server.Close()
			request := escalateRequest(onError, config.ConfidenceMethodAverageLogprob)
			request.Algorithm.Escalate.Accept.Confidence.GTE = gte(0.99)

			response, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request)
			if !slices.Equal(*calls, []string{"small", "large"}) {
				t.Fatalf("backend calls = %v, want [small large]", *calls)
			}
			if onError == config.EscalateOnErrorLastAttempt {
				if err != nil || response.Model != "large" || response.Usage.TotalTokens != 6 {
					t.Fatalf("last_attempt = %+v, %v, want the large answer with both attempts", response, err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "no model's answer was accepted") {
				t.Fatalf("Execute() error = %v, want an unaccepted escalation", err)
			}
			evidence, ok := EscalateEvidenceFromError(err)
			if !ok || evidence.Iterations != 2 || evidence.Usage.TotalTokens != 6 || !slices.Equal(evidence.ModelsUsed, []string{"small", "large"}) {
				t.Fatalf("partial evidence = %+v (present=%v), want both paid attempts", evidence, ok)
			}
			if strings.Contains(err.Error(), "large answer") {
				t.Fatalf("partial execution error leaked candidate content: %v", err)
			}
		})
	}
}

func TestEscalateLooperFailsWhenEveryCandidateOmitsLogprobs(t *testing.T) {
	server, calls := newEscalateLogprobBackend(t, map[string]int{"small": 0, "large": 0}, true)
	defer server.Close()

	_, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		escalateRequest(config.EscalateOnErrorLastAttempt, config.ConfidenceMethodAverageLogprob),
	)
	if err == nil || !strings.Contains(err.Error(), "all models failed") || !strings.Contains(err.Error(), "token logprobs") {
		t.Fatalf("Execute() error = %v, want terminal missing-logprobs failure", err)
	}
	if !slices.Equal(*calls, []string{"small", "large"}) {
		t.Fatalf("backend calls = %v, want [small large]", *calls)
	}
	evidence, ok := EscalateEvidenceFromError(err)
	if !ok || evidence.Iterations != 2 || evidence.Usage.TotalTokens != 6 || !slices.Equal(evidence.ModelsUsed, []string{"small", "large"}) {
		t.Fatalf("partial evidence = %+v (present=%v), want both paid attempts", evidence, ok)
	}
}

// Missing top-logprob alternatives leave a margin answer unscored, so the
// escalation moves on; the large model's margin (0.49) stays below 0.72.
func TestEscalateLooperTreatsMissingMarginAlternativesAsUnscored(t *testing.T) {
	server, calls := newEscalateLogprobBackend(t, map[string]int{"small": 1, "large": 2}, true)
	defer server.Close()

	response, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		escalateRequest(config.EscalateOnErrorLastAttempt, config.ConfidenceMethodMargin),
	)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if response.Model != "large" || response.Usage.TotalTokens != 6 {
		t.Fatalf("response = model %q usage %+v, want escalated aggregate", response.Model, response.Usage)
	}
	if !slices.Equal(*calls, []string{"small", "large"}) {
		t.Fatalf("backend calls = %v, want [small large]", *calls)
	}
}

// stubAnswerChecker answers the answer_check and refusal questions from
// per-answer tables, and counts the checks it was asked.
type stubAnswerChecker struct {
	correct map[string]float64
	refused map[string]float64
	err     error
	asked   []string
}

func (s *stubAnswerChecker) CheckAnswer(_ context.Context, request, answer string) (float64, error) {
	s.asked = append(s.asked, "answer_check:"+request+"|"+answer)
	return s.correct[answer], s.err
}

func (s *stubAnswerChecker) Refusal(_ context.Context, request, answer string) (float64, float64, error) {
	s.asked = append(s.asked, "refusal:"+request+"|"+answer)
	return s.refused[answer], 0.5, s.err
}

// accept.systemone asks the answer_check question about the request and each
// answer, without logprobs, and accepts the first at or above gte.
func TestEscalateLooperAcceptsByAnswerCheck(t *testing.T) {
	server, calls := newEscalateLogprobBackend(t, map[string]int{}, false)
	defer server.Close()
	checker := &stubAnswerChecker{correct: map[string]float64{"small answer": 0.4, "large answer": 0.9}}
	request := escalateRequest("", "")
	request.Algorithm.Escalate.Accept = &config.EscalateAccept{SystemOne: &config.EscalateAnswerCheck{GTE: gte(0.7)}}
	request.AnswerChecks = checker

	response, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request)
	if err != nil || response.Model != "large" {
		t.Fatalf("Execute() = %+v, %v, want the large answer", response, err)
	}
	if !slices.Equal(*calls, []string{"small", "large"}) || !slices.Equal(checker.asked, []string{
		"answer_check:answer with calibrated confidence|small answer",
		"answer_check:answer with calibrated confidence|large answer",
	}) {
		t.Fatalf("calls = %v, checks = %v", *calls, checker.asked)
	}
}

// refusal: false accepts the first answer the refusal preset does not flag.
func TestEscalateLooperAcceptsByRefusal(t *testing.T) {
	server, calls := newEscalateLogprobBackend(t, map[string]int{}, false)
	defer server.Close()
	refusal := false
	checker := &stubAnswerChecker{refused: map[string]float64{"small answer": 0.8, "large answer": 0.1}}
	request := escalateRequest("", "")
	request.Algorithm.Escalate.Accept = &config.EscalateAccept{Refusal: &refusal}
	request.AnswerChecks = checker

	response, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request)
	if err != nil || response.Model != "large" || !slices.Equal(*calls, []string{"small", "large"}) {
		t.Fatalf("Execute() = %+v, %v (calls %v), want the large answer", response, err, *calls)
	}
}

// A check the decision model cannot answer leaves every answer unscored.
func TestEscalateLooperFailsWhenTheAnswerCheckFails(t *testing.T) {
	server, _ := newEscalateLogprobBackend(t, map[string]int{}, false)
	defer server.Close()
	request := escalateRequest("", "")
	request.Algorithm.Escalate.Accept = &config.EscalateAccept{SystemOne: &config.EscalateAnswerCheck{GTE: gte(0.7)}}
	request.AnswerChecks = &stubAnswerChecker{err: errors.New("decision model unavailable")}

	_, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request)
	if err == nil || !strings.Contains(err.Error(), "all models failed") || !strings.Contains(err.Error(), "decision model unavailable") {
		t.Fatalf("Execute() error = %v, want every check failed", err)
	}
}

func TestEscalateLooperRequiresAnAnswerCheckerForSystemOneChecks(t *testing.T) {
	request := escalateRequest("", "")
	request.Algorithm.Escalate.Accept = &config.EscalateAccept{SystemOne: &config.EscalateAnswerCheck{GTE: gte(0.7)}}
	if _, err := NewEscalateLooper(&config.LooperConfig{}).Execute(context.Background(), request); err == nil ||
		!strings.Contains(err.Error(), "answer checker") {
		t.Fatalf("Execute() error = %v, want a missing answer checker", err)
	}
}

// escalateRequest is a two-model escalation accepted by confidence method
// at 0.72; an empty method leaves accept for the test to set.
func escalateRequest(onError, method string) *Request {
	params := openai.ChatCompletionNewParams{
		Model: "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("answer with calibrated confidence"),
		},
	}
	escalate := &config.EscalateAlgorithmConfig{Order: config.EscalateOrderSize}
	if method != "" {
		escalate.Accept = &config.EscalateAccept{Confidence: &config.EscalateConfidence{Method: method, GTE: gte(0.72)}}
	}
	return &Request{
		OriginalRequest: &params,
		ModelRefs: []config.ModelRef{
			{Model: "small"},
			{Model: "large"},
		},
		ModelParams: map[string]config.ModelParams{
			"small": {ParamSize: "1b"},
			"large": {ParamSize: "10b"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type:     config.DecisionAlgorithmEscalate,
			OnError:  onError,
			Escalate: escalate,
		},
		DecisionName: "escalate_test",
	}
}

// newEscalateLogprobBackend answers "<model> answer" with as many top
// logprobs as topLogprobsByModel gives the model, none when it has none.
func newEscalateLogprobBackend(t *testing.T, topLogprobsByModel map[string]int, wantLogprobs bool) (*httptest.Server, *[]string) {
	t.Helper()
	calls := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var requestBody struct {
			Model    string `json:"model"`
			Logprobs bool   `json:"logprobs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		calls = append(calls, requestBody.Model)
		if requestBody.Logprobs != wantLogprobs {
			t.Errorf("model %q request logprobs = %v, want %v", requestBody.Model, requestBody.Logprobs, wantLogprobs)
		}

		choice := map[string]interface{}{
			"index":         0,
			"message":       map[string]interface{}{"role": "assistant", "content": requestBody.Model + " answer"},
			"finish_reason": "stop",
		}
		if topLogprobsByModel[requestBody.Model] > 0 {
			topLogprobs := []map[string]interface{}{
				{"token": "answer", "logprob": -0.1, "bytes": []int{97}},
			}
			if topLogprobsByModel[requestBody.Model] > 1 {
				topLogprobs = append(topLogprobs, map[string]interface{}{
					"token": "alternative", "logprob": -2.1, "bytes": []int{98},
				})
			}
			choice["logprobs"] = map[string]interface{}{
				"content": []map[string]interface{}{
					{
						"token":        "answer",
						"logprob":      -0.1,
						"bytes":        []int{97},
						"top_logprobs": topLogprobs,
					},
				},
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-escalate-test",
			"object":  "chat.completion",
			"created": 0,
			"model":   requestBody.Model,
			"choices": []map[string]interface{}{choice},
			"usage": map[string]int{
				"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
			},
		})
	}))
	return server, &calls
}
