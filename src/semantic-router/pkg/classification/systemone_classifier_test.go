package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// systemOneHandler answers the one Choice question the request asked, after
// asserting the request really is the contract this backend promises to send.
// wantOptions is the declared label count, which the request always carries in
// full even when the handler is set up to answer with an incomplete
// distribution.
func systemOneHandler(t *testing.T, wantState string, wantOptions int, probabilities map[string]float32) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/systemone" {
			t.Errorf("request path = %q, want /v1/systemone", r.URL.Path)
		}
		var decoded systemOneRequest
		if err := json.NewDecoder(r.Body).Decode(&decoded); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if decoded.State != wantState {
			t.Errorf("state = %q, want %q", decoded.State, wantState)
		}
		if decoded.Model != "decision-kai" {
			t.Errorf("model = %q, want the configured llm_model_name", decoded.Model)
		}
		if len(decoded.Questions) != 1 {
			t.Errorf("questions = %d, want 1", len(decoded.Questions))
		}
		question, ok := decoded.Questions["tone"]
		if !ok {
			t.Errorf("questions has no entry for the signal name, got %v", decoded.Questions)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if question.Type != "choice" {
			t.Errorf("question type = %q, want choice", question.Type)
		}
		if question.Instructions != "Which tone does this request use?" {
			t.Errorf("instructions = %q, want the rule's instructions", question.Instructions)
		}
		if len(question.Criteria) != wantOptions {
			t.Errorf("criteria = %d options, want %d", len(question.Criteria), wantOptions)
		}
		for option, description := range question.Criteria {
			if description != nil {
				t.Errorf("criteria[%q] = %q, want an explicit null", option, *description)
			}
		}
		choice, confidence := systemOneChoiceFor(probabilities)
		systemOneReply("decision-kai", map[string]any{
			"type":          "choice",
			"choice":        choice,
			"confidence":    confidence,
			"probabilities": probabilities,
		})(w, r)
	}
}

// systemOneChoiceFor returns the choice and confidence the SystemOne contract
// requires for a distribution: a most probable option and the top-two margin.
func systemOneChoiceFor(probabilities map[string]float32) (string, float32) {
	var choice string
	var first, second float32
	for label, probability := range probabilities {
		switch {
		case probability > first:
			choice, first, second = label, probability, first
		case probability > second:
			second = probability
		}
	}
	return choice, first - second
}

// systemOneReply answers every request with one fixed answer to the tone
// question, which lets a test send what the contract forbids.
func systemOneReply(model string, answer map[string]any) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   model,
			"usage":   map[string]int{"input_tokens": 7, "output_tokens": 0},
			"answers": map[string]any{"tone": answer},
		})
	}
}

func systemOneRule() config.ClassifierSignalRule {
	return config.ClassifierSignalRule{
		Name:         "tone",
		Type:         config.ClassifierSignalTypeSystemOne,
		Model:        "decision-runtime",
		Labels:       []string{"formal", "casual", "hostile"},
		Instructions: "Which tone does this request use?",
	}
}

func systemOneExternal(t *testing.T, server *httptest.Server) *config.ExternalModelConfig {
	t.Helper()
	return &config.ExternalModelConfig{
		Name:          "decision-runtime",
		ModelEndpoint: endpointForTestServer(t, server),
		ModelName:     "decision-kai",
		ModelRole:     config.ModelRoleClassification,
	}
}

func newTestSystemOneBackend(t *testing.T, server *httptest.Server) *SystemOneClassifierInference {
	t.Helper()
	rule := systemOneRule()
	backend, err := NewSystemOneClassifierInference(
		systemOneExternal(t, server),
		newDeclaredLabelMapping(rule.Labels),
		rule.Name,
		rule.Instructions,
		0,
	)
	if err != nil {
		t.Fatalf("NewSystemOneClassifierInference: %v", err)
	}
	return backend
}

// The declared label order is the class-index contract every other backend
// produces, so an answer whose option order differs must still land on the
// same indices.
func TestSystemOneBackendMapsOptionsToDeclaredLabelOrder(t *testing.T) {
	server := httptest.NewServer(systemOneHandler(t, "please advise", 3, map[string]float32{
		"hostile": 0.1,
		"formal":  0.7,
		"casual":  0.2,
	}))
	defer server.Close()

	backend := newTestSystemOneBackend(t, server)
	defer backend.Close()

	result, err := backend.Classify(context.Background(), "please advise")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	assertProbabilities(t, result.Probabilities, []float32{0.7, 0.2, 0.1})
}

// A distribution missing one declared option would otherwise default that entry
// to zero and under-report the label, so it has to fail instead.
func TestSystemOneBackendRejectsIncompleteDistribution(t *testing.T) {
	server := httptest.NewServer(systemOneHandler(t, "please advise", 3, map[string]float32{
		"formal": 0.8,
		"casual": 0.2,
	}))
	defer server.Close()

	backend := newTestSystemOneBackend(t, server)
	defer backend.Close()

	_, err := backend.Classify(context.Background(), "please advise")
	if err == nil {
		t.Fatal("Classify accepted a distribution missing a declared label")
	}
	if !strings.Contains(err.Error(), "hostile") {
		t.Fatalf("error does not name the missing label: %v", err)
	}
}

// The contract ties the response to the requested model and the choice to the
// answer's own distribution. A response that breaks one of them, such as a
// hostile choice over formal=0.7 and hostile=0.3, would route on scores its own
// answer disagrees with, so it has to fail, and so does a confidence that is
// not a probability. The accepted responses first show each rejection comes
// from the one field changed, and that a provider's own confidence, here not
// the top-two margin #4086 reports, is not held against it.
func TestSystemOneBackendRejectsInconsistentResponses(t *testing.T) {
	distribution := map[string]float64{"formal": 0.7, "casual": 0, "hostile": 0.3}
	classify := func(model, choice string, confidence float64) error {
		server := httptest.NewServer(systemOneReply(model, map[string]any{
			"type":          "choice",
			"choice":        choice,
			"confidence":    confidence,
			"probabilities": distribution,
		}))
		defer server.Close()
		backend := newTestSystemOneBackend(t, server)
		defer backend.Close()
		_, err := backend.Classify(context.Background(), "please advise")
		return err
	}
	for _, confidence := range []float64{0.4, 0.7} {
		if err := classify("decision-kai", "formal", confidence); err != nil {
			t.Fatalf("Classify rejected a consistent response with confidence %v: %v", confidence, err)
		}
	}
	for _, tc := range []struct {
		name, model, choice string
		confidence          float64
		want                string
	}{
		{"wrong model", "decision-other", "formal", 0.4, `model "decision-other"`},
		{"choice against the distribution", "decision-kai", "hostile", 0.4, `chose "hostile"`},
		{"confidence outside [0, 1]", "decision-kai", "formal", 1.4, "confidence 1.4"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := classify(tc.model, tc.choice, tc.confidence)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Classify error = %v, want one naming %s", err, tc.want)
			}
		})
	}
}

// An absolute address is the API base the operator configured, so the request
// has to reach its path plus /systemone, the target remoteOperationIdentity
// records, rather than a URL that wraps the address in scheme and port again.
func TestSystemOneBackendSendsToAbsoluteBaseURL(t *testing.T) {
	targets := make(chan string, 1)
	reply := systemOneReply("decision-kai", map[string]any{
		"type":          "choice",
		"choice":        "formal",
		"confidence":    0.5,
		"probabilities": map[string]float64{"formal": 0.7, "casual": 0.2, "hostile": 0.1},
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case targets <- "http://" + r.Host + r.URL.Path:
		default:
		}
		reply(w, r)
	}))
	defer server.Close()

	external := systemOneExternal(t, server)
	external.ModelEndpoint.Address = server.URL + "/v1"
	rule := systemOneRule()
	backend, err := NewSystemOneClassifierInference(
		external, newDeclaredLabelMapping(rule.Labels), rule.Name, rule.Instructions, 0,
	)
	if err != nil {
		t.Fatalf("NewSystemOneClassifierInference: %v", err)
	}
	defer backend.Close()
	if _, err = backend.Classify(context.Background(), "please advise"); err != nil {
		t.Fatalf("Classify: %v", err)
	}
	identity, _, err := remoteOperationIdentity(config.RemoteClassifierProtocolHTTPSystemOne, external)
	if err != nil {
		t.Fatalf("remoteOperationIdentity: %v", err)
	}
	if target := <-targets; target != server.URL+"/v1/systemone" || target != identity {
		t.Fatalf("request went to %q, want %q, the recorded identity %q", target, server.URL+"/v1/systemone", identity)
	}
}

// An endpoint that answers something other than the question asked is a
// contract violation, not a result to be attributed to this signal.
func TestSystemOneBackendRejectsUnaskedAnswer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "decision-kai",
			"usage": map[string]int{"input_tokens": 7, "output_tokens": 0},
			"answers": map[string]any{
				"sentiment": map[string]any{
					"type":          "choice",
					"choice":        "formal",
					"confidence":    0.7,
					"probabilities": map[string]float32{"formal": 0.7, "casual": 0.2, "hostile": 0.1},
				},
			},
		})
	}))
	defer server.Close()

	backend := newTestSystemOneBackend(t, server)
	defer backend.Close()

	_, err := backend.Classify(context.Background(), "please advise")
	if err == nil {
		t.Fatal("Classify accepted an answer to a question it never asked")
	}
	if !strings.Contains(err.Error(), "tone") {
		t.Fatalf("error does not name the asked question: %v", err)
	}
}

// A Noul or Score answer carries no option distribution, so accepting one would
// hand a consumer an empty result rather than an error.
func TestSystemOneBackendRejectsNonChoiceAnswer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "decision-kai",
			"usage":   map[string]int{"input_tokens": 7, "output_tokens": 0},
			"answers": map[string]any{"tone": map[string]any{"type": "noul", "noul": 0.9}},
		})
	}))
	defer server.Close()

	backend := newTestSystemOneBackend(t, server)
	defer backend.Close()

	_, err := backend.Classify(context.Background(), "please advise")
	if err == nil {
		t.Fatal("Classify accepted a Noul answer for a Choice question")
	}
	if !strings.Contains(err.Error(), "choice") {
		t.Fatalf("error does not name the expected answer type: %v", err)
	}
}

// The request names the model explicitly, so a catalog entry without one cannot
// be served and must fail at construction rather than on the first request.
func TestSystemOneBackendRequiresExplicitModelName(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()

	external := systemOneExternal(t, server)
	external.ModelName = ""
	rule := systemOneRule()
	if _, err := NewSystemOneClassifierInference(
		external, newDeclaredLabelMapping(rule.Labels), rule.Name, rule.Instructions, 0,
	); err == nil {
		t.Fatal("NewSystemOneClassifierInference accepted an external model with no llm_model_name")
	}
}

// The Choice contract rejects a null question, so an empty instructions string
// cannot be sent.
func TestSystemOneBackendRequiresInstructions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()

	rule := systemOneRule()
	if _, err := NewSystemOneClassifierInference(
		systemOneExternal(t, server), newDeclaredLabelMapping(rule.Labels), rule.Name, "   ", 0,
	); err == nil {
		t.Fatal("NewSystemOneClassifierInference accepted blank instructions")
	}
}

// A mapping larger than the Choice arity cannot be served at all, so it fails
// at construction instead of producing a request the endpoint will reject.
func TestSystemOneBackendRejectsMoreLabelsThanChoiceAccepts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()

	labels := make([]string, systemOneMaxOptions+1)
	for i := range labels {
		labels[i] = "label-" + string(rune('a'+i%26)) + string(rune('a'+i/26))
	}
	if _, err := NewSystemOneClassifierInference(
		systemOneExternal(t, server), newDeclaredLabelMapping(labels), "tone", "Which?", 0,
	); err == nil {
		t.Fatalf("NewSystemOneClassifierInference accepted %d labels", len(labels))
	}
}

// The rule type has to reach the builder, not only the validator, or a
// configured signal silently disappears.
func TestSystemOneRuleBuildsThroughGenericClassifierBuilder(t *testing.T) {
	server := httptest.NewServer(systemOneHandler(t, "please advise", 3, map[string]float32{
		"formal": 0.7, "casual": 0.2, "hostile": 0.1,
	}))
	defer server.Close()

	rule := systemOneRule()
	classifier, err := newSystemOneLabelClassifier(rule, systemOneExternal(t, server))
	if err != nil {
		t.Fatalf("newSystemOneLabelClassifier: %v", err)
	}
	if closer, ok := classifier.(interface{ Close() error }); ok {
		defer closer.Close()
	}
	got, err := classifier.Classify(context.Background(), "please advise")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	for label, want := range map[string]float64{"formal": 0.7, "casual": 0.2, "hostile": 0.1} {
		if diff := got.Scores[label] - want; diff > 1e-6 || diff < -1e-6 {
			t.Fatalf("score[%q] = %v, want %v", label, got.Scores[label], want)
		}
	}
}
