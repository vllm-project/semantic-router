package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// fakeSystemOneAPI is a System One API that answers every choice question
// "a", about the request's own state and each of its states. It fails a call
// that asks failing as a whole on that question, or every call with status.
type fakeSystemOneAPI struct {
	mu      sync.Mutex
	calls   []systemOneCall
	failing string
	status  int
	listing string
	hold    chan struct{}
}

// systemOneCall is one call the API answered: the questions about each state
// by its text.
type systemOneCall map[string][]string

func (f *fakeSystemOneAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		if f.hold != nil {
			select {
			case <-f.hold:
			case <-r.Context().Done():
				return
			}
		}
		if f.listing == "" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(f.listing))
		return
	}
	var body struct {
		Model     string                     `json:"model"`
		State     string                     `json:"state"`
		Questions map[string]json.RawMessage `json:"questions"`
		States    map[string]struct {
			State     string                     `json:"state"`
			Questions map[string]json.RawMessage `json:"questions"`
		} `json:"states"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	call := systemOneCall{body.State: sortedIDs(body.Questions)}
	for _, state := range body.States {
		call[state.State] = sortedIDs(state.Questions)
	}
	f.mu.Lock()
	f.calls = append(f.calls, call)
	f.mu.Unlock()
	if f.status != 0 {
		w.WriteHeader(f.status)
		_, _ = w.Write([]byte(`{"error":{"code":"unavailable"}}`))
		return
	}
	for _, ids := range call {
		if slices.Contains(ids, f.failing) {
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"error":{"code":"question_failed","question":"` + f.failing + `","question_error":"invalid_question"}}`))
			return
		}
	}
	answers := func(questions map[string]json.RawMessage) map[string]any {
		answered := make(map[string]any, len(questions))
		for id := range questions {
			answered[id] = map[string]any{"type": "choice", "choice": "a", "probabilities": map[string]float64{"a": 0.7, "b": 0.3}, "confidence": 0.7}
		}
		return answered
	}
	response := map[string]any{"model": body.Model, "answers": answers(body.Questions), "usage": map[string]int{"input_tokens": 3}}
	if len(body.States) > 0 {
		states := make(map[string]any, len(body.States))
		for name, state := range body.States {
			states[name] = map[string]any{"model": body.Model, "answers": answers(state.Questions), "usage": map[string]int{"input_tokens": 3}}
		}
		response["states"] = states
	}
	_ = json.NewEncoder(w).Encode(response)
}

func (f *fakeSystemOneAPI) answered() []systemOneCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

func sortedIDs(questions map[string]json.RawMessage) []string {
	ids := make([]string, 0, len(questions))
	for id := range questions {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

// fakeRemote serves api as the systemone deployment name.
func fakeRemote(t *testing.T, name string, api *fakeSystemOneAPI, deployment config.ModelDeployment) *RemoteSystemOne {
	t.Helper()
	server := httptest.NewServer(api)
	t.Cleanup(server.Close)
	deployment.Provider, deployment.Endpoint = config.SystemOneProvider, server.URL+"/v1/systemone"
	if deployment.Model == "" {
		deployment.Model = "vllm-sr/d3-ultra"
	}
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{name: deployment}
	remote, err := NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = remote.Close() })
	return remote
}

func TestARemoteDeploymentAnswersAStagesQuestionsInOneCall(t *testing.T) {
	api := &fakeSystemOneAPI{}
	remote := fakeRemote(t, "hosted-stage", api, config.ModelDeployment{})
	first := testutil.ToFloat64(stageDecisionCalls.WithLabelValues("hosted-stage", "first"))
	later := testutil.ToFloat64(stageDecisionCalls.WithLabelValues("hosted-stage", "later"))
	responses := make([]Response, 3)
	ask := func(index int, state, question string) func(ctx context.Context) error {
		return func(ctx context.Context) error {
			var err error
			responses[index], err = remote.Decide(ctx, "hosted-stage", Request{State: state, Questions: []Question{choiceQuestion(question)}})
			return err
		}
	}
	_, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"hosted-stage"}, ask: ask(0, "the request", "support_team")},
		asker{deployments: []string{"hosted-stage"}, delay: 20 * time.Millisecond, ask: ask(1, "the request", "domain_classifier:domain")},
		asker{deployments: []string{"hosted-stage"}, delay: 40 * time.Millisecond, ask: ask(2, "a tool result", "pii_classifier:pii")},
	)
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
	calls := api.answered()
	want := systemOneCall{"the request": {"domain_classifier:domain", "support_team"}, "a tool result": {"pii_classifier:pii"}}
	if len(calls) != 1 || !equalCall(calls[0], want) {
		t.Fatalf("calls %v, want one call %v", calls, want)
	}
	for index, id := range []string{"support_team", "domain_classifier:domain", "pii_classifier:pii"} {
		if answer := responses[index].Answers[id]; len(responses[index].Answers) != 1 || answer.Choice != "a" || answer.Error != "" {
			t.Fatalf("caller %d got %+v, want only its own answer to %s", index, responses[index].Answers, id)
		}
	}
	if got := testutil.ToFloat64(stageDecisionCalls.WithLabelValues("hosted-stage", "first")) - first; got != 1 {
		t.Fatalf("first-round calls %v, want 1", got)
	}
	if got := testutil.ToFloat64(stageDecisionCalls.WithLabelValues("hosted-stage", "later")) - later; got != 0 {
		t.Fatalf("later-round calls %v, want 0", got)
	}
}

func equalCall(got, want systemOneCall) bool {
	if len(got) != len(want) {
		return false
	}
	for state, ids := range want {
		if !slices.Equal(got[state], ids) {
			return false
		}
	}
	return true
}

func TestARemoteCallFailedOnOneQuestionAsksTheRestOnceMore(t *testing.T) {
	api := &fakeSystemOneAPI{failing: "bad"}
	remote := fakeRemote(t, "hosted-retry", api, config.ModelDeployment{})
	retries := testutil.ToFloat64(systemOneRetries.WithLabelValues("hosted-retry"))
	response, err := remote.Decide(context.Background(), "hosted-retry", Request{State: "x", Questions: []Question{choiceQuestion("good"), choiceQuestion("bad")}})
	if err != nil {
		t.Fatal(err)
	}
	if answer := response.Answers["good"]; answer.Choice != "a" || answer.Error != "" {
		t.Fatalf("good: %+v, want answered", answer)
	}
	if answer := response.Answers["bad"]; answer.Error != "invalid_question" || answer.Type != "choice" {
		t.Fatalf("bad: %+v, want unknown with the API's question error", answer)
	}
	calls := api.answered()
	if len(calls) != 2 || !equalCall(calls[1], systemOneCall{"x": {"good"}}) {
		t.Fatalf("calls %v, want the call and one more without bad", calls)
	}
	if got := testutil.ToFloat64(systemOneRetries.WithLabelValues("hosted-retry")) - retries; got != 1 {
		t.Fatalf("retries %v, want 1", got)
	}
}

func TestARetryAsksAnotherStateWhenTheRequestsOwnAskedOnlyTheFailedQuestion(t *testing.T) {
	api := &fakeSystemOneAPI{failing: "bad"}
	remote := fakeRemote(t, "hosted-promote", api, config.ModelDeployment{})
	responses := make([]Response, 2)
	_, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"hosted-promote"}, ask: func(ctx context.Context) error {
			var err error
			responses[0], err = remote.Decide(ctx, "hosted-promote", Request{State: "text-1", Questions: []Question{choiceQuestion("bad")}})
			return err
		}},
		asker{deployments: []string{"hosted-promote"}, delay: 20 * time.Millisecond, ask: func(ctx context.Context) error {
			var err error
			responses[1], err = remote.Decide(ctx, "hosted-promote", Request{State: "text-2", Questions: []Question{choiceQuestion("good")}})
			return err
		}},
	)
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
	calls := api.answered()
	if len(calls) != 2 || !equalCall(calls[0], systemOneCall{"text-1": {"bad"}, "text-2": {"good"}}) || !equalCall(calls[1], systemOneCall{"text-2": {"good"}}) {
		t.Fatalf("calls %v, want one stage call and a retry about text-2 alone", calls)
	}
	if answer := responses[0].Answers["bad"]; answer.Error != "invalid_question" {
		t.Fatalf("bad: %+v, want unknown", answer)
	}
	if answer := responses[1].Answers["good"]; answer.Choice != "a" || answer.Error != "" {
		t.Fatalf("good: %+v, want answered", answer)
	}
}

func TestARemoteCallThatFailsAsAWholeLeavesEveryQuestionUnknown(t *testing.T) {
	api := &fakeSystemOneAPI{status: http.StatusServiceUnavailable}
	remote := fakeRemote(t, "hosted-down", api, config.ModelDeployment{})
	_, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"hosted-down"}, ask: func(ctx context.Context) error {
			_, err := remote.Decide(ctx, "hosted-down", Request{State: "x", Questions: []Question{choiceQuestion("one")}})
			return err
		}},
		asker{deployments: []string{"hosted-down"}, ask: func(ctx context.Context) error {
			_, err := remote.Decide(ctx, "hosted-down", Request{State: "y", Questions: []Question{choiceQuestion("two")}})
			return err
		}},
	)
	for index, err := range errs {
		if !errors.Is(err, ErrUnavailable) {
			t.Fatalf("caller %d: %v, want unavailable", index, err)
		}
	}
	if calls := api.answered(); len(calls) != 1 {
		t.Fatalf("calls %v, want one, never retried", calls)
	}
}

func TestARemoteAnswerServesARepeatedCallUntilTheCacheIsReset(t *testing.T) {
	api := &fakeSystemOneAPI{}
	remote := fakeRemote(t, "hosted-cache", api, config.ModelDeployment{})
	request := Request{State: "x", Questions: []Question{choiceQuestion("one")}}
	for range 2 {
		if response, err := remote.Decide(context.Background(), "hosted-cache", request); err != nil || response.Answers["one"].Choice != "a" {
			t.Fatalf("%+v, %v", response, err)
		}
	}
	if calls := api.answered(); len(calls) != 1 {
		t.Fatalf("calls %d, want the repeated call answered from the cache", len(calls))
	}
	remote.ResetCache()
	if _, err := remote.Decide(context.Background(), "hosted-cache", request); err != nil {
		t.Fatal(err)
	}
	if calls := api.answered(); len(calls) != 2 {
		t.Fatalf("calls %d, want a new generation to ask again", len(calls))
	}
}

func TestARetriedAnswerIsNotCached(t *testing.T) {
	api := &fakeSystemOneAPI{failing: "bad"}
	remote := fakeRemote(t, "hosted-nocache", api, config.ModelDeployment{})
	request := Request{State: "x", Questions: []Question{choiceQuestion("good"), choiceQuestion("bad")}}
	for range 2 {
		if _, err := remote.Decide(context.Background(), "hosted-nocache", request); err != nil {
			t.Fatal(err)
		}
	}
	if calls := api.answered(); len(calls) != 4 {
		t.Fatalf("calls %d, want each answer with an unknown question asked again", len(calls))
	}
}

func TestExpiringResultCacheForgetsOldResults(t *testing.T) {
	cache := newExpiringResultCache(4, 20*time.Millisecond)
	key := decideKey(Request{State: "x"})
	cache.put(key, Response{Model: "m"})
	if _, ok := cache.get(key); !ok {
		t.Fatal("a fresh result is served")
	}
	time.Sleep(40 * time.Millisecond)
	if _, ok := cache.get(key); ok {
		t.Fatal("an expired result is served")
	}
}

func TestPreparationReadsTheListingAndFallsBackWithinItsDeadline(t *testing.T) {
	listed := &fakeSystemOneAPI{listing: `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["choice","noul","span"],"presets":["pii"]}]}`}
	remote := fakeRemote(t, "hosted-listed", listed, config.ModelDeployment{})
	card, err := remote.PrepareCard(context.Background(), "hosted-listed")
	if err != nil || !card.Answers("span") || !card.HasPreset("pii") || card.Device != RemoteDevice {
		t.Fatalf("card %+v, %v, want the listing's types and presets", card, err)
	}

	silent := &fakeSystemOneAPI{hold: make(chan struct{})}
	t.Cleanup(func() { close(silent.hold) })
	remote = fakeRemote(t, "hosted-silent", silent, config.ModelDeployment{Capabilities: &config.DeploymentCapabilities{QuestionTypes: []string{"choice", "noul"}, Presets: []string{"halu"}}})
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	started := time.Now()
	card, err = remote.PrepareCard(ctx, "hosted-silent")
	if err != nil || time.Since(started) > 2*time.Second {
		t.Fatalf("%v after %v, want the declared card at the preparation's deadline", err, time.Since(started))
	}
	if card.Answers("span") || !card.Answers("noul") || !card.HasPreset("halu") {
		t.Fatalf("card %+v, want the declared types and presets", card)
	}

	unlisted := &fakeSystemOneAPI{}
	remote = fakeRemote(t, "hosted-unlisted", unlisted, config.ModelDeployment{})
	if card, err = remote.PrepareCard(context.Background(), "hosted-unlisted"); err != nil || strings.Join(card.QuestionTypes, ",") != "" || !card.Answers("choice") {
		t.Fatalf("card %+v, %v, want the System One types", card, err)
	}
}
