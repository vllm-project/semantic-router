package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
// "a", of a standard request or of each entry of a batch, and refuses a body
// that mixes the two forms. It fails a call that asks failing as a whole on
// that question, or every call with status. With failingIn, failing fails
// only about that text, and a batch's failure names its entry, as the model
// runtime names it.
type fakeSystemOneAPI struct {
	mu        sync.Mutex
	calls     []systemOneCall
	batches   []bool
	failing   string
	failingIn string
	status    int
	listing   string
	hold      chan struct{}
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
		State     *string                    `json:"state"`
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
	batch := body.States != nil
	if batch && (body.State != nil || body.Questions != nil) || !batch && body.State == nil {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"code":"invalid_request","message":"a request is a standard request or a batch, not both"}}`))
		return
	}
	call := systemOneCall{}
	if !batch {
		call[*body.State] = sortedIDs(body.Questions)
	}
	for _, state := range body.States {
		call[state.State] = sortedIDs(state.Questions)
	}
	f.mu.Lock()
	f.calls = append(f.calls, call)
	f.batches = append(f.batches, batch)
	f.mu.Unlock()
	if f.status != 0 {
		w.WriteHeader(f.status)
		_, _ = w.Write([]byte(`{"error":{"code":"unavailable"}}`))
		return
	}
	for text, ids := range call {
		if !slices.Contains(ids, f.failing) || f.failingIn != "" && text != f.failingIn {
			continue
		}
		entry := ""
		for name, state := range body.States {
			if f.failingIn != "" && state.State == text {
				entry = `,"state":"` + name + `"`
			}
		}
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"error":{"code":"question_failed","question":"` + f.failing + `"` + entry + `,"question_error":"invalid_question"}}`))
		return
	}
	answers := func(questions map[string]json.RawMessage) map[string]any {
		answered := make(map[string]any, len(questions))
		for id := range questions {
			answered[id] = map[string]any{"type": "choice", "choice": "a", "probabilities": map[string]float64{"a": 0.7, "b": 0.3}, "confidence": 0.7}
		}
		return answered
	}
	response := map[string]any{"model": body.Model, "usage": map[string]int{"input_tokens": 3 * len(call)}}
	if !batch {
		response["answers"] = answers(body.Questions)
	} else {
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

// forms is whether each call was a batch.
func (f *fakeSystemOneAPI) forms() []bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.batches)
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
	if len(calls) != 1 || !equalCall(calls[0], want) || !slices.Equal(api.forms(), []bool{true}) {
		t.Fatalf("calls %v (batch %v), want one batch %v", calls, api.forms(), want)
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

// The stage's questions about every text it reads go to the deployment in
// one batch, however many texts there are.
func TestARemoteStageAsksAllItsStatesInOneBatch(t *testing.T) {
	api := &fakeSystemOneAPI{}
	remote := fakeRemote(t, "hosted-batch", api, config.ModelDeployment{})
	askers := make([]asker, 20)
	want := systemOneCall{}
	for index := range askers {
		text := fmt.Sprintf("candidate %d", index)
		want[text] = []string{"match:relevance"}
		askers[index] = asker{deployments: []string{"hosted-batch"}, ask: func(ctx context.Context) error {
			_, err := remote.Decide(ctx, "hosted-batch", Request{State: text, Questions: []Question{choiceQuestion("match:relevance")}})
			return err
		}}
	}
	if _, errs := runStage(DefaultBundleWindow, askers...); errors.Join(errs...) != nil {
		t.Fatal(errors.Join(errs...))
	}
	if calls := api.answered(); len(calls) != 1 || !equalCall(calls[0], want) || !slices.Equal(api.forms(), []bool{true}) {
		t.Fatalf("calls %v (batch %v), want one batch with an entry per text", calls, api.forms())
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
	if len(calls) != 2 || !equalCall(calls[1], systemOneCall{"x": {"good"}}) || !slices.Equal(api.forms(), []bool{false, false}) {
		t.Fatalf("calls %v (batch %v), want the standard request and one more without bad", calls, api.forms())
	}
	if got := testutil.ToFloat64(systemOneRetries.WithLabelValues("hosted-retry")) - retries; got != 1 {
		t.Fatalf("retries %v, want 1", got)
	}
}

// A retry asks only about the states with questions left: a batch about
// several, a standard request about one.
func TestARetryAsksOnlyTheStatesWithOtherQuestions(t *testing.T) {
	for name, tc := range map[string]struct {
		texts   []string
		retried systemOneCall
		forms   []bool
	}{
		"one state left":  {[]string{"text-1", "text-2"}, systemOneCall{"text-2": {"good"}}, []bool{true, false}},
		"two states left": {[]string{"text-1", "text-2", "text-3"}, systemOneCall{"text-2": {"good"}, "text-3": {"good"}}, []bool{true, true}},
	} {
		t.Run(name, func(t *testing.T) {
			api := &fakeSystemOneAPI{failing: "bad"}
			remote := fakeRemote(t, "hosted-retry-states", api, config.ModelDeployment{})
			responses := make([]Response, len(tc.texts))
			askers := make([]asker, len(tc.texts))
			for index, text := range tc.texts {
				question := choiceQuestion("good")
				if index == 0 {
					question = choiceQuestion("bad")
				}
				askers[index] = asker{deployments: []string{"hosted-retry-states"}, ask: func(ctx context.Context) error {
					var err error
					responses[index], err = remote.Decide(ctx, "hosted-retry-states", Request{State: text, Questions: []Question{question}})
					return err
				}}
			}
			if _, errs := runStage(DefaultBundleWindow, askers...); errors.Join(errs...) != nil {
				t.Fatal(errors.Join(errs...))
			}
			calls := api.answered()
			if len(calls) != 2 || !slices.Equal(api.forms(), tc.forms) || !equalCall(calls[1], tc.retried) {
				t.Fatalf("calls %v (batch %v), want the stage's call and a retry about %v", calls, api.forms(), tc.retried)
			}
			if answer := responses[0].Answers["bad"]; answer.Error != "invalid_question" {
				t.Fatalf("bad: %+v, want unknown", answer)
			}
			for _, response := range responses[1:] {
				if answer := response.Answers["good"]; answer.Choice != "a" || answer.Error != "" {
					t.Fatalf("good: %+v, want answered", answer)
				}
			}
		})
	}
}

// A question the API fails about one entry of a batch, and names that entry,
// is unknown about that entry only: a long earlier turn does not fail the
// same question about the current one.
func TestAQuestionFailedAboutOneEntryIsAnsweredAboutTheOthers(t *testing.T) {
	texts := []string{"a long earlier turn", "What is 2+2?"}
	for name, tc := range map[string]struct {
		questions []string
		retried   systemOneCall
		forms     []bool
	}{
		"only that question":    {[]string{"jailbreak"}, systemOneCall{texts[1]: {"jailbreak"}}, []bool{true, false}},
		"with another question": {[]string{"domain", "jailbreak"}, systemOneCall{texts[0]: {"domain"}, texts[1]: {"domain", "jailbreak"}}, []bool{true, true}},
	} {
		t.Run(name, func(t *testing.T) {
			api := &fakeSystemOneAPI{failing: "jailbreak", failingIn: texts[0]}
			remote := fakeRemote(t, "hosted-entry-failure", api, config.ModelDeployment{})
			questions := make([]Question, 0, len(tc.questions))
			for _, id := range tc.questions {
				questions = append(questions, choiceQuestion(id))
			}
			responses := make([]Response, len(texts))
			askers := make([]asker, len(texts))
			for index, text := range texts {
				askers[index] = asker{deployments: []string{"hosted-entry-failure"}, ask: func(ctx context.Context) error {
					var err error
					responses[index], err = remote.Decide(ctx, "hosted-entry-failure", Request{State: text, Questions: questions})
					return err
				}}
			}
			if _, errs := runStage(DefaultBundleWindow, askers...); errors.Join(errs...) != nil {
				t.Fatal(errors.Join(errs...))
			}
			calls := api.answered()
			if len(calls) != 2 || !slices.Equal(api.forms(), tc.forms) || !equalCall(calls[1], tc.retried) {
				t.Fatalf("calls %v (batch %v), want the stage's call and a retry about %v", calls, api.forms(), tc.retried)
			}
			if answer := responses[0].Answers["jailbreak"]; answer.Error != "invalid_question" {
				t.Fatalf("jailbreak about %q: %+v, want unknown", texts[0], answer)
			}
			for index, response := range responses {
				for _, id := range tc.questions {
					if answer := response.Answers[id]; (index > 0 || id != "jailbreak") && (answer.Choice != "a" || answer.Error != "") {
						t.Fatalf("%s about %q: %+v, want answered", id, texts[index], answer)
					}
				}
			}
		})
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
