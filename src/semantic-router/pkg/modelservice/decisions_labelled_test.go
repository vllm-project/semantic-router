package modelservice

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func vela2Fake(id string) runtimetest.Model {
	return runtimetest.Model{ID: id, Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON", "EMAIL_ADDRESS"}, BroadHead: true}}
}

func topicsQuestion(id string) Question {
	return Question{ID: id, Type: "set", Instructions: "Which topics?", Labels: []Choice{{Key: "billing", Description: "payments"}, {Key: "shipping"}}}
}

func entitiesQuestion(id string) Question {
	return Question{ID: id, Type: "span", Instructions: "Which spans name a city?", Labels: []Choice{{Key: "city", Description: "a city name"}}, Head: "router"}
}

func TestSetAndSpanQuestionsEncodeTheirLabelsInOrder(t *testing.T) {
	threshold := 0.4
	question := topicsQuestion("topics")
	question.Labels = []Choice{{Key: "z", Description: "last"}, {Key: "a"}}
	question.Threshold = &threshold
	encoded, err := json.Marshal(encodeQuestion(question))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"criteria":{"z":"last","a":null}`) || !strings.Contains(string(encoded), `"threshold":0.4`) || strings.Contains(string(encoded), "choices") {
		t.Fatalf("set question = %s", encoded)
	}
	encoded, _ = json.Marshal(encodeQuestion(Question{ID: "pii", Preset: "pii"}))
	if string(encoded) != `{"preset":"pii"}` {
		t.Fatalf("a preset question names only its preset: %s", encoded)
	}
	encoded, _ = json.Marshal(encodeQuestion(entitiesQuestion("cities")))
	if !strings.Contains(string(encoded), `"head":"router"`) {
		t.Fatalf("span question = %s", encoded)
	}
}

func TestDecodeReadsSetsSpansAndTheirThresholds(t *testing.T) {
	noul, setLabel := 0.97, 0.9
	answerType := "noul"
	sets := map[string]api.SetAnswer{"topics": {Selected: []string{"billing"}, Probabilities: map[string]float64{"billing": 0.9, "shipping": 0.1}}}
	spans := map[string][]api.Span{"cities": {{Label: "city", Start: 0, End: 5, Text: "Paris", Probability: 0.93}}}
	thresholds := map[string]float64{"topics": 0.3, "cities": 0.5}
	heads := map[string]string{"cities": "router"}
	body := api.DecisionResponse{
		Answers: map[string]api.Answer{
			"cities":          {Type: &answerType, Noul: &noul},
			"topics.billing":  {Type: &answerType, Noul: &setLabel},
			"topics.shipping": {Type: &answerType, Noul: &setLabel},
		},
		Sets: &sets, Spans: &spans, Thresholds: &thresholds, SpanHeads: &heads,
	}
	response := decodeResponse(body, []Question{topicsQuestion("topics"), entitiesQuestion("cities"), {ID: "missing", Type: "span"}})
	set := response.Answers["topics"]
	if set.Type != "set" || set.Probabilities["billing"] != 0.9 || len(set.Selected) != 1 || set.Threshold != 0.3 {
		t.Fatalf("set answer = %+v", set)
	}
	span := response.Answers["cities"]
	if span.Type != "span" || span.Noul != 0.97 || len(span.Spans) != 1 || span.Spans[0].Text != "Paris" || span.Head != "router" || span.Threshold != 0.5 {
		t.Fatalf("span answer = %+v", span)
	}
	if _, ok := response.Answers["topics.billing"]; ok {
		t.Fatal("a Set label's Noul repeats sets and is not an answer of its own")
	}
	if _, ok := response.Answers["missing"]; ok {
		t.Fatal("an unanswered question has no answer")
	}
	nan := map[string][]api.Span{"cities": {{Label: "city", Start: 0, End: 1, Text: "P", Probability: nanValue()}}}
	body.Spans = &nan
	if got := decodeResponse(body, []Question{entitiesQuestion("cities")}).Answers["cities"]; got.Error != "invalid_model_output" {
		t.Fatalf("a non-finite span probability is invalid model output, got %+v", got)
	}
}

func nanValue() float64 {
	zero := 0.0
	return zero / zero
}

// stage runs calls as participants of one bundle, as a request stage's signals do.
func stage(window time.Duration, calls ...func(context.Context)) *Bundle {
	ctx, bundle := WithBundle(context.Background(), window)
	var wg sync.WaitGroup
	for _, call := range calls {
		leave := bundle.Join()
		wg.Add(1)
		go func(call func(context.Context)) {
			defer wg.Done()
			defer leave()
			call(ctx)
		}(call)
	}
	wg.Wait()
	return bundle
}

func TestBundleAsksOneModelTheStagesQuestionsAboutOneStateInOneTask(t *testing.T) {
	runtime := runtimetest.New(vela2Fake("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	signals := Request{State: "refund my billing to Tom", Questions: []Question{topicsQuestion("topics"), {ID: "urgent", Type: "noul", Instructions: "Urgent?"}}}
	pii := Request{State: "refund my billing to Tom", Questions: []Question{{ID: "pii_classifier:pii", Preset: "pii"}}}
	other := Request{State: "a different message", Questions: []Question{{ID: "pii_classifier:pii", Preset: "pii"}}}
	var answers [3]Response
	var errs [3]error
	stage(time.Second,
		func(ctx context.Context) { answers[0], errs[0] = lease.Decide(ctx, "vela", signals) },
		func(ctx context.Context) { answers[1], errs[1] = lease.Decide(ctx, "vela", pii) },
		func(ctx context.Context) { answers[2], errs[2] = lease.Decide(ctx, "vela", other) },
	)
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if calls, tasks := runtime.Bundles(); calls != 1 || tasks != 2 {
		t.Fatalf("one bundle with one task per state expected: %d bundles, %d tasks", calls, tasks)
	}
	if len(answers[0].Answers) != 2 || answers[0].Answers["topics"].Probabilities["billing"] != 0.9 || answers[0].Answers["topics"].Selected[0] != "billing" {
		t.Fatalf("the signals' caller gets its own answers: %+v", answers[0].Answers)
	}
	if len(answers[1].Answers) != 1 || answers[1].Answers["pii_classifier:pii"].Type != "span" {
		t.Fatalf("the PII caller gets its own answer: %+v", answers[1].Answers)
	}
}

func TestBundleKeepsCallsWhoseAnswerKeysCouldCollideApart(t *testing.T) {
	runtime := runtimetest.New(vela2Fake("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	set := Request{State: "x", Questions: []Question{topicsQuestion("topics")}}
	label := Request{State: "x", Questions: []Question{{ID: "topics.billing", Type: "noul", Instructions: "?"}}}
	same := Request{State: "x", Questions: []Question{{ID: "topics", Type: "noul", Instructions: "?"}}}
	stage(time.Second,
		func(ctx context.Context) { _, _ = lease.Decide(ctx, "vela", set) },
		func(ctx context.Context) { _, _ = lease.Decide(ctx, "vela", label) },
		func(ctx context.Context) { _, _ = lease.Decide(ctx, "vela", same) },
	)
	if _, tasks := runtime.Bundles(); tasks != 3 {
		t.Fatalf("an equal ID or a Set label's answer key keeps calls apart: %d tasks", tasks)
	}
}

func TestCacheKeysTheFusedCallSoAnAnswerAskedAloneIsNotServedFromIt(t *testing.T) {
	runtime := runtimetest.New(vela2Fake("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	signals := Request{State: "billing for Tom", Questions: []Question{topicsQuestion("topics")}}
	pii := Request{State: "billing for Tom", Questions: []Question{{ID: "pii_classifier:pii", Preset: "pii"}}}
	both := func() {
		stage(time.Second,
			func(ctx context.Context) { _, _ = lease.Decide(ctx, "vela", signals) },
			func(ctx context.Context) { _, _ = lease.Decide(ctx, "vela", pii) },
		)
	}
	both()
	both()
	if _, tasks := runtime.Bundles(); tasks != 1 {
		t.Fatalf("the repeated fused call is a cache hit: %d tasks", tasks)
	}
	stage(time.Second, func(ctx context.Context) {
		if response, err := lease.Decide(ctx, "vela", signals); err != nil || response.Answers["topics"].Type != "set" {
			t.Errorf("alone: %+v %v", response, err)
		}
	})
	if _, tasks := runtime.Bundles(); tasks != 2 {
		t.Fatalf("questions asked alone are a new call, not the fused call's cached part: %d tasks", tasks)
	}
}

func TestDecisionModelsWithoutSetAndSpanAnswerThemInvalid(t *testing.T) {
	runtime := runtimetest.New(runtimetest.Model{ID: "kai"})
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"kai"}})
	card, err := lease.Card(context.Background(), "kai")
	if err != nil {
		t.Fatal(err)
	}
	if card.Answers("set") || card.Answers("span") || !card.Answers("choice") || card.HasPreset("pii") {
		t.Fatalf("a System One card = %+v", card)
	}
	response, err := lease.Decide(context.Background(), "kai", Request{State: "x", Questions: []Question{topicsQuestion("topics")}})
	if err != nil || response.Answers["topics"].Error != "invalid_question" {
		t.Fatalf("set to a System One model: %+v %v", response, err)
	}
	vela := runtimetest.New(vela2Fake("vela"))
	lease = attachedLease(t, map[*runtimetest.Runtime][]string{vela: {"vela"}})
	if card, _ = lease.Card(context.Background(), "vela"); !card.Answers("span") || !card.HasPreset("halu") {
		t.Fatalf("a Vela 2.0 card = %+v", card)
	}
}

func TestAScanBudgetTravelsAsMaxTokensAndKeysTheCache(t *testing.T) {
	request := Request{State: "a long text", Questions: []Question{{ID: "q", Type: "noul", Instructions: "Is it?"}}}
	body, err := encodeDecisionRequest(context.Background(), request)
	if err != nil || body.Options == nil || body.Options.MaxTokens != nil || body.Questions["q"].Overflow != nil {
		t.Fatalf("no scan budget, no max_tokens: %+v %v", body.Options, err)
	}
	scanned := request
	scanned.MaxTokens = 4096
	body, err = encodeDecisionRequest(context.Background(), scanned)
	if err != nil || body.Options == nil || body.Options.MaxTokens == nil || *body.Options.MaxTokens != 4096 {
		t.Fatalf("scan budget: %+v %v", body.Options, err)
	}
	truncating := request
	truncating.Questions = []Question{{ID: "q", Type: "noul", Instructions: "Is it?", Truncate: true}}
	body, err = encodeDecisionRequest(context.Background(), truncating)
	if err != nil || body.Questions["q"].Overflow == nil || *body.Questions["q"].Overflow != "truncate" {
		t.Fatalf("a truncating question: %+v %v", body.Questions["q"], err)
	}
	if decideKey(request) == decideKey(scanned) || decideKey(request) == decideKey(truncating) {
		t.Fatal("a scan budget or a truncating question changes the answers, so it keys the cache")
	}
	bounded := boundedRead(Request{MaxTokens: 4096, Questions: truncating.Questions})
	if bounded.MaxTokens != 0 || bounded.Questions[0].Truncate || !truncating.Questions[0].Truncate {
		t.Fatalf("a model without a scan budget takes neither option, and the caller's request is kept: %+v", bounded)
	}
}
