package modelservice

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// jointVela2 answers like the 0.3B encoder in one respect: a Choice answer
// depends on the questions asked with it, so a stage that splits its
// questions gets other answers.
func jointVela2(id string) runtimetest.Model {
	model := vela2Fake(id)
	model.Joint = true
	return model
}

func choiceQuestion(id string) Question {
	return Question{ID: id, Type: "choice", Instructions: "Which one?", Choices: []Choice{{Key: "a"}, {Key: "b"}}}
}

// asker is one participant of a stage: the deployments it declares, how long
// it works before it asks, and what it asks.
type asker struct {
	deployments []string
	delay       time.Duration
	ask         func(ctx context.Context) error
}

// runStage runs the askers as declared participants of one bundle.
func runStage(window time.Duration, askers ...asker) (*Bundle, []error) {
	stage, bundle := WithBundle(context.Background(), window)
	errs := make([]error, len(askers))
	var wg sync.WaitGroup
	type joined struct {
		ctx   context.Context
		leave func()
	}
	participants := make([]joined, len(askers))
	for i, a := range askers {
		participants[i].ctx, participants[i].leave = bundle.JoinAsking(stage, a.deployments...)
	}
	for i, a := range askers {
		wg.Add(1)
		go func(i int, a asker) {
			defer wg.Done()
			defer participants[i].leave()
			time.Sleep(a.delay)
			if a.ask != nil {
				errs[i] = a.ask(participants[i].ctx)
			}
		}(i, a)
	}
	wg.Wait()
	return bundle, errs
}

func TestAStageAsksADeploymentOnceWhateverItsAskersTiming(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	const idle = 2 * time.Second
	stageAnswers := func(delay time.Duration) [2]Response {
		// A new word each run keeps the result cache out of it.
		text := "how do I refund a billing error " + delay.String()
		var answers [2]Response
		started := time.Now()
		var answered time.Duration
		_, errs := runStage(DefaultBundleWindow,
			asker{deployments: []string{"vela"}, ask: func(ctx context.Context) (err error) {
				answers[0], err = lease.Decide(ctx, "vela", Request{State: text, Questions: []Question{choiceQuestion("domain_classifier:domain")}})
				return err
			}},
			asker{deployments: []string{"vela"}, delay: delay, ask: func(ctx context.Context) (err error) {
				answers[1], err = lease.Decide(ctx, "vela", Request{State: text, Questions: []Question{choiceQuestion("fact_check_classifier:factcheck"), topicsQuestion("topics")}})
				answered = time.Since(started)
				return err
			}},
			// A participant that asks no decision model never holds the call back.
			asker{deployments: nil, delay: idle},
		)
		if err := errors.Join(errs...); err != nil {
			t.Fatal(err)
		}
		if answered > delay+idle/2 {
			t.Fatalf("the call waited for a participant that asks no decision model: answered after %v", answered)
		}
		return answers
	}
	together := stageAnswers(0)
	calls := len(runtime.Decisions())
	for _, delay := range []time.Duration{50 * time.Millisecond, 200 * time.Millisecond, 500 * time.Millisecond} {
		answers := stageAnswers(delay)
		if asked := len(runtime.Decisions()) - calls; asked != 1 {
			t.Fatalf("an asker %v late: %d decisions calls, want one", delay, asked)
		}
		calls = len(runtime.Decisions())
		if !reflect.DeepEqual(answers, together) {
			t.Fatalf("an asker %v late changed the answers:\n%+v\n%+v", delay, answers, together)
		}
	}
	last := runtime.Decisions()[calls-1]
	if len(last.Questions) != 3 || last.States != nil {
		t.Fatalf("the stage's three questions about one text in one call: %+v", last)
	}
}

func TestADeploymentsCallWaitsOnlyForItsOwnAskers(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"), runtimetest.Model{ID: "kai"})
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela", "kai"}})
	var answered time.Duration
	started := time.Now()
	_, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"kai"}, ask: func(ctx context.Context) error {
			_, err := lease.Decide(ctx, "kai", sampleRequest("is this hard"))
			answered = time.Since(started)
			return err
		}},
		asker{deployments: []string{"vela"}, delay: 300 * time.Millisecond, ask: func(ctx context.Context) error {
			_, err := lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion("q")}})
			return err
		}},
	)
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
	if answered > 200*time.Millisecond {
		t.Fatalf("kai's call waited %v for an asker of another deployment", answered)
	}
}

func TestAnUndeclaredParticipantHoldsDecisionsUntilItIsBlocked(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	ctx, bundle := WithBundle(context.Background(), DefaultBundleWindow)
	working := bundle.Join()
	leave := bundle.Join()
	released := make(chan struct{})
	go func() {
		time.Sleep(100 * time.Millisecond)
		close(released)
		working()
	}()
	if _, err := lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion("q")}}); err != nil {
		t.Fatal(err)
	}
	leave()
	select {
	case <-released:
	default:
		t.Fatal("a participant that may ask any deployment was not waited for")
	}
}

func TestAFanOutAsksEveryStateInItsCallersPlace(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	texts := []string{"the request", "an earlier message for person", "a tool result"}
	answers := make([]Response, len(texts))
	_, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"vela"}, ask: func(ctx context.Context) error {
			errs := make([]error, len(texts))
			Fan(ctx, len(texts), func(i int) {
				time.Sleep(time.Duration(i) * 30 * time.Millisecond)
				answers[i], errs[i] = lease.Decide(ctx, "vela", Request{State: texts[i], Questions: []Question{{ID: "pii_classifier:pii", Preset: "pii"}, choiceQuestion("prompt_guard:attack")}})
			})
			return errors.Join(errs...)
		}},
		asker{deployments: []string{"vela"}, ask: func(ctx context.Context) error {
			_, err := lease.Decide(ctx, "vela", Request{State: texts[0], Questions: []Question{choiceQuestion("domain_classifier:domain")}})
			return err
		}},
	)
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
	asked := runtime.Decisions()
	if len(asked) != 1 || asked[0].State != texts[0] || len(asked[0].Questions) != 3 || asked[0].States == nil || len(*asked[0].States) != 2 {
		t.Fatalf("one call: the request with its three questions and the two other texts as states: %+v", asked)
	}
	for i, text := range texts {
		spans := answers[i].Answers["pii_classifier:pii"].Spans
		if found := len(spans) > 0; found != (i == 1) || answers[i].Answers["prompt_guard:attack"].Type != "choice" {
			t.Fatalf("the answers about %q went to another caller: %+v", text, answers[i].Answers)
		}
	}
}

func TestAnAskerWaitingForAdmissionCountsAsAsked(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	gate := admission.NewSemaphore(1, 4, 0, admission.OverflowWait)
	admitted := func(ctx context.Context, id string) error {
		return func() error {
			ticket, err := gate.Acquire(ctx)
			if err != nil {
				return err
			}
			defer ticket()
			_, err = lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion(id)}})
			return err
		}()
	}
	done := make(chan []error, 1)
	go func() {
		_, errs := runStage(DefaultBundleWindow,
			asker{deployments: []string{"vela"}, ask: func(ctx context.Context) error { return admitted(ctx, "first") }},
			asker{deployments: []string{"vela"}, ask: func(ctx context.Context) error { return admitted(ctx, "second") }},
		)
		done <- errs
	}()
	select {
	case errs := <-done:
		if err := errors.Join(errs...); err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("an asker waiting for the slot a parked call holds deadlocked the stage")
	}
	if calls := len(runtime.Decisions()); calls != 2 {
		t.Fatalf("admission of one call at a time asks %d calls, want 2", calls)
	}
}

func TestOnceSentAShortDeadlineWaitsForTheSharedCall(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	runtime.SetDelay(300 * time.Millisecond)
	var short Response
	_, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"vela"}, ask: func(ctx context.Context) error {
			ctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
			defer cancel()
			var err error
			short, err = lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion("routing")}})
			return err
		}},
		asker{deployments: []string{"vela"}, ask: func(ctx context.Context) error {
			ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			_, err := lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion("safety")}})
			return err
		}},
	)
	if err := errors.Join(errs...); err != nil {
		t.Fatalf("a slow shared call must not drop the routing answer: %v", err)
	}
	if short.Answers["routing"].Type != "choice" {
		t.Fatalf("the routing answer: %+v", short.Answers)
	}
}

func TestBeforeItsQuestionsAreSentACallerKeepsItsOwnDeadline(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	var waited time.Duration
	_, errs := runStage(DefaultBundleWindow,
		asker{deployments: []string{"vela"}, ask: func(ctx context.Context) error {
			ctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
			defer cancel()
			started := time.Now()
			_, err := lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion("early")}})
			waited = time.Since(started)
			return err
		}},
		asker{deployments: []string{"vela"}, delay: 300 * time.Millisecond, ask: func(ctx context.Context) error {
			_, err := lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion("late")}})
			return err
		}},
	)
	if !errors.Is(errs[0], context.DeadlineExceeded) || waited > 250*time.Millisecond || errs[1] != nil {
		t.Fatalf("the early asker stops at its own deadline (%v after %v); the late one is answered (%v)", errs[0], waited, errs[1])
	}
	asked := runtime.Decisions()
	if len(asked) != 1 || len(asked[0].Questions) != 2 {
		t.Fatalf("the stage still sends both questions in one call: %+v", asked)
	}
}

func TestTheStageCountsEveryDecisionsCallAfterItsFirst(t *testing.T) {
	deployment := fmt.Sprintf("vela-%d", time.Now().UnixNano())
	runtime := runtimetest.New(jointVela2(deployment))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {deployment}})
	first := func() float64 { return testutil.ToFloat64(stageDecisionCalls.WithLabelValues(deployment, "first")) }
	later := func() float64 { return testutil.ToFloat64(stageDecisionCalls.WithLabelValues(deployment, "later")) }
	ask := func(id string) func(context.Context) error {
		return func(ctx context.Context) error {
			_, err := lease.Decide(ctx, deployment, Request{State: "x", Questions: []Question{choiceQuestion(id)}})
			return err
		}
	}
	if _, errs := runStage(DefaultBundleWindow, asker{deployments: []string{deployment}, ask: ask("a")}, asker{deployments: []string{deployment}, ask: ask("b")}); errors.Join(errs...) != nil {
		t.Fatal(errs)
	}
	if first() != 1 || later() != 0 {
		t.Fatalf("one stage, one call: first=%v later=%v", first(), later())
	}
	// An asker that declared nothing is not waited for, so its question goes
	// in a call of its own, which the stage counts.
	if _, errs := runStage(DefaultBundleWindow, asker{deployments: []string{deployment}, ask: ask("a")}, asker{delay: 100 * time.Millisecond, ask: ask("b")}); errors.Join(errs...) != nil {
		t.Fatal(errs)
	}
	if first() != 2 || later() != 1 {
		t.Fatalf("an undeclared asker's late question is a later call: first=%v later=%v", first(), later())
	}
}

func TestAStateCallIsCachedUnderEveryStateItAsks(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	ask := func(text string) func(context.Context) error {
		return func(ctx context.Context) error {
			_, err := lease.Decide(ctx, "vela", Request{State: text, Questions: []Question{choiceQuestion("prompt_guard:attack")}})
			return err
		}
	}
	both := func(a, b string) {
		if _, errs := runStage(DefaultBundleWindow, asker{deployments: []string{"vela"}, ask: ask(a)}, asker{deployments: []string{"vela"}, ask: ask(b)}); errors.Join(errs...) != nil {
			t.Fatal(errs)
		}
	}
	both("one", "two")
	both("two", "one")
	if calls := len(runtime.Decisions()); calls != 1 {
		t.Fatalf("the same texts in another order are the same call, served from the cache: %d calls", calls)
	}
	both("one", "three")
	if calls := len(runtime.Decisions()); calls != 2 {
		t.Fatalf("another text is another call: %d calls", calls)
	}
}
