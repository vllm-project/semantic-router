package graph

import (
	"context"
	"encoding/json"
	"errors"
	"regexp"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestRunAnswersWithTheCallResult(t *testing.T) {
	caller := &fakeCaller{}
	outcome, err := run(t, &Program{Name: "one", Steps: Sequence{call("draft", "small"), respond()}}, caller)
	if err != nil {
		t.Fatal(err)
	}
	if got := answerText(t, outcome); got != "ok" {
		t.Fatalf("answer %q", got)
	}
	if outcome.Hops != 1 || outcome.Usage.TotalTokens != 1 {
		t.Fatalf("hops %d usage %+v", outcome.Hops, outcome.Usage)
	}
	sent := caller.requests()[0]
	if sent.Hop != (routing.Hop{Decision: "panel", Recipe: "default", Iteration: 1}) {
		t.Fatalf("hop %+v", sent.Hop)
	}
	if sent.Request.Header.Get(":path") != "/v1/chat/completions" || modelOf(sent) != "small" {
		t.Fatalf("request %+v %s", sent.Request.Header, sent.Request.Body)
	}
	if got := outcome.Attempts; len(got) != 1 || got[0].Step != "draft" || got[0].Status != AttemptSucceeded {
		t.Fatalf("attempts %+v", got)
	}
}

func TestEveryStepRecordsItsDurationByNodeTypeAndTemplate(t *testing.T) {
	observations := func(nodeType string) uint64 {
		t.Helper()
		histogram, ok := metrics.RequestGraphNodeDuration.WithLabelValues(nodeType, "durations").(prometheus.Metric)
		if !ok {
			t.Fatal("the duration metric is not a histogram")
		}
		var sample dto.Metric
		if err := histogram.Write(&sample); err != nil {
			t.Fatal(err)
		}
		return sample.GetHistogram().GetSampleCount()
	}
	want := map[string]uint64{TypeParallel: 1, TypeCall: 2, TypeAggregate: 1, TypeRespond: 1}
	before := map[string]uint64{}
	for nodeType := range want {
		before[nodeType] = observations(nodeType)
	}
	program := &Program{Name: "durations", Steps: Sequence{
		parallel("panel", &Parallel{Branches: branchesOf("a", "b")}), aggregate("join", Concat{Separator: ","}), respond(),
	}}
	if _, err := run(t, program, &fakeCaller{}); err != nil {
		t.Fatal(err)
	}
	for nodeType, count := range want {
		if got := observations(nodeType) - before[nodeType]; got != count {
			t.Errorf("%s steps observed %d times, want %d", nodeType, got, count)
		}
	}
}

func TestRunWithoutRespondFails(t *testing.T) {
	_, err := run(t, &Program{Steps: Sequence{call("draft", "small")}}, &fakeCaller{})
	if !errors.Is(err, ErrNoResponse) {
		t.Fatalf("err %v", err)
	}
}

func TestHopLimitFailsClosedBeforeSending(t *testing.T) {
	caller := &fakeCaller{}
	program := &Program{
		Steps:  Sequence{{ID: "rounds", Type: TypeLoop, Node: &Loop{Body: Sequence{call("c", "m")}, MaxRounds: 5}}, respond()},
		Limits: Limits{MaxHops: 2},
	}
	outcome, err := run(t, program, caller)
	if !errors.Is(err, ErrHopLimit) {
		t.Fatalf("err %v", err)
	}
	if len(caller.requests()) != 2 || outcome.Hops != 2 {
		t.Fatalf("sent %d hops, outcome %d", len(caller.requests()), outcome.Hops)
	}
}

func TestTokenCeilingCancelsHopsInFlight(t *testing.T) {
	cancelled := make(chan struct{})
	slowInFlight := make(chan struct{})
	caller := &fakeCaller{answer: func(ctx context.Context, req *HopRequest) (*HopResponse, error) {
		if req.Model == "slow" {
			close(slowInFlight)
			<-ctx.Done()
			close(cancelled)
			return nil, ctx.Err()
		}
		<-slowInFlight
		return completion(req.Model, "big", 80), nil
	}}
	program := &Program{
		Steps:  Sequence{parallel("panel", &Parallel{Branches: branchesOf("fast", "slow")}), respond()},
		Limits: Limits{MaxTokens: 50},
	}
	outcome, err := run(t, program, caller)
	if !errors.Is(err, ErrTokenLimit) {
		t.Fatalf("err %v", err)
	}
	select {
	case <-cancelled:
	case <-time.After(5 * time.Second):
		t.Fatal("the slow hop was not cancelled")
	}
	if outcome.Usage.TotalTokens != 80 {
		t.Fatalf("usage %+v", outcome.Usage)
	}
}

type pricePerToken map[string]float64

func (p pricePerToken) Cost(model string, usage Usage) (float64, bool) {
	price, ok := p[model]
	return price * float64(usage.TotalTokens), ok
}

func TestCostCeiling(t *testing.T) {
	program := &Program{
		Steps:  Sequence{{ID: "rounds", Type: TypeLoop, Node: &Loop{Body: Sequence{call("c", "m")}, MaxRounds: 4}}, respond()},
		Limits: Limits{MaxCost: 2.5},
	}
	outcome, err := run(t, program, &fakeCaller{}, func(o *Options) { o.Pricing = pricePerToken{"m": 1} })
	if !errors.Is(err, ErrCostLimit) || outcome.Hops != 3 {
		t.Fatalf("err %v hops %d cost %v", err, outcome.Hops, outcome.Cost)
	}
	_, err = run(t, program, &fakeCaller{})
	if !errors.Is(err, ErrCostUnknown) {
		t.Fatalf("an unpriced model under a cost ceiling: %v", err)
	}
}

func TestTimeoutCancelsTheRun(t *testing.T) {
	saw := make(chan error, 1)
	caller := &fakeCaller{answer: func(ctx context.Context, _ *HopRequest) (*HopResponse, error) {
		<-ctx.Done()
		saw <- context.Cause(ctx)
		return nil, ctx.Err()
	}}
	program := &Program{Steps: Sequence{call("c", "m"), respond()}, Limits: Limits{Timeout: 20 * time.Millisecond}}
	_, err := run(t, program, caller)
	if !errors.Is(err, ErrDeadline) || !errors.Is(<-saw, ErrDeadline) {
		t.Fatalf("err %v", err)
	}
}

func TestClientCancellationReachesEveryHop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var started, stopped atomic.Int32
	caller := &fakeCaller{answer: func(ctx context.Context, _ *HopRequest) (*HopResponse, error) {
		if started.Add(1) == 3 {
			cancel()
		}
		<-ctx.Done()
		stopped.Add(1)
		return nil, ctx.Err()
	}}
	program := &Program{Steps: Sequence{parallel("panel", &Parallel{Branches: branchesOf("a", "b", "c")}), respond()}}
	_, err := Run(ctx, program, Input{Request: chatRequest(t, "hi")}, Options{Caller: caller})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	if stopped.Load() != 3 {
		t.Fatalf("%d of 3 hops saw the cancellation", stopped.Load())
	}
}

func TestPanicFailsTheStepNotTheProcess(t *testing.T) {
	program := &Program{Steps: Sequence{
		parallel("panel", &Parallel{Branches: []Sequence{{{ID: "bad", Type: "test", Node: panicNode{}}}, {call("c", "m")}}}),
		respond(),
	}}
	_, err := run(t, program, &fakeCaller{})
	var panicErr *PanicError
	var stepErr *StepError
	if !errors.As(err, &panicErr) || !errors.As(err, &stepErr) || stepErr.Step != "bad" {
		t.Fatalf("err %v", err)
	}
}

func TestLoopStopsWhenItsConditionHolds(t *testing.T) {
	var n atomic.Int32
	caller := &fakeCaller{answer: func(_ context.Context, req *HopRequest) (*HopResponse, error) {
		if n.Add(1) == 3 {
			return completion(req.Model, "FINAL answer", 1), nil
		}
		return completion(req.Model, "draft", 1), nil
	}}
	loop := &Loop{Body: Sequence{call("c", "m")}, Until: ContentMatches(regexp.MustCompile(`^FINAL`)), MaxRounds: 10}
	outcome, err := run(t, &Program{Steps: Sequence{{ID: "refine", Type: TypeLoop, Node: loop}, respond()}}, caller)
	if err != nil || answerText(t, outcome) != "FINAL answer" || outcome.Hops != 3 {
		t.Fatalf("err %v hops %d", err, outcome.Hops)
	}
	if round, _ := Round.Get(&State{Values: outcome.Values}); round != 3 {
		t.Fatalf("round %d", round)
	}
}

func TestBranchFollowsSignalsAndResults(t *testing.T) {
	caller := &fakeCaller{}
	branch := &Branch{
		Cases: []Case{
			{When: SignalMatched("domain", "math"), Then: Sequence{call("math", "math-model")}},
			{When: SignalMatched("domain", "code"), Then: Sequence{call("code", "code-model")}},
		},
		Else: Sequence{call("general", "general-model")},
	}
	program := &Program{Steps: Sequence{{ID: "route", Type: TypeBranch, Node: branch}, respond()}}
	for signal, want := range map[string]string{"code": "code-model", "art": "general-model"} {
		before := len(caller.requests())
		_, err := run(t, program, caller, func(o *Options) { o.Signals = map[string][]string{"domain": {signal}} })
		if err != nil || caller.models()[before] != want {
			t.Fatalf("signal %s: err %v models %v", signal, err, caller.models())
		}
	}
}

func TestRespondFinalHandsTheCallToTheGateway(t *testing.T) {
	caller := &fakeCaller{}
	program := &Program{Steps: Sequence{
		{ID: "system", Type: TypeTransform, Node: &Transform{Transformer: SystemPrompt{Content: "Be brief.", Mode: SystemReplace}}},
		{ID: "final", Type: TypeRespond, Node: &Respond{Final: "large"}},
	}}
	outcome, err := run(t, program, caller)
	if err != nil || outcome.Response.Final == nil || outcome.Response.Final.Model != "large" || outcome.Hops != 0 {
		t.Fatalf("err %v outcome %+v", err, outcome)
	}
	messages, _ := outcome.Response.Final.Request.Messages()
	if MessageRole(messages[0]) != "system" || MessageText(messages[0]) != "Be brief." {
		t.Fatalf("messages %v", messages)
	}
}

func TestAttemptEvidenceIsBounded(t *testing.T) {
	program := &Program{Steps: Sequence{{ID: "many", Type: TypeLoop, Node: &Loop{Body: Sequence{call("c", "m")}, MaxRounds: 130}}, respond()}}
	outcome, err := run(t, program, &fakeCaller{})
	if err != nil || len(outcome.Attempts) != maxAttempts || outcome.DroppedAttempts != 30 || outcome.Hops != 130 {
		t.Fatalf("err %v attempts %d dropped %d", err, len(outcome.Attempts), outcome.DroppedAttempts)
	}
}

func TestCallStepsSetFieldsAndDecision(t *testing.T) {
	caller := &fakeCaller{}
	fields := map[string]json.RawMessage{"temperature": []byte("0")}
	step := Step{ID: "judge", Type: TypeCall, Node: &Call{Model: "judge", Decision: "judging", Fields: fields}}
	if _, err := run(t, &Program{Steps: Sequence{step, respond()}}, caller); err != nil {
		t.Fatal(err)
	}
	sent := caller.requests()[0]
	parsed, _ := ParseRequest(sent.Request.Body)
	if string(parsed.fields["temperature"]) != "0" || sent.Hop.Decision != "judging" || sent.Hop.Recipe != "default" {
		t.Fatalf("sent %+v %s", sent.Hop, sent.Request.Body)
	}
	if !slices.Equal(caller.models(), []string{"judge"}) {
		t.Fatalf("models %v", caller.models())
	}
}
