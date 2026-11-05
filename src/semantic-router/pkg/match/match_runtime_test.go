package match

import (
	"context"
	"net/http/httptest"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// runtimeAnswerer asks a lease's deployments, with the cards their workers
// last reported.
type runtimeAnswerer struct{ *modelservice.Lease }

func (a runtimeAnswerer) Card(deployment string) (modelservice.ModelCard, bool) {
	return a.CurrentCard(deployment)
}

// judgeRuntime serves one decision model on the fake runtime as the
// configuration's decision model.
func judgeRuntime(t *testing.T) (*Matcher, *runtimetest.Runtime, *modelservice.Lease) {
	t.Helper()
	runtime := runtimetest.New(runtimetest.Model{ID: "vela"})
	server := httptest.NewServer(runtime.Handler())
	t.Cleanup(server.Close)
	cfg := &config.RouterConfig{}
	cfg.DecisionModel = "vela"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"vela": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL}}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(cfg.ModelDeployments)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if _, err := lease.Card(ctx, "vela"); err != nil {
		t.Fatal(err)
	}
	return New(cfg, runtimeAnswerer{lease}), runtime, lease
}

// A judge asks about every recalled candidate in one decisions call: one
// state per candidate, each asking the relevance question.
func TestAJudgeAsksAboutEveryCandidateInOneCall(t *testing.T) {
	matcher, runtime, _ := judgeRuntime(t)
	store := &index{recalled: recalled(0.9, 0.8, 0.7, 0.6, 0.5)}
	result, err := matcher.Match(context.Background(), config.MatchRAG, nil, Input{Request: "what is our refund window?", Index: store})
	if err != nil {
		t.Fatal(err)
	}
	calls := runtime.Decisions()
	if len(calls) != 1 || calls[0].States == nil || len(*calls[0].States) != 4 {
		t.Fatalf("%d decisions calls, want one with the 5 candidates as its states", len(calls))
	}
	texts := []string{stateCandidate(t, calls[0].State)}
	for _, state := range *calls[0].States {
		if _, ok := state.Questions["match:relevance"]; !ok || len(state.Questions) != 1 {
			t.Fatalf("state questions %v", state.Questions)
		}
		texts = append(texts, stateCandidate(t, state.State))
	}
	sort.Strings(texts)
	if !reflect.DeepEqual(texts, []string{"doc 0", "doc 1", "doc 2", "doc 3", "doc 4"}) {
		t.Fatalf("states read %v", texts)
	}
	if len(result.Kept) != 5 || result.Stages[1].Outcome != OutcomeOK {
		t.Fatalf("the fake answers yes at 0.8, so every candidate passes 0.5: %+v", result)
	}
}

// A caller that bundles a choice judge with its own question about the same
// request (tool selection's tool_need) sends both in one call, about one
// state.
func TestAChoiceJudgeSharesItsCallWithTheCallersQuestions(t *testing.T) {
	matcher, runtime, lease := judgeRuntime(t)
	request := "book a room for the design review"
	tools := []Candidate{{ID: "calendar", Text: "Book a meeting room"}, {ID: "search", Text: "Search the docs"}, {ID: "billing", Text: "Look up an invoice"}}
	stage, bundle := modelservice.WithBundle(context.Background(), 0)
	leave := bundle.Join()
	var result Result
	var need modelservice.Response
	errs := make([]error, 2)
	modelservice.Fan(stage, 2, func(i int) {
		if i == 0 {
			result, errs[0] = matcher.Match(stage, config.MatchToolsSelect, nil, Input{Request: request, Candidates: tools})
			return
		}
		need, errs[1] = lease.Decide(stage, "vela", modelservice.Request{State: request, Questions: []modelservice.Question{
			{ID: "tools:tool_need", Type: config.SystemOneQuestionNoul, Instructions: "Does answering this request need a call to one of the available tools?"},
		}})
	})
	leave()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	calls := runtime.Decisions()
	if len(calls) != 1 || calls[0].States != nil || calls[0].State != request || len(calls[0].Questions) != 2 {
		t.Fatalf("decisions calls %+v, want one about the request with both questions", calls)
	}
	if choices := calls[0].Questions["match:tool_rank"].Choices; choices == nil || len(*choices) != 3 {
		t.Fatalf("tool_rank options %+v", calls[0].Questions["match:tool_rank"])
	}
	if got := ids(result.Kept); !reflect.DeepEqual(got, []string{"calendar", "search", "billing"}) {
		t.Fatalf("kept %v: the fake chooses the first option at 0.7", got)
	}
	if answer := need.Answers["tools:tool_need"]; answer.Error != "" || answer.Noul != 0.8 {
		t.Fatalf("tool_need = %+v", answer)
	}
}

func stateCandidate(t *testing.T, state any) string {
	t.Helper()
	parts, ok := state.(map[string]any)
	if !ok {
		t.Fatalf("state %#v is no object of parts", state)
	}
	text, _ := parts[candidatePart].(string)
	return text
}
