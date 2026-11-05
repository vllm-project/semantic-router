package serving

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type relevanceQuestions struct {
	*fakeServices
	mu    sync.Mutex
	calls []modelservice.Request
	reply func(context.Context, modelservice.Request) (modelservice.Response, error)
}

func (s *relevanceQuestions) Decide(ctx context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	s.mu.Lock()
	s.calls = append(s.calls, request)
	reply := s.reply
	s.mu.Unlock()
	if reply != nil {
		return reply(ctx, request)
	}
	return relevanceAnswer(request), nil
}

func relevanceAnswer(request modelservice.Request) modelservice.Response {
	return modelservice.Response{InputTokens: 7, Answers: map[string]modelservice.Answer{request.Questions[0].ID: {Type: "score", Score: 2, InputCoverage: "complete"}}}
}

func decisionRankFixture() (*relevanceQuestions, config.ResolvedModelBinding) {
	services := &relevanceQuestions{fakeServices: newFake()}
	services.cards["rank"] = modelservice.ModelCard{ID: "rank", Family: "unrelated-family", ModelSHA256: strings.Repeat("b", 64), Revision: "one", Device: "cpu", Dtype: "float32", Ready: true, Surfaces: []string{"decisions"}, QuestionTypes: []string{"score"}, MaxInputTokens: 4096, MaxScanTokens: 8192}
	binding := spec("rank", config.DecisionTaskContract, config.ModelInputBudget{Overflow: "window", MaxTokens: 6000})
	binding.Name = config.RAGRerankerConsumer
	return services, binding
}

func TestDecisionRelevanceCoverageBoundsIdentityAndDiagnostics(t *testing.T) {
	services, spec := decisionRankFixture()
	runtime := New(services, nil)
	scorer, err := runtime.Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer scorer.Close()
	pairs := []tasks.QueryDocument{{Query: "q", Document: "first"}, {Query: "q", Document: "second"}}
	result, err := scorer.ScorePairs(context.Background(), "default", pairs)
	if err != nil {
		t.Fatal(err)
	}
	if result.Semantics().Unit != "decision_relevance" || result.Decision.InputTokens != 14 || result.InputMetadata() != nil || scorer.CacheIdentity() == "" {
		t.Fatalf("incorrect semantics: %+v", result)
	}
	services.mu.Lock()
	for _, request := range services.calls {
		q := request.Questions[0]
		if !q.RequireFullInput || q.Truncate || q.Stage != "retrieval" || q.TaskID != "rag_relevance" || request.MaxTokens != 6000 || request.Parts["request"] == "" || request.Parts["context"] == "" {
			t.Errorf("lost question contract: %+v", request)
		}
	}
	before := len(services.calls)
	services.mu.Unlock()
	if _, err = scorer.ScorePairs(context.Background(), "foreign", pairs); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe: %v", err)
	}
	for _, bad := range [][]tasks.QueryDocument{nil, make([]tasks.QueryDocument, 33), {{Query: "q", Document: strings.Repeat("x", decisionRelevanceMaxBytes)}}, {{Query: "", Document: "x"}}, {{Query: "q", Document: "\xff"}}} {
		if _, err = scorer.ScorePairs(context.Background(), "default", bad); !errors.Is(err, binding.ErrInvalidInput) {
			t.Fatalf("unbounded input: %v", err)
		}
	}
	services.mu.Lock()
	after := len(services.calls)
	services.mu.Unlock()
	if after != before {
		t.Fatal("invalid input reached runtime")
	}
	diagnostic, err := runtime.DiagnoseRerank(context.Background(), "default", config.RAGRerankerConsumer, pairs)
	if err != nil || diagnostic.Result.Decision == nil {
		t.Fatalf("decision diagnostic: %+v %v", diagnostic, err)
	}
	original := scorer.CacheIdentity()
	card := services.cards["rank"]
	card.Revision = "two"
	services.cards["rank"] = card
	changed, err := runtime.Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer changed.Close()
	if changed.CacheIdentity() == original {
		t.Fatal("revision reused identity")
	}
	spec.Deployment.Input.MaxTokens = 5000
	budget, err := runtime.Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer budget.Close()
	if budget.CacheIdentity() == changed.CacheIdentity() {
		t.Fatal("read policy reused identity")
	}
	scorer.Close()
	if _, err := scorer.ScorePairs(context.Background(), "default", pairs); err == nil {
		t.Fatal("closed scorer executed")
	}
}

func TestDecisionRelevanceRejectsInvalidJudgments(t *testing.T) {
	for _, kind := range []string{"missing", "extra", "foreign", "type", "nan", "inf", "range", "negative", "unknown", "coverage", "partial", "unavailable", "timeout", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			services, spec := decisionRankFixture()
			scorer, err := New(services, nil).Relevance(context.Background(), spec)
			if err != nil {
				t.Fatal(err)
			}
			defer scorer.Close()
			services.reply = func(ctx context.Context, request modelservice.Request) (modelservice.Response, error) {
				response := relevanceAnswer(request)
				id := request.Questions[0].ID
				answer := response.Answers[id]
				switch kind {
				case "missing":
					delete(response.Answers, id)
				case "extra":
					response.Answers["undeclared"] = answer
				case "foreign":
					delete(response.Answers, id)
					response.Answers["other"] = answer
				case "type":
					answer.Type = "noul"
				case "nan":
					answer.Score = math.NaN()
				case "inf":
					answer.Score = math.Inf(1)
				case "range":
					answer.Score = 4
				case "negative":
					answer.Score = -1
				case "unknown":
					answer.Error = "unknown"
				case "coverage":
					answer.InputCoverage = ""
				case "partial":
					answer.InputCoverage = "partial"
				case "unavailable":
					return modelservice.Response{}, modelservice.ErrUnavailable
				case "timeout", "cancel":
					<-ctx.Done()
					return modelservice.Response{}, ctx.Err()
				}
				if kind != "missing" && kind != "foreign" {
					response.Answers[id] = answer
				}
				return response, nil
			}
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer cancel()
			if kind == "cancel" {
				cancel()
			}
			output, err := scorer.ScorePairs(ctx, "default", []tasks.QueryDocument{{Query: "q", Document: "d"}})
			if err == nil || len(output.Scores) != 0 {
				t.Fatalf("invalid judgment produced scores: %+v %v", output, err)
			}
		})
	}
}

func TestDecisionRelevanceRequiresScoreCapability(t *testing.T) {
	services, spec := decisionRankFixture()
	card := services.cards["rank"]
	card.QuestionTypes = []string{"choice"}
	services.cards["rank"] = card
	if _, err := New(services, nil).Relevance(context.Background(), spec); err == nil {
		t.Fatal("accepted unsupported score")
	}
	if len(services.calls) != 0 {
		t.Fatal("unsupported capability reached inference")
	}
}
