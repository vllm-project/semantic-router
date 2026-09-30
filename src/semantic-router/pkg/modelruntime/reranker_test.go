package modelruntime

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type lifecyclePairScorer struct{ closed int }

func (s *lifecyclePairScorer) Close() error        { s.closed++; return nil }
func (*lifecyclePairScorer) CacheIdentity() string { return "actual-bytes" }
func (*lifecyclePairScorer) ScorePairs(context.Context, string, []tasks.QueryDocument) (tasks.RelevanceScores, error) {
	return tasks.RelevanceScores{}, nil
}

func TestPrepareRerankersReachabilityAndCandidateRollback(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"local": {Provider: "candle", Artifact: "model"}}
	for _, name := range []config.RecipeName{"a", "b", "unused"} {
		cfg.Recipes = append(cfg.Recipes, config.RoutingRecipe{Name: name, Profile: config.RoutingProfile{ModelBindings: map[string]config.ModelBinding{config.RAGRerankerConsumer: {Deployment: "local", Adapter: "vela_reranker", Contract: config.RelevanceScoresContract}}, Decisions: []config.Decision{{Name: "retrieve", Plugins: []config.DecisionPlugin{{Type: "rag", Configuration: config.MustStructuredPayload(config.RAGPluginConfig{Enabled: true, Backend: "vectorstore", Rerank: &config.RAGRerankConfig{}})}}}}}})
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"first"}, Recipe: "a"}, {ModelNames: []string{"second"}, Recipe: "b"}}
	a := &lifecyclePairScorer{}
	calls := 0
	if _, err := prepareRerankers(context.Background(), cfg, func(_ context.Context, spec config.ResolvedModelBinding) (PairScorer, error) {
		calls++
		if spec.Recipe == "a" {
			return a, nil
		}
		return nil, errors.New("candidate failed")
	}); err == nil || a.closed != 1 || calls != 2 {
		t.Fatalf("candidate rollback: err=%v close=%d calls=%d", err, a.closed, calls)
	}
	models, err := prepareRerankers(context.Background(), cfg, func(_ context.Context, spec config.ResolvedModelBinding) (PairScorer, error) {
		return &lifecyclePairScorer{}, nil
	})
	if err != nil || len(models) != 2 || models["unused"] != nil {
		t.Fatalf("unreachable owner loaded: %+v %v", models, err)
	}
	if err = CloseRerankers(models); err != nil {
		t.Fatal(err)
	}
	for _, model := range models {
		if model.(*lifecyclePairScorer).closed != 1 {
			t.Fatal("owner leaked")
		}
	}
	cfg.Recipes[0].Profile.Decisions = nil
	cfg.Recipes[1].Profile.Decisions = nil
	if models, err = prepareRerankers(context.Background(), cfg, func(context.Context, config.ResolvedModelBinding) (PairScorer, error) {
		t.Fatal("unused model loaded")
		return nil, nil
	}); err != nil || len(models) != 0 {
		t.Fatal(err)
	}
}
