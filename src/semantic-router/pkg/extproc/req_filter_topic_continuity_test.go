package extproc

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

func topicText(role llmprotocol.Role, value string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
}

func topicChangeRequest() *llmprotocol.Request {
	return &llmprotocol.Request{Messages: []llmprotocol.Message{
		topicText(llmprotocol.RoleUser, "Refactor the routing module so plugins load lazily"),
		topicText(llmprotocol.RoleAssistant, "Done. The loader now defers plugin initialization."),
		topicText(llmprotocol.RoleUser, "Unrelated question: how do I renew a passport?"),
	}}
}

func topicContinuityRouter(t *testing.T, rules ...config.TopicContinuityRule) *OpenAIRouter {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.TopicContinuityRules = rules
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	return &OpenAIRouter{Config: cfg, Classifier: classifier}
}

func topicContinuityContext(request *llmprotocol.Request) *RequestContext {
	ctx := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}, SemanticRequest: request}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	captureOriginalContextHistory(ctx)
	return ctx
}

func TestTopicContinuityDeclaredRulesAreEvaluatedInOrder(t *testing.T) {
	falseValue := false
	router := topicContinuityRouter(t,
		config.TopicContinuityRule{Name: "topic_boundary"},
		config.TopicContinuityRule{Name: "user_only", IncludeAssistant: &falseValue},
	)
	ctx := topicContinuityContext(topicChangeRequest())
	router.evaluateTopicContinuity(ctx)

	if len(ctx.TopicContinuityEvaluations) != 2 ||
		ctx.TopicContinuityEvaluations[0].Result.Signal != "topic_boundary" ||
		ctx.TopicContinuityEvaluations[1].Result.Signal != "user_only" {
		t.Fatalf("evaluations not in declaration order: %+v", ctx.TopicContinuityEvaluations)
	}
	result, ok := ctx.TopicContinuityResults["topic_boundary"]
	if !ok || result.Reason != topiccontinuity.ReasonExplicitChange ||
		result.HistorySource != topiccontinuity.SourceOriginalSnapshot {
		t.Fatalf("result = %+v, %v", result, ok)
	}
	if _, ok := ctx.TopicContinuityResults["undeclared"]; ok {
		t.Fatal("an undeclared rule produced a result")
	}
}

func TestTopicContinuityNoRulesNeverDecodes(t *testing.T) {
	router := topicContinuityRouter(t)
	ctx := topicContinuityContext(topicChangeRequest())
	router.evaluateTopicContinuity(ctx)
	if ctx.TopicContinuityResults != nil || ctx.originalConversationLoaded {
		t.Fatalf("no rules declared, yet results=%v decoded=%v", ctx.TopicContinuityResults, ctx.originalConversationLoaded)
	}
}

func TestTopicContinuityOnlySelectedRecipeRulesRun(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{}},
		{Name: "contextual", Profile: config.RoutingProfile{Signals: config.Signals{
			TopicContinuityRules: []config.TopicContinuityRule{{Name: "topic_boundary"}},
		}}},
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"contextual-model"}, Recipe: "contextual"}}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}

	contextual := &RequestContext{
		TraceContext: context.Background(), Headers: map[string]string{},
		SemanticRequest: topicChangeRequest(),
	}
	router.resolveEntrypointForRequest("contextual-model", contextual)
	captureOriginalContextHistory(contextual)
	router.evaluateTopicContinuity(contextual)
	if _, ok := contextual.TopicContinuityResults["topic_boundary"]; !ok {
		t.Fatal("the contextual recipe's rule did not run")
	}

	plain := &RequestContext{
		TraceContext: context.Background(), Headers: map[string]string{},
		SemanticRequest: topicChangeRequest(),
	}
	router.resolveEntrypointForRequest(config.DefaultEntrypointModel, plain)
	captureOriginalContextHistory(plain)
	router.evaluateTopicContinuity(plain)
	if plain.TopicContinuityResults != nil {
		t.Fatalf("the default recipe declares no rule, yet got %v", plain.TopicContinuityResults)
	}
}

func TestTopicContinuityReadsTheOriginalHistoryOnly(t *testing.T) {
	router := topicContinuityRouter(t, config.TopicContinuityRule{Name: "topic_boundary"})
	baseline := topicContinuityContext(topicChangeRequest())
	router.evaluateTopicContinuity(baseline)

	enriched := topicContinuityContext(topicChangeRequest())
	// Memory and RAG run after capture and change the provider-bound request.
	memory := topicText(llmprotocol.RoleUser, "Remembered: the user renews passports in Lisbon")
	enriched.SemanticRequest.Messages = append([]llmprotocol.Message{memory}, enriched.SemanticRequest.Messages...)
	router.evaluateTopicContinuity(enriched)

	want, _ := baseline.TopicContinuityResults["topic_boundary"]
	got, _ := enriched.TopicContinuityResults["topic_boundary"]
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("enrichment after capture changed the result:\n got %+v\nwant %+v", got, want)
	}
}

func TestTopicContinuityMissingSnapshotIsUnavailable(t *testing.T) {
	router := topicContinuityRouter(t, config.TopicContinuityRule{Name: "topic_boundary"})
	ctx := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	router.evaluateTopicContinuity(ctx)
	result, ok := ctx.TopicContinuityResults["topic_boundary"]
	if !ok || result.Reason != topiccontinuity.ReasonHistoryUnavailable || !result.Fallback {
		t.Fatalf("result = %+v, %v", result, ok)
	}
}

func TestOriginalConversationIsDecodedOnceAndNotMutated(t *testing.T) {
	falseValue := false
	router := topicContinuityRouter(t,
		config.TopicContinuityRule{Name: "a"},
		config.TopicContinuityRule{Name: "b", IncludeAssistant: &falseValue},
		config.TopicContinuityRule{Name: "c", Limits: &config.TopicContinuityEvidenceLimits{MaxPriorTurns: 2}},
	)
	ctx := topicContinuityContext(topicChangeRequest())
	before, ok := originalConversation(ctx)
	if !ok {
		t.Fatal("snapshot not available")
	}
	memo := ctx.originalConversation
	snapshot := contextcompression.CaptureHistory(&llmprotocol.Request{Messages: before.Messages})
	router.evaluateTopicContinuity(ctx)
	if ctx.originalConversation != memo {
		t.Fatal("the snapshot was decoded more than once")
	}
	after, _ := originalConversation(ctx)
	if !reflect.DeepEqual(snapshot.Conversation().Messages, after.Messages) {
		t.Fatal("evaluation mutated the shared decoded history")
	}
}

func TestTopicContinuityEvaluatesOncePerRequest(t *testing.T) {
	router := topicContinuityRouter(t, config.TopicContinuityRule{Name: "topic_boundary"})
	ctx := topicContinuityContext(topicChangeRequest())
	router.evaluateTopicContinuity(ctx)
	first := ctx.TopicContinuityEvaluations
	router.evaluateTopicContinuity(ctx)
	if &first[0] != &ctx.TopicContinuityEvaluations[0] {
		t.Fatal("a second call re-evaluated the request")
	}
}

func TestTopicContinuityReplayReceiptsFollowDeclarationOrder(t *testing.T) {
	router := topicContinuityRouter(t,
		config.TopicContinuityRule{Name: "z_rule"},
		config.TopicContinuityRule{Name: "a_rule"},
		config.TopicContinuityRule{Name: "m_rule"},
	)
	ctx := topicContinuityContext(topicChangeRequest())
	router.evaluateTopicContinuity(ctx)
	diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "route", 0, 0)
	if len(diagnostics.TopicContinuity) != 3 {
		t.Fatalf("receipts = %+v", diagnostics.TopicContinuity)
	}
	for i, want := range []string{"z_rule", "a_rule", "m_rule"} {
		if got := diagnostics.TopicContinuity[i].Signal; got != want {
			t.Fatalf("receipt %d = %q, want %q", i, got, want)
		}
	}
	if diagnostics.TopicContinuity[0].Reason != string(topiccontinuity.ReasonExplicitChange) {
		t.Fatalf("receipt = %+v", diagnostics.TopicContinuity[0])
	}
}
