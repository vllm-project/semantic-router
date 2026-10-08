package classification

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestComplexityConstructionKeepsPrototypeMarginsWithDefaultDecisionModel(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	hard, easy := .025, -.08
	models.cfg.ComplexityRules = []config.ComplexityRule{{Name: "difficulty", HardAbove: &hard, EasyBelow: &easy,
		Hard: config.ComplexityCandidates{Candidates: []string{"hard example"}},
		Easy: config.ComplexityCandidates{Candidates: []string{"easy example"}},
	}}
	classifier := buildComplexityExecutionTest(t, models)
	for _, test := range []struct{ query, want string }{{"hard example", "hard"}, {"easy example", "easy"}, {"balanced query", "medium"}} {
		results, err := classifier.classifyDetailedWithImageCached(t.Context(), test.query, "", nil)
		if err != nil || len(results) != 1 || results[0].Difficulty != test.want || results[0].SignalSource != "text" || !results[0].ConfidenceReported {
			t.Fatalf("%s: results=%+v err=%v", test.query, results, err)
		}
	}
	if len(services.requests) != 0 {
		t.Fatal("authored prototype questions were sent to the default decision model")
	}
}

func TestComplexityConstructionCombinesPrototypeAndGenericJudgmentRules(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	hard, easy := .7, .3
	models.cfg.ComplexityRules = []config.ComplexityRule{
		{Name: "prototypes", Threshold: .1, Hard: config.ComplexityCandidates{Candidates: []string{"hard example"}}, Easy: config.ComplexityCandidates{Candidates: []string{"easy example"}}},
		{Name: "judgment", HardAbove: &hard, EasyBelow: &easy},
	}
	classifier := buildComplexityExecutionTest(t, models)
	results, err := classifier.classifyDetailedWithImageCached(t.Context(), "balanced query", "", nil)
	if err != nil || len(results) != 2 {
		t.Fatalf("results=%+v err=%v", results, err)
	}
	if results[0].RuleName != "prototypes" || results[0].Difficulty != "medium" || results[0].SignalSource != "text" || results[1].RuleName != "judgment" || results[1].Difficulty != "easy" || results[1].SignalSource != "decision_score" {
		t.Fatalf("rule sources/order changed: %+v", results)
	}
	if len(services.requests) != 1 || len(services.requests[0].Questions) != 1 {
		t.Fatal("mixed rules did not keep separate execution paths")
	}
	definition, _ := modelservice.BuiltinTask("complexity")
	question := services.requests[0].Questions[0]
	if question.Instructions != definition.Question.Instructions || !reflect.DeepEqual(question.Levels, definition.Question.Levels) {
		t.Fatalf("generic complexity changed its canonical question: %+v", question)
	}
}

func TestComplexityExplicitDecisionBindingUsesNativeNormalizedScore(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	hard, easy := .7, .3
	models.cfg.ComplexityRules = []config.ComplexityRule{{Name: "difficulty", HardAbove: &hard, EasyBelow: &easy,
		Hard: config.ComplexityCandidates{Candidates: []string{"hard example"}},
		Easy: config.ComplexityCandidates{Candidates: []string{"easy example"}},
	}}
	models.cfg.ModelBindings = map[string]config.ModelBinding{"complexity": {Deployment: "primary", Contract: config.DecisionTaskContract}}
	var err error
	models, err = newClassifierModelRuntime(models.cfg, RecipeRuntimeOptions{Runtime: serving.New(services, nil)})
	if err != nil {
		t.Fatal(err)
	}
	builder := newClassifierOptionBuilder(models.cfg, nil)
	builder.models = models
	option, err := builder.buildComplexityClassifierOption()
	if err != nil {
		t.Fatal(err)
	}
	owner := &Classifier{}
	option(owner)
	results, err := owner.complexityClassifier.classifyDetailedWithImageCached(t.Context(), "hard example", "", nil)
	if err != nil || len(results) != 1 || results[0].FusedMargin != .1 || results[0].Difficulty != "easy" || results[0].SignalSource != "decision_score" {
		t.Fatalf("native score or units changed: %+v %v", results, err)
	}
	if builder.embeddingSet != nil || len(services.requests) != 1 {
		t.Fatal("explicit judgment prepared unused prototype embeddings")
	}
}

func buildComplexityExecutionTest(t *testing.T, models *classifierModelRuntime) *ComplexityClassifier {
	t.Helper()
	builder := newClassifierOptionBuilder(models.cfg, nil)
	builder.models = models
	builder.provider = stubEmbeddingLookup(t, map[string][]float32{
		"hard example": {1, 0}, "easy example": {0, 1}, "balanced query": {1, 1},
	})
	// The fixture owns its tiny embedding provider; construction and scoring
	// otherwise follow the production classifier option path.
	builder.providerInitOnce.Do(func() {})
	option, err := builder.buildComplexityClassifierOption()
	if err != nil {
		t.Fatal(err)
	}
	owner := &Classifier{}
	option(owner)
	return owner.complexityClassifier
}
