package classification

import (
	"context"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type judgmentTestServices struct {
	cardServices
	mu       sync.Mutex
	requests []modelservice.Request
	scores   map[string]float64
}

func (s *judgmentTestServices) Decide(ctx context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	if err := ctx.Err(); err != nil {
		return modelservice.Response{}, err
	}
	s.mu.Lock()
	s.requests = append(s.requests, request)
	s.mu.Unlock()
	answers := map[string]modelservice.Answer{}
	for _, question := range request.Questions {
		answer := modelservice.Answer{Type: question.Type}
		if question.RequireFullInput {
			answer.InputCoverage = "complete"
		}
		switch question.Type {
		case "choice":
			answer.Choice = question.Choices[0].Key
			answer.Probabilities = map[string]float64{question.Choices[0].Key: .8, question.Choices[1].Key: .2}
			if s.scores != nil {
				answer.Probabilities = s.scores
			}
		case "score":
			answer.Score = .2
		case "noul":
			answer.Noul = .85
		}
		answers[question.ID] = answer
	}
	return modelservice.Response{Answers: answers}, nil
}

func TestDecisionPreferenceRejectsInvalidProbabilities(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), -.1, 1.1} {
		models, services := preparedJudgmentModels(t)
		rules := []config.PreferenceRule{{Name: "concise"}, {Name: "detailed"}}
		judgment, err := prepareDecisionPreference(models, rules)
		if err != nil {
			t.Fatal(err)
		}
		services.scores = map[string]float64{"concise": value, "detailed": .2}
		preference := &PreferenceClassifier{judgment: judgment, preferenceRules: rules}
		if _, err := preference.ClassifyContext(t.Context(), "input"); err == nil {
			t.Fatalf("invalid preference probability accepted: %v", value)
		}
	}
}

func preparedJudgmentModels(t *testing.T) (*classifierModelRuntime, *judgmentTestServices) {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.DecisionModel = "primary"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Artifact: "example/untrained-general-model"}}
	services := &judgmentTestServices{cardServices: cardServices{card: modelservice.ModelCard{ID: "arbitrary-family", Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice", "score", "noul"}}}}
	models, err := newClassifierModelRuntime(cfg, RecipeRuntimeOptions{Runtime: serving.New(services, nil)})
	if err != nil {
		t.Fatal(err)
	}
	return models, services
}

func TestDecisionClassifierAndPreferenceUseSharedTaskWithoutFamilyGate(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	classifier, err := prepareDecisionLabelClassifier(models, config.ClassifierSignalRule{Name: "intent", Labels: []string{"coding", "other"}, Instructions: "Find the intent"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := classifier.Classify(t.Context(), "explain this code")
	if err != nil || result.Scores["coding"] != .8 {
		t.Fatalf("%+v %v", result, err)
	}
	rules := []config.PreferenceRule{{Name: "concise", Threshold: .7}, {Name: "detailed"}}
	models.cfg.PreferenceRules = rules
	owner := &Classifier{Config: models.cfg, models: models}
	if err = owner.initializePreferenceClassifier(); err != nil || owner.preferenceClassifier == nil || owner.preferenceClassifier.judgment == nil {
		t.Fatalf("default preference failed to prepare judgment: %v", err)
	}
	judgment, err := prepareDecisionPreference(models, rules)
	if err != nil {
		t.Fatal(err)
	}
	preference := &PreferenceClassifier{judgment: judgment, preferenceRules: rules}
	got, err := preference.ClassifyContext(t.Context(), "Please be brief")
	if err != nil || got.Preference != "concise" {
		t.Fatalf("%+v %v", got, err)
	}
	if services.requests[1].State != "Please be brief" {
		t.Fatal("input changed")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := preference.ClassifyContext(ctx, "ignored"); err == nil {
		t.Fatal("request cancellation ignored")
	}
}

func TestDecisionSafetyComposesIndependentHazardsWithoutNormalizing(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	classifier, err := prepareDecisionSafety(models, "safety.risk.hazard", []string{"violence", "fraud"}, true)
	if err != nil {
		t.Fatal(err)
	}
	result, err := classifier.Classify(t.Context(), "input")
	if err != nil || result.Scores["violence"] != .85 || result.Scores["fraud"] != .85 {
		t.Fatalf("%+v %v", result, err)
	}
	if len(services.requests) != 1 || len(services.requests[0].Questions) != 2 {
		t.Fatal("composed set did not batch its questions")
	}
	if services.requests[0].Questions[0].Truncate {
		t.Fatal("safety judgment may not silently truncate")
	}
}

func TestDecisionComplexityPreservesThresholdScaleWithoutEmbedding(t *testing.T) {
	models, _ := preparedJudgmentModels(t)
	hard, easy := .7, .3
	rules := []config.ComplexityRule{{Name: "symmetric", Threshold: .2}, {Name: "native", HardAbove: &hard, EasyBelow: &easy}}
	classifier, err := prepareDecisionComplexity(models, rules)
	if err != nil {
		t.Fatal(err)
	}
	results, err := classifier.classifyDetailedWithImageCached(t.Context(), "short request", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if results[0].FusedMargin != -.8 || results[1].FusedMargin != .1 {
		t.Fatal(results)
	}
	for _, result := range results {
		if result.Difficulty != "easy" || result.ConfidenceReported {
			t.Fatal(result)
		}
	}
}

func TestDecisionSafetyPreservesPublishedBinaryQuestion(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	classifier, err := prepareDecisionSafety(models, "safety.unsafe", []string{"safe", "unsafe"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = classifier.Classify(t.Context(), "input"); err != nil {
		t.Fatal(err)
	}
	want := modelservice.Question{
		ID: "safety.unsafe:p_harm", Type: "choice", Instructions: "Is this request harmful?",
		RequireFullInput: true,
		Choices: []modelservice.Choice{
			{Key: "safe", Description: "a benign request that does not violate any safety policy"},
			{Key: "unsafe", Description: "a request that violates a safety policy or seeks harmful assistance"},
		},
	}
	if len(services.requests) != 1 || !reflect.DeepEqual(services.requests[0].Questions, []modelservice.Question{want}) {
		t.Fatalf("published safety question changed: %+v", services.requests)
	}
	if classifier.(*decisionLabelClassifier).judgment.plan.Definition.ID != "safety" {
		t.Fatal("wire question identity must not replace the shared semantic task identity")
	}
}

func TestDecisionSafetyRetainsCustomLabels(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	classifier, err := prepareDecisionSafety(models, "safety.policy", []string{"allowed", "restricted"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = classifier.Classify(t.Context(), "input"); err != nil {
		t.Fatal(err)
	}
	question := services.requests[0].Questions[0]
	if question.ID != "safety.policy:safety" || !reflect.DeepEqual(question.Choices, []modelservice.Choice{{Key: "allowed", Description: "allowed"}, {Key: "restricted", Description: "restricted"}}) {
		t.Fatalf("custom safety question changed: %+v", question)
	}
}
