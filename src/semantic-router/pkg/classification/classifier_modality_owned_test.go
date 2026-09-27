package classification

import (
	"context"
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func modalityPreparationConfig(recipe config.RecipeName, method string) *config.RouterConfig {
	cfg := &config.RouterConfig{RoutingScope: recipe}
	cfg.ModalityDetector = config.ModalityDetectorConfig{
		Enabled: true,
		ModalityDetectionConfig: config.ModalityDetectionConfig{
			Method: method, ConfidenceThreshold: 0.6, LowerThresholdRatio: 0.7,
			Keywords:   []string{"make a poster"},
			Classifier: &config.ModalityClassifierConfig{ModelPath: "unused-inherited-model", MaxSequenceLength: 32768},
		},
	}
	return cfg
}

func modalityPreparationModels(t *testing.T, cfg *config.RouterConfig, explicit bool) *classifierModelRuntime {
	t.Helper()
	if explicit {
		cfg.ModelDeployments = map[string]config.ModelDeployment{
			"selected": {Artifact: "selected-model", Provider: "ort", Device: "rocm:0", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 8192, Overflow: "reject"}},
		}
		cfg.ModelBindings = map[string]config.ModelBinding{
			"modality_detector": {Deployment: "selected", Adapter: "mmbert32k", Contract: config.RemoteClassifierContractLabelDistribution},
		}
	}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &classifierModelRuntime{cfg: cfg, plan: plan, recipe: cfg.RoutingScope}
}

func TestOwnedModalitySkipsRecipesWithoutGenerationIntentRules(t *testing.T) {
	for _, recipe := range []config.RecipeName{"images-only", config.DefaultRecipeName} {
		for _, explicit := range []bool{false, true} {
			cfg := modalityPreparationConfig(recipe, config.ModalityDetectionClassifier)
			cfg.ConversationRules = []config.ConversationRule{{Name: "images", Feature: config.ConversationFeature{Type: "exists", Source: config.ConversationSource{Type: "image_content"}}}}
			cfg.InputModalityRules = []config.InputModalityRule{{Name: "image-input", Modality: config.InputModalityImage}}
			models := modalityPreparationModels(t, cfg, explicit)
			calls := 0
			apply, err := buildOwnedModalityOption(cfg, models, func(context.Context, config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelDistribution], error) {
				calls++
				return nil, errors.New("native loader must not run")
			})
			if err != nil || apply != nil || calls != 0 {
				t.Fatalf("recipe %q explicit=%v loaded unused modality: calls=%d option=%v err=%v", recipe, explicit, calls, apply != nil, err)
			}
			// The real builder also succeeds with no runtime to call. Catalog
			// availability alone must never create a native owner.
			builder := &classifierOptionBuilder{cfg: cfg, models: models}
			if apply, err = builder.buildModalityClassifierOption(); err != nil || apply != nil {
				t.Fatalf("real builder prepared unused modality: %v", err)
			}
		}
	}
}

type modalityTestResource struct{ closes int }

func (r *modalityTestResource) Close() error { r.closes++; return nil }

func preparedModalityHandle(t *testing.T, spec config.ResolvedModelBinding, resource *modalityTestResource, calls *int) *binding.Resolved[string, tasks.LabelDistribution] {
	t.Helper()
	owned, err := binding.NewPool().Acquire(context.Background(), binding.ResourceIdentity{Artifact: "synthetic", Provider: "test", Device: "cpu", Precision: "native"}, "", nil, func(context.Context) (io.Closer, error) { return resource, nil })
	if err != nil {
		t.Fatal(err)
	}
	task, err := binding.Register(binding.NewRegistry(), spec.Binding.Contract, func(string) error { return nil }, func(string, tasks.LabelDistribution) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	handle, err := task.Resolve(
		binding.Identity{Recipe: string(spec.Recipe), Name: spec.Name, Deployment: spec.Binding.Deployment, Adapter: spec.Binding.Adapter, Contract: spec.Binding.Contract},
		binding.Capability{Contract: spec.Binding.Contract, Provider: "test", Device: "cpu", Precision: "native", Labels: []string{"AR", "DIFFUSION", "BOTH"}},
		owned, func(context.Context, io.Closer, string) (tasks.LabelDistribution, error) {
			*calls++
			return tasks.LabelDistribution{Probabilities: []float32{0.1, 0.8, 0.1}}, nil
		})
	if err != nil {
		_ = owned.Close()
		t.Fatal(err)
	}
	return handle
}

func TestOwnedModalityRulesPrepareAndEvaluateForNamedAndDefaultConsumers(t *testing.T) {
	for _, recipe := range []config.RecipeName{"generate", config.DefaultRecipeName} {
		cfg := modalityPreparationConfig(recipe, config.ModalityDetectionClassifier)
		cfg.ModalityRules = []config.ModalityRule{{Name: "DIFFUSION"}}
		models := modalityPreparationModels(t, cfg, true)
		loads, calls := 0, 0
		resource := &modalityTestResource{}
		apply, err := buildOwnedModalityOption(cfg, models, func(_ context.Context, spec config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelDistribution], error) {
			loads++
			if spec.Recipe != recipe || spec.Deployment.Artifact != "selected-model" || spec.Deployment.Input.MaxTokens != 8192 {
				t.Fatalf("consumer lost explicit deployment: %+v", spec)
			}
			return preparedModalityHandle(t, spec, resource, &calls), nil
		})
		if err != nil || apply == nil || loads != 1 {
			t.Fatalf("modality consumer was not prepared: loads=%d err=%v", loads, err)
		}
		classifier := &Classifier{Config: cfg}
		apply(classifier)
		if !classifier.signalReadiness()[config.SignalTypeModality] {
			t.Fatal("prepared modality consumer is not ready for routing/classify API")
		}
		results := newMetricScopeResults()
		var mu sync.Mutex
		classifier.evaluateModalitySignal(context.Background(), results, &mu, "make a poster")
		if calls != 1 || !reflect.DeepEqual(results.MatchedModalityRules, []string{"DIFFUSION"}) || len(results.SignalErrors) != 0 {
			t.Fatalf("prepared consumer did not evaluate: calls=%d matches=%v errors=%v", calls, results.MatchedModalityRules, results.SignalErrors)
		}
		if err := classifier.modalityInference.Close(); err != nil || resource.closes != 1 {
			t.Fatalf("consumer did not release its model: closes=%d err=%v", resource.closes, err)
		}
	}
}

func TestOwnedModalityPreparationFailureKeepsConfiguredFallbackContract(t *testing.T) {
	failed := errors.New("synthetic native preparation failure")
	for _, method := range []string{config.ModalityDetectionClassifier, config.ModalityDetectionHybrid} {
		for _, explicit := range []bool{false, true} {
			cfg := modalityPreparationConfig("generate", method)
			cfg.ModalityRules = []config.ModalityRule{{Name: "DIFFUSION"}}
			models := modalityPreparationModels(t, cfg, explicit)
			loads := 0
			apply, err := buildOwnedModalityOption(cfg, models, func(context.Context, config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelDistribution], error) {
				loads++
				return nil, failed
			})
			fallback := method == config.ModalityDetectionHybrid && !explicit
			if loads != 1 || apply != nil || (fallback && err != nil) || (!fallback && !errors.Is(err, failed)) {
				t.Fatalf("method=%s explicit=%v lost preparation failure/fallback: loads=%d err=%v", method, explicit, loads, err)
			}
		}
	}
}

func TestOwnedModalityKeywordRulesDoNotLoadNativeModel(t *testing.T) {
	cfg := modalityPreparationConfig("generate", config.ModalityDetectionKeyword)
	cfg.ModalityRules = []config.ModalityRule{{Name: "DIFFUSION"}}
	models := modalityPreparationModels(t, cfg, true)
	apply, err := buildOwnedModalityOption(cfg, models, func(context.Context, config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelDistribution], error) {
		t.Fatal("keyword method invoked native loader")
		return nil, nil
	})
	if err != nil || apply != nil {
		t.Fatalf("keyword method prepared model: %v", err)
	}
	classifier := &Classifier{Config: cfg}
	results := newMetricScopeResults()
	var mu sync.Mutex
	classifier.evaluateModalitySignal(context.Background(), results, &mu, "make a poster")
	if !reflect.DeepEqual(results.MatchedModalityRules, []string{"DIFFUSION"}) || results.Metrics.Modality.Method != "keyword" {
		t.Fatalf("keyword evaluation changed: %+v", results.Metrics.Modality)
	}
}
