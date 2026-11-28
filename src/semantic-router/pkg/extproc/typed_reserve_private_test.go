package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type typedReserveRuntime struct{}

func (*typedReserveRuntime) Card(context.Context, string) (modelservice.ModelCard, error) {
	return modelservice.ModelCard{ID: "fixture", Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul"}, Modalities: []string{"text", "image"}}, nil
}

func (r *typedReserveRuntime) CurrentCard(name string) (modelservice.ModelCard, bool) {
	card, _ := r.Card(context.Background(), name)
	return card, true
}

func (*typedReserveRuntime) Embed(context.Context, string, modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	return modelservice.EmbedResponse{}, nil
}

func (*typedReserveRuntime) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	answers := map[string]modelservice.Answer{}
	for _, q := range request.Questions {
		answers[q.ID] = modelservice.Answer{Type: "noul", Noul: 0.9, InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: answers}, nil
}

const typedReserveConfig = `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    signals:
      systemone:
        - name: vision
          input: {image: true}
          question: {type: noul, instructions: "Is this an image?"}
    decisions:
      - name: image_route
        priority: 100
        rules:
          operator: AND
          conditions: [{type: systemone, name: vision}]
        modelRefs: [{model: general}]
global:
  model_catalog:
    system: {decision_model: {deployment: primary}}
    deployments:
      primary: {provider: model_runtime, artifact: /models/custom-image, device: cpu}
`

// The real eval selector receives a typed inline image from the service. The
// model and image inference are synthetic, and no network or GPU is used.
func TestPrivateTypedImageReserveSelectionAndRefusal(t *testing.T) {
	for _, tc := range []struct {
		name                          string
		window                        int
		output                        string
		want                          string
		fullRejected, reducedRejected bool
	}{
		{"reserve fallback", 8192, "512", services.EvalSelectionSelected, true, false},
		{"full estimate fits", 32768, "512", services.EvalSelectionSelected, false, false},
		{"true output overflow", 8192, "16384", services.EvalSelectionUnavailable, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(typedReserveConfig))
			if err != nil {
				t.Fatal(err)
			}
			model := cfg.ModelConfig["general"]
			model.Capabilities = []string{"chat", "vision"}
			model.ContextWindowSize = tc.window
			model.MaxOutputTokens = 32768
			cfg.ModelConfig["general"] = model
			cfg.CandidateRequirements = &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
			wire := fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}}]}],"max_tokens":%s}`, tc.output)
			neutral, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest([]byte(wire), llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatal(err)
			}
			demand := selection.DemandForRequest(&neutral)
			if !demand.Known || demand.InputReserveTokens < classification.RequestContextImageTokenBudget || !demand.ModelCapabilities.Supports(llmprotocol.CapabilityImageInput) || demand.MaxOutputTokens == nil {
				t.Fatalf("missing image reserve, capability or output bound: %+v", demand)
			}
			for _, check := range []struct {
				name   string
				demand selection.CandidateDemand
				reject bool
			}{
				{"full", demand, tc.fullRejected}, {"without reserves", demand.WithoutReserves(), tc.reducedRejected},
			} {
				validationErr := selection.ValidateCandidateRequirements(cfg.CandidateRequirements, "general", model, check.demand)
				var budgetErr *selection.RequestBudgetError
				if check.reject {
					if !errors.As(validationErr, &budgetErr) || budgetErr.Code != "context_length_exceeded" {
						t.Fatalf("%s: expected context overflow, got %v", check.name, validationErr)
					}
				} else if validationErr != nil {
					t.Fatalf("%s: unexpected rejection: %v", check.name, validationErr)
				}
			}
			classifiers, err := classification.BuildRecipeClassifiers(cfg, classification.RecipeRuntimeOptions{Runtime: serving.New(&typedReserveRuntime{}, nil)})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = classifiers.Close() })
			if initErr := classifiers.InitializeRuntime(); initErr != nil {
				t.Fatal(initErr)
			}
			svc := services.NewRecipeClassificationService(classifiers, cfg)
			router := routingTestRouter("general")
			router.Config = cfg
			router.ModelSelector = selection.NewRegistry()
			router.ModelSelector.Register(selection.MethodStatic, selection.NewStaticSelector(nil))
			svc.SetEvalModelSelector(router)
			uri := "data:image/png;base64,YQ=="
			result, err := svc.ClassifyIntentForEval(context.Background(), services.IntentRequest{Input: &services.ClassificationInput{Type: "image", ImageURL: &uri}, MaxTokens: json.RawMessage(tc.output)})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("selection status=%s model=%s reason=%s", result.SelectionStatus, result.SelectedModel, result.SelectionReason)
			if result.SelectionStatus != tc.want {
				t.Fatalf("status=%s reason=%s want=%s", result.SelectionStatus, result.SelectionReason, tc.want)
			}
			if tc.want == services.EvalSelectionSelected && result.SelectedModel != "general" {
				t.Fatalf("selected %q", result.SelectedModel)
			}
			legacy, err := svc.ClassifyIntentForEval(context.Background(), services.IntentRequest{
				Messages:  []services.IntentMessage{{Role: "user", Content: json.RawMessage(`[{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}}]`)}},
				MaxTokens: json.RawMessage(tc.output),
			})
			if err != nil || legacy.SelectionStatus != result.SelectionStatus || legacy.SelectedModel != result.SelectedModel {
				t.Fatalf("typed=%+v legacy=%+v err=%v", result, legacy, err)
			}
		})
	}
}
