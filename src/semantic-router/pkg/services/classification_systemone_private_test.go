package services

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type typedImageRuntime struct {
	mu          sync.Mutex
	requests    []modelservice.Request
	failedImage string
	score       float64
	entered     chan struct{}
	modalities  []string
	failedState string
	delays      map[string]time.Duration
}

func (s *typedImageRuntime) Card(context.Context, string) (modelservice.ModelCard, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	modalities := append([]string(nil), s.modalities...)
	if modalities == nil {
		modalities = []string{"text", "image"}
	}
	return modelservice.ModelCard{ID: "fixture", Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul"}, Modalities: modalities}, nil
}

func (s *typedImageRuntime) CurrentCard(name string) (modelservice.ModelCard, bool) {
	c, _ := s.Card(context.Background(), name)
	return c, true
}

func (*typedImageRuntime) Embed(context.Context, string, modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	return modelservice.EmbedResponse{}, nil
}

func (s *typedImageRuntime) Decide(ctx context.Context, _ string, req modelservice.Request) (modelservice.Response, error) {
	s.mu.Lock()
	s.requests = append(s.requests, req)
	failure := (len(req.Images) > 0 && req.Images[0] == s.failedImage) || (s.failedState != "" && req.State == s.failedState)
	var delay time.Duration
	if len(req.Images) > 0 {
		delay = s.delays[req.Images[0]]
	}
	score := s.score
	entered := s.entered
	s.mu.Unlock()
	if entered != nil {
		entered <- struct{}{}
		<-ctx.Done()
		return modelservice.Response{}, ctx.Err()
	}
	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return modelservice.Response{}, ctx.Err()
		}
	}
	if failure {
		return modelservice.Response{}, modelservice.ErrUnavailable
	}
	ans := map[string]modelservice.Answer{}
	for _, q := range req.Questions {
		ans[q.ID] = modelservice.Answer{Type: "noul", Noul: score, InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: ans}, nil
}

func (s *typedImageRuntime) snapshot() []modelservice.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]modelservice.Request(nil), s.requests...)
}

const systemoneTypedImageConfig = `version: v0.5
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

func typedImageService(t *testing.T) (*ClassificationService, *typedImageRuntime) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(systemoneTypedImageConfig))
	if err != nil {
		t.Fatal(err)
	}
	fake := &typedImageRuntime{score: 0.9}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, classification.RecipeRuntimeOptions{Runtime: serving.New(fake, nil)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	if err = classifiers.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	return NewRecipeClassificationService(classifiers, cfg), fake
}

func TestPrivateTypedImageReachesSystemOne(t *testing.T) {
	svc, fake := typedImageService(t)
	const data = "data:image/png;base64,YQ=="
	var input ClassificationInput
	if err := json.Unmarshal([]byte(`{"type":"image","image_url":"`+data+`"}`), &input); err != nil {
		t.Fatal(err)
	}
	response, err := svc.ClassifyIntent(context.Background(), IntentRequest{Input: &input})
	if err != nil {
		t.Fatalf("single typed image: %v", err)
	}
	if response.Classification.Category != "image_route" {
		t.Fatalf("image did not route: %+v", response)
	}
	reqs := fake.snapshot()
	if len(reqs) != 1 || len(reqs[0].Images) != 1 || reqs[0].Images[0] != data {
		t.Fatalf("image did not reach model: %+v", reqs)
	}
}

func TestPrivateTypedImageBatchKeepsOrderedSiblings(t *testing.T) {
	svc, fake := typedImageService(t)
	fake.failedImage = "data:image/png;base64,Yg=="
	var inputs []ClassificationInput
	if err := json.Unmarshal([]byte(`[{"type":"image","image_url":"data:image/png;base64,YQ=="},{"type":"image","image_url":"data:image/png;base64,Yg=="},{"type":"image","image_url":"data:image/png;base64,Yw=="},{"type":"image","image_url":"https://example.invalid/image"}]`), &inputs); err != nil {
		t.Fatal(err)
	}
	result, err := svc.ClassifyInputsContext(context.Background(), inputs, "intent", nil)
	if err != nil || len(result.Results) != 4 {
		t.Fatalf("batch: %+v %v", result, err)
	}
	for _, index := range []int{0, 2} {
		if result.Results[index].Error != nil || result.Results[index].Intent == nil || result.Results[index].Intent.Classification.Category != "image_route" {
			t.Fatalf("sibling %d: %+v", index, result.Results[index])
		}
	}
	if result.Results[1].Error == nil || result.Results[1].Intent != nil {
		t.Fatalf("failed item lost: %+v", result.Results[1])
	}
	if result.Results[3].Error == nil || result.Results[3].Error.Code != "INVALID_INPUT" {
		t.Fatalf("URL not rejected: %+v", result.Results[3])
	}
	if got := len(fake.snapshot()); got != 3 {
		t.Fatalf("expected three model calls, got %d", got)
	}
}

func TestPrivateTypedImageWithoutMatchingDecisionDoesNotAskTextFallback(t *testing.T) {
	svc, fake := typedImageService(t)
	fake.score = 0.1
	const data = "data:image/png;base64,YQ=="
	var input ClassificationInput
	if err := json.Unmarshal([]byte(`{"type":"image","image_url":"`+data+`"}`), &input); err != nil {
		t.Fatal(err)
	}
	single, err := svc.ClassifyIntent(context.Background(), IntentRequest{Input: &input})
	if err != nil || single.Classification.Category != "other" {
		t.Fatalf("single: %+v %v", single, err)
	}
	batch, err := svc.ClassifyInputsContext(context.Background(), []ClassificationInput{input}, "intent", nil)
	if err != nil || batch.Results[0].Error != nil || batch.Results[0].Intent.Classification.Category != "other" {
		t.Fatalf("batch: %+v %v", batch, err)
	}
	requests := fake.snapshot()
	if len(requests) != 2 {
		t.Fatalf("unexpected text-only fallback: %+v", requests)
	}
	for _, request := range requests {
		if len(request.Images) != 1 || request.Images[0] != data {
			t.Fatalf("missing image: %+v", request)
		}
	}
}

func TestPrivateTypedImageCancellationPropagates(t *testing.T) {
	svc, fake := typedImageService(t)
	fake.entered = make(chan struct{}, 1)
	var input ClassificationInput
	if err := json.Unmarshal([]byte(`{"type":"image","image_url":"data:image/png;base64,YQ=="}`), &input); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan *ClassificationInputsResponse, 1)
	go func() {
		result, _ := svc.ClassifyInputsContext(ctx, []ClassificationInput{input}, "intent", nil)
		done <- result
	}()
	select {
	case <-fake.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("model was not called")
	}
	cancel()
	select {
	case result := <-done:
		if result == nil || result.Results[0].Error == nil || result.Results[0].Error.Code != "CANCELED" {
			t.Fatalf("cancellation lost: %+v", result)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("batch did not stop")
	}
}

func TestPrivateTypedImagePreflightStopsBeforeInference(t *testing.T) {
	source := strings.Replace(systemoneTypedImageConfig, "artifact: /models/custom-image, device: cpu", "endpoint: http://runtime.invalid", 1)
	cfg, err := config.ParseYAMLBytes([]byte(source))
	if err != nil {
		t.Fatal(err)
	}
	fake := &typedImageRuntime{score: 0.9}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, classification.RecipeRuntimeOptions{Runtime: serving.New(fake, nil)})
	if err != nil {
		t.Fatal(err)
	}
	defer classifiers.Close()
	if initErr := classifiers.InitializeRuntime(); initErr != nil {
		t.Fatal(initErr)
	}
	fake.mu.Lock()
	fake.modalities = []string{"text"}
	fake.mu.Unlock()
	service := NewRecipeClassificationService(classifiers, cfg)
	var input ClassificationInput
	if decodeErr := json.Unmarshal([]byte(`{"type":"image","image_url":"data:image/png;base64,YQ=="}`), &input); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	_, err = service.ClassifyIntent(context.Background(), IntentRequest{Input: &input})
	if !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("expected unsupported input: %v", err)
	}
	batch, err := service.ClassifyInputsContext(context.Background(), []ClassificationInput{input}, "intent", nil)
	if err != nil || batch.Results[0].Error == nil || batch.Results[0].Error.Code != "UNSUPPORTED_INPUT" {
		t.Fatalf("batch expected unsupported input: %+v %v", batch, err)
	}
	if len(fake.snapshot()) != 0 {
		t.Fatal("unsupported model received inference")
	}
}

func TestPrivateLegacyEmptyTextStillAsksFallback(t *testing.T) {
	cfg := &config.RouterConfig{}
	classifier, err := classification.BuildClassifier(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer classifier.Close()
	fake := &typedImageRuntime{score: 0.9}
	classifier.SetDecisionDecider(fake)
	_, _ = resolveIntentCategoryWithError(context.Background(), classifier, nil, nil, "", false)
	if len(fake.snapshot()) != 1 {
		t.Fatalf("legacy empty-text fallback was not asked: %+v", fake.snapshot())
	}
}
