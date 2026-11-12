package services

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func typedBatchInputs(t *testing.T, body string) []ClassificationInput {
	t.Helper()
	var inputs []ClassificationInput
	if err := json.Unmarshal([]byte(body), &inputs); err != nil {
		t.Fatal(err)
	}
	return inputs
}

func TestTypedBatchImageFailureKeepsOrderedSiblings(t *testing.T) {
	images := []string{"data:image/png;base64,YQ==", "data:image/png;base64,Yg==", "data:image/png;base64,Yw=="}
	inputs := typedBatchInputs(t, `[{"type":"image","image_url":"data:image/png;base64,YQ=="},{"type":"image","image_url":"data:image/png;base64,Yg=="},{"type":"image","image_url":"data:image/png;base64,Yw=="}]`)
	for failed := range images {
		t.Run(images[failed], func(t *testing.T) {
			svc, fake := typedImageService(t)
			fake.failedImage = images[failed]
			fake.delays = map[string]time.Duration{images[0]: 15 * time.Millisecond, images[1]: 10 * time.Millisecond, images[2]: 5 * time.Millisecond}
			result, err := svc.ClassifyInputsContext(context.Background(), inputs, "intent", nil)
			if err != nil || len(result.Results) != len(inputs) {
				t.Fatalf("batch: %+v %v", result, err)
			}
			for i, item := range result.Results {
				if i == failed {
					if item.Error == nil || item.Error.Code != "CLASSIFICATION_ERROR" || item.Intent != nil {
						t.Fatalf("missing failure: %+v", item)
					}
					continue
				}
				if item.Error != nil || item.Intent == nil || item.Intent.Classification.Category != "image_route" {
					t.Fatalf("sibling %d: %+v", i, item)
				}
			}
		})
	}
}

func TestTypedBatchMixedInputsAndPreflight(t *testing.T) {
	svc, fake := typedImageService(t)
	inputs := typedBatchInputs(t, `[{"type":"text","text":"hello"},{"type":"image","image_url":"data:image/png;base64,YQ=="},{"type":"image","image_url":"https://example.com/no-fetch"},null,{"type":"audio","text":"no"}]`)
	legacy, err := svc.ClassifyIntent(context.Background(), IntentRequest{Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	before := len(fake.snapshot())
	result, err := svc.ClassifyInputsContext(context.Background(), inputs, "intent", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Results[0].Error != nil || result.Results[0].Intent.Classification.Category != legacy.Classification.Category {
		t.Fatalf("legacy text parity: %+v", result.Results[0])
	}
	if result.Results[1].Error != nil || result.Results[1].Intent.Classification.Category != "image_route" {
		t.Fatalf("image: %+v", result.Results[1])
	}
	for i := 2; i < len(inputs); i++ {
		if result.Results[i].Error == nil || result.Results[i].Error.Code != "INVALID_INPUT" {
			t.Fatalf("invalid %d: %+v", i, result.Results[i])
		}
	}
	requests := fake.snapshot()
	if len(requests)-before != 2 {
		t.Fatalf("invalid items reached model: %+v", requests)
	}
	foundText, foundImage := false, false
	for _, request := range requests[before:] {
		if request.State == "hello" && len(request.Images) == 0 {
			foundText = true
		}
		if len(request.Images) == 1 && request.Images[0] == "data:image/png;base64,YQ==" {
			foundImage = true
		}
	}
	if !foundText || !foundImage {
		t.Fatalf("mixed requests lost their modality: %+v", requests[before:])
	}
	for _, task := range []string{"pii", "security", "all"} {
		before = len(fake.snapshot())
		out, e := svc.ClassifyInputsContext(context.Background(), inputs[1:2], task, nil)
		if e != nil || out.Results[0].Error == nil || out.Results[0].Error.Code != "UNSUPPORTED_INPUT" || len(fake.snapshot()) != before {
			t.Fatalf("task %s: %+v %v", task, out, e)
		}
	}
}

func TestTypedBatchCancellationAndRuntimeReplacement(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		t.Run(map[bool]string{false: "cancel", true: "deadline"}[deadline], func(t *testing.T) {
			svc, fake := typedImageService(t)
			fake.entered = make(chan struct{}, 1)
			ctx, cancel := context.WithCancel(context.Background())
			if deadline {
				ctx, cancel = context.WithTimeout(context.Background(), 500*time.Millisecond)
			}
			defer cancel()
			done := make(chan *ClassificationInputsResponse, 1)
			inputs := typedBatchInputs(t, `[{"type":"image","image_url":"data:image/png;base64,YQ=="}]`)
			go func() { result, _ := svc.ClassifyInputsContext(ctx, inputs, "intent", nil); done <- result }()
			select {
			case <-fake.entered:
			case <-time.After(3 * time.Second):
				t.Fatal("model did not start")
			}
			replaced := make(chan error, 1)
			go func() { replaced <- svc.TryRefreshRuntimeConfig(&config.RouterConfig{}) }()
			select {
			case e := <-replaced:
				t.Fatalf("replacement retired in-flight batch: %v", e)
			case <-time.After(20 * time.Millisecond):
			}
			if !deadline {
				cancel()
			}
			select {
			case result := <-done:
				want := "CANCELED"
				if deadline {
					want = "DEADLINE_EXCEEDED"
				}
				if result == nil || result.Results[0].Error == nil || result.Results[0].Error.Code != want {
					t.Fatalf("got %+v want %s", result, want)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("batch did not drain")
			}
			select {
			case err := <-replaced:
				if err != nil {
					t.Fatalf("replacement failed after drain: %v", err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("replacement remained blocked")
			}
		})
	}
}

func fallbackService(t *testing.T, fake *typedImageRuntime) *ClassificationService {
	t.Helper()
	cfg := &config.RouterConfig{}
	classifier, err := classification.BuildClassifier(cfg)
	if err != nil {
		t.Fatal(err)
	}
	classifier.SetDecisionDecider(fake)
	t.Cleanup(func() { _ = classifier.Close() })
	return NewClassificationService(classifier, cfg)
}

func TestTypedBatchReportsCategoryFallbackFailure(t *testing.T) {
	fake := &typedImageRuntime{score: 0.9, failedState: "fallback"}
	svc := fallbackService(t, fake)
	legacy, err := svc.ClassifyIntent(context.Background(), IntentRequest{Text: "fallback"})
	if err != nil || legacy.Classification.Category != "other" {
		t.Fatalf("legacy fallback: %+v %v", legacy, err)
	}
	result, err := svc.ClassifyInputsContext(context.Background(), typedBatchInputs(t, `[{"type":"text","text":"fallback"}]`), "intent", nil)
	if err != nil || result.Results[0].Error == nil || result.Results[0].Intent != nil {
		t.Fatalf("batch fallback failure: %+v %v", result, err)
	}
}

func TestTypedBatchCategoryFallbackPreservesCancellation(t *testing.T) {
	fake := &typedImageRuntime{score: 0.9, entered: make(chan struct{}, 1)}
	svc := fallbackService(t, fake)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan *ClassificationInputsResponse, 1)
	inputs := typedBatchInputs(t, `[{"type":"text","text":"wait"}]`)
	go func() { result, _ := svc.ClassifyInputsContext(ctx, inputs, "intent", nil); done <- result }()
	select {
	case <-fake.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("fallback did not start")
	}
	cancel()
	select {
	case result := <-done:
		if result == nil || result.Results[0].Error == nil || result.Results[0].Error.Code != "CANCELED" {
			t.Fatalf("cancellation: %+v", result)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("fallback did not stop")
	}
}
