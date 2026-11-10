package modelservice

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

func fusionClient(t *testing.T, runtime *surfaceRuntime) *Client {
	t.Helper()
	client := newSurfaceClient(t, runtime)
	if _, err := client.Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	return client
}

// classifyStage makes one classify call per text in one bundle, as a fanned-out signal does.
func classifyStage(client *Client, model string, texts []string, request func(i int) ClassifyRequest) ([]string, []error) {
	ctx, bundle := WithBundle(context.Background(), time.Second)
	leave := bundle.Join()
	labels, errs := make([]string, len(texts)), make([]error, len(texts))
	Fan(ctx, len(texts), func(i int) {
		response, err := client.Classify(ctx, model, request(i))
		errs[i] = err
		if err == nil && len(response.Results) == 1 {
			labels[i] = response.Results[0].Label
		}
	})
	leave()
	return labels, errs
}

func pieces(n int) []string {
	texts := make([]string, n)
	for i := range texts {
		texts[i] = fmt.Sprintf("piece-%d", i)
	}
	return texts
}

func TestFusionSendsAStagesClassifyCallsAsOneTask(t *testing.T) {
	runtime := &surfaceRuntime{maxTasks: DefaultBundleTasks, maxInputs: map[string]int{"echo": 2048}}
	client := fusionClient(t, runtime)
	texts := pieces(3*DefaultBundleTasks + 1)
	labels, errs := classifyStage(client, "echo", texts, func(i int) ClassifyRequest {
		return ClassifyRequest{Head: "default", Inputs: []ClassifyInput{{Text: texts[i]}}}
	})
	for i := range texts {
		if errs[i] != nil || labels[i] != texts[i] {
			t.Fatalf("call %d got %q, %v; want its own label %q", i, labels[i], errs[i], texts[i])
		}
	}
	if runtime.bundles.Load() != 1 || runtime.tasks.Load() != 1 {
		t.Fatalf("bundles=%d tasks=%d, want %d calls fused into one task", runtime.bundles.Load(), runtime.tasks.Load(), len(texts))
	}
}

func TestFusionStopsAtTheModelsInputCap(t *testing.T) {
	runtime := &surfaceRuntime{maxInputs: map[string]int{"echo": 4}}
	client := fusionClient(t, runtime)
	texts := pieces(10)
	labels, errs := classifyStage(client, "echo", texts, func(i int) ClassifyRequest {
		return ClassifyRequest{Inputs: []ClassifyInput{{Text: texts[i]}}}
	})
	for i := range texts {
		if errs[i] != nil || labels[i] != texts[i] {
			t.Fatalf("call %d got %q, %v", i, labels[i], errs[i])
		}
	}
	if runtime.tasks.Load() != 3 {
		t.Fatalf("tasks=%d, want 10 inputs in tasks of at most 4", runtime.tasks.Load())
	}
}

func TestFusionKeepsDifferentHeadsAndOptionsApart(t *testing.T) {
	runtime := &surfaceRuntime{maxInputs: map[string]int{"echo": 2048}}
	client := fusionClient(t, runtime)
	texts := pieces(6)
	_, errs := classifyStage(client, "echo", texts, func(i int) ClassifyRequest {
		request := ClassifyRequest{Head: "default", Inputs: []ClassifyInput{{Text: texts[i]}}}
		switch i % 3 {
		case 1:
			request.Head = "other"
		case 2:
			request.Overflow = "truncate"
		}
		return request
	})
	for i, err := range errs {
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}
	if runtime.tasks.Load() != 3 {
		t.Fatalf("tasks=%d, want one per head and options", runtime.tasks.Load())
	}
}

func TestAFusedTasksErrorReachesEveryCaller(t *testing.T) {
	runtime := &surfaceRuntime{maxInputs: map[string]int{"busy": 2048}}
	client := fusionClient(t, runtime)
	texts := pieces(5)
	_, errs := classifyStage(client, "busy", texts, func(i int) ClassifyRequest {
		return ClassifyRequest{Inputs: []ClassifyInput{{Text: texts[i]}}}
	})
	for i, err := range errs {
		if !errors.Is(err, ErrOverloaded) {
			t.Fatalf("call %d: got %v, want the fused task's overload", i, err)
		}
	}
	if runtime.tasks.Load() != 1 {
		t.Fatalf("tasks=%d", runtime.tasks.Load())
	}
}

func TestSplitClassifyRefusesResultsThatMissAnInput(t *testing.T) {
	label := "x"
	body := api.ClassifyResponse{Results: []api.ClassifyResult{{Index: 0, Label: &label}, {Index: 0, Label: &label}, {Index: 2, Label: &label}}}
	if parts := splitClassify(body, []int{1, 2}); parts != nil {
		t.Fatalf("a duplicated index split into %+v", parts)
	}
	body.Results[1].Index = 1
	parts := splitClassify(body, []int{1, 2})
	if len(parts) != 2 || len(parts[1].Results) != 2 || parts[1].Results[0].Index != 0 || parts[1].Results[1].Index != 1 {
		t.Fatalf("split %+v", parts)
	}
}
