//go:build !windows

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Exercise the real runtime client/bundle/bindings for typed text tasks. The
// image error must not drop either text sibling or dispatch an image to a head.
func TestTypedTextTasksPreserveLegacyProjectionAndRuntimeBundling(t *testing.T) {
	runtime, fake, requests := typedTextRuntime(t)
	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.Threshold = .5
	cfg.Categories = []config.Category{{CategoryMetadata: config.CategoryMetadata{Name: "math"}}, {CategoryMetadata: config.CategoryMetadata{Name: "law"}}}
	cfg.PIIRules = []config.PIIRule{{Name: "personal_data", Threshold: .5}}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "attack", Threshold: .5}}
	cfg.Decisions = []config.Decision{{Name: "heads", Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
		{Type: config.SignalTypeDomain, Name: "math"},
		{Type: config.SignalTypePII, Name: "personal_data"},
		{Type: config.SignalTypeJailbreak, Name: "attack"},
	}}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"batch-heads": {Provider: config.ModelRuntimeProvider, Endpoint: "http://fake"}}
	cfg.ModelBindings = map[string]config.ModelBinding{
		"domain_classifier": {Deployment: "batch-heads", Head: "domain", Contract: config.RemoteClassifierContractLabelDistribution, Adapter: "modernbert"},
		"pii_classifier":    {Deployment: "batch-heads", Head: "pii", Contract: config.RemoteClassifierContractTokenSpans, Adapter: "modernbert"},
		"prompt_guard":      {Deployment: "batch-heads", Head: "guard", Contract: config.RemoteClassifierContractLabelDistribution, Adapter: "modernbert"},
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	if err := classifiers.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	service := services.NewRecipeClassificationService(classifiers, cfg)
	if !service.HasUnifiedClassifier() {
		t.Fatal("fixture did not prepare all three text tasks")
	}
	api := &ClassificationAPIServer{classificationSvc: service, config: cfg}
	classify := func(t *testing.T, body string) BatchClassificationResponse {
		t.Helper()
		recorder := httptest.NewRecorder()
		api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		if recorder.Code != http.StatusOK {
			t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
		}
		var response BatchClassificationResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		return response
	}
	for _, task := range []string{"pii", "security", "all"} {
		t.Run(task, func(t *testing.T) {
			// Measure fresh typed inputs before the legacy comparison warms the
			// deployment cache. Each task gets distinct (equivalent) queries.
			first, last := "math person "+task, "law benign "+task
			before, beforeTasks := fake.Bundles()
			beforeDirect, beforeRequests := fake.Calls("classify"), len(requests())
			typed := classify(t, fmt.Sprintf(`{"inputs":[{"type":"text","text":%q},{"type":"image","image_url":"data:image/png;base64,iVBORw0KGgo="},{"type":"text","text":%q}],"task_type":%q}`, first, last, task))
			after, afterTasks := fake.Bundles()
			// The existing runtime timer may split physical sends under load.
			// Require bundled transport and every logical head/input exactly once.
			if after-before < 1 || after-before > 6 || afterTasks-beforeTasks < 3 || afterTasks-beforeTasks > 6 || fake.Calls("classify") != beforeDirect {
				t.Fatalf("invalid bundled dispatch: bundles=%d tasks=%d results=%+v", after-before, afterTasks-beforeTasks, typed.Results)
			}
			observed := map[string]int{}
			for _, request := range requests()[beforeRequests:] {
				for _, task := range request.Tasks {
					if task.Classify == nil || task.Classify.Head == nil {
						t.Fatalf("unexpected non-classify or unnamed task: %+v", task)
					}
					for _, text := range typedRuntimeClassifyTexts(t, task.Classify.Input) {
						observed[*task.Classify.Head+":"+text]++
					}
				}
			}
			wanted := map[string]int{}
			for _, head := range []string{"domain", "pii", "guard"} {
				wanted[head+":"+first], wanted[head+":"+last] = 1, 1
			}
			if !reflect.DeepEqual(observed, wanted) {
				t.Fatalf("bundles lost, duplicated or misrouted input/head pairs: %+v != %+v", observed, wanted)
			}
			legacy := classify(t, fmt.Sprintf(`{"texts":[%q,%q],"task_type":%q}`, first, last, task))
			partial := classify(t, fmt.Sprintf(`{"texts":[%q,"",%q],"task_type":%q}`, first, last, task))
			if len(partial.Results) != 3 || partial.Results[1].Error == nil || partial.Results[1].Error.Code != "INVALID_INPUT" || partial.Results[1].Index != nil || !reflect.DeepEqual(partial.Statistics, legacy.Statistics) {
				t.Fatalf("legacy item failure lost siblings or changed successful statistics: %+v", partial)
			}
			if len(legacy.Results) != 2 || len(typed.Results) != 3 || typed.Results[1].Error == nil || typed.Results[1].Error.Code != "UNSUPPORTED_INPUT" {
				t.Fatalf("unexpected outcomes: legacy=%+v typed=%+v", legacy, typed)
			}
			for offset, index := range []int{0, 2} {
				got, want := typed.Results[index], legacy.Results[offset]
				if got.Error != nil || want.Error != nil || got.Index == nil || *got.Index != index {
					t.Fatalf("text sibling %d failed or moved: %+v", index, got)
				}
				got.Index = nil
				got.ProcessingTimeMs, want.ProcessingTimeMs = 0, 0
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("typed projection changed legacy %s result: %+v != %+v", task, got, want)
				}
				legacySibling := partial.Results[index]
				legacySibling.ProcessingTimeMs = 0
				if !reflect.DeepEqual(legacySibling, want) {
					t.Fatalf("legacy partial projection changed %s sibling: %+v != %+v", task, legacySibling, want)
				}
			}
		})
	}
}

// Record the real HTTP bundle bodies without changing the runtime's timer or
// fusion policy. This is the servingtest fixture with a test-local recorder.
func typedTextRuntime(t *testing.T) (*serving.Runtime, *runtimetest.Runtime, func() []api.BundleRequest) {
	t.Helper()
	fake := runtimetest.New(runtimetest.Model{ID: "batch-heads", Heads: []runtimetest.Head{
		{Name: "domain", Kind: "sequence", Labels: []string{"math", "law"}},
		{Name: "pii", Kind: "token", Labels: []string{"O", "B-PERSON", "I-PERSON"}},
		{Name: "guard", Kind: "sequence", Labels: []string{"benign", "jailbreak"}},
	}})
	var mu sync.Mutex
	var requests []api.BundleRequest
	handler := fake.Handler()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/bundle" {
			body, err := io.ReadAll(r.Body)
			_ = r.Body.Close()
			var request api.BundleRequest
			if err != nil || json.Unmarshal(body, &request) != nil {
				http.Error(w, "invalid test bundle", http.StatusBadRequest)
				return
			}
			mu.Lock()
			requests = append(requests, request)
			mu.Unlock()
			r.Body = io.NopCloser(bytes.NewReader(body))
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(server.Close)
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"batch-heads": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL}})
	if err != nil {
		t.Fatal(err)
	}
	return serving.New(lease, nil), fake, func() []api.BundleRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]api.BundleRequest(nil), requests...)
	}
}

func typedRuntimeClassifyTexts(t *testing.T, input api.ClassifyInput) []string {
	t.Helper()
	if text, err := input.AsInputText(); err == nil {
		return []string{text}
	}
	if texts, err := input.AsTextList(); err == nil {
		return texts
	}
	items, err := input.AsClassifyItemList()
	if err != nil {
		t.Fatal(err)
	}
	texts := make([]string, len(items))
	for index, item := range items {
		if item.Text == nil || item.TextPair != nil || item.Context != nil || item.Question != nil || item.Answer != nil {
			t.Fatalf("non-text head input: %+v", item)
		}
		texts[index] = *item.Text
	}
	return texts
}
