//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

const typedTextSystemOneConfig = `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
global:
  model_catalog:
    system: {decision_model: {deployment: vela2}}
    deployments:
      vela2: {provider: model_runtime, endpoint: "http://vela2.invalid:8100"}
    bindings: {pii: vela2, jailbreak: vela2}
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
          predicate: {gte: 0.5}
      jailbreak:
        - name: attack
          predicate: {gte: 0.9}
    decisions:
      - name: heads
        priority: 100
        rules:
          operator: OR
          conditions:
            - {type: pii, name: personal_data}
            - {type: jailbreak, name: attack}
        modelRefs: [{model: general}]
`

// The HTTP test runtime exercises the actual decision client and bundled
// transport. It does not establish real-model behaviour or numerical parity.
func TestTypedTextTasksPreserveLegacyProjectionAndRuntimeBundling(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(typedTextSystemOneConfig))
	if err != nil {
		t.Fatal(err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"vela2": {Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON", "EMAIL_ADDRESS"}}, Joint: true},
	})
	classifiers, err := classification.BuildRecipeClassifiers(cfg, classification.RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	if err := classifiers.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.DeploymentCard(context.Background(), "vela2", cfg.ModelDeployments["vela2"]); err != nil {
		t.Fatal(err)
	}
	svc := services.NewRecipeClassificationService(classifiers, cfg)
	if !svc.HasUnifiedClassifier() {
		t.Fatal("text classifier not prepared")
	}
	server := &ClassificationAPIServer{classificationSvc: svc, config: cfg}
	classify := func(body string) BatchClassificationResponse {
		t.Helper()
		recorder := httptest.NewRecorder()
		server.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		if recorder.Code != http.StatusOK {
			t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
		}
		var result BatchClassificationResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	for _, task := range []string{"pii", "security", "all"} {
		t.Run(task, func(t *testing.T) {
			first, last := "person first "+task, "benign last "+task
			before := len(fake.Decisions())
			beforeBundles, beforeTasks := fake.Bundles()
			beforeDirect := fake.Calls("decisions")
			typed := classify(fmt.Sprintf(`{"inputs":[{"type":"text","text":%q},{"type":"image","image_url":"data:image/png;base64,iVBORw0KGgo="},{"type":"text","text":%q}],"task_type":%q}`, first, last, task))
			calls := fake.Decisions()[before:]
			afterBundles, afterTasks := fake.Bundles()
			if len(calls) == 0 || afterBundles <= beforeBundles || afterTasks <= beforeTasks || fake.Calls("decisions") != beforeDirect {
				t.Fatalf("typed text bypassed bundled runtime: bundles=%d tasks=%d decisions=%d direct=%d", afterBundles-beforeBundles, afterTasks-beforeTasks, len(calls), fake.Calls("decisions")-beforeDirect)
			}
			encoded, err := json.Marshal(calls)
			if err != nil {
				t.Fatal(err)
			}
			for _, text := range []string{first, last} {
				if !strings.Contains(string(encoded), text) {
					t.Fatalf("runtime did not receive %q: %s", text, encoded)
				}
			}
			if strings.Contains(string(encoded), "iVBORw0KGgo=") {
				t.Fatal("unsupported image reached text runtime")
			}
			// Count logical questions per state even if the runtime combines
			// several states or splits physical sends across bundles.
			observed := map[string]int{}
			count := func(state interface{}, questions map[string]api.Question) {
				t.Helper()
				body, err := json.Marshal(state)
				if err != nil {
					t.Fatal(err)
				}
				name := ""
				for _, text := range []string{first, last} {
					if strings.Contains(string(body), text) {
						if name != "" {
							t.Fatalf("one state contains both inputs: %s", body)
						}
						name = text
					}
				}
				if name == "" {
					t.Fatalf("unknown runtime state: %s", body)
				}
				for id := range questions {
					observed[name+":"+id]++
				}
			}
			for _, call := range calls {
				if call.States == nil {
					count(call.State, call.Questions)
					continue
				}
				for _, state := range *call.States {
					count(state.State, state.Questions)
				}
			}
			wantQuestions := map[string]int{}
			for _, text := range []string{first, last} {
				for _, id := range []string{"preset:domain", "preset:pii", "preset:jailbreak"} {
					wantQuestions[text+":"+id] = 1
				}
			}
			if !reflect.DeepEqual(observed, wantQuestions) {
				t.Fatalf("lost, duplicate or misrouted questions: %v want %v", observed, wantQuestions)
			}
			legacy := classify(fmt.Sprintf(`{"texts":[%q,%q],"task_type":%q}`, first, last, task))
			if len(typed.Results) != 3 || len(legacy.Results) != 2 || typed.Results[1].Error == nil || typed.Results[1].Error.Code != "UNSUPPORTED_INPUT" {
				t.Fatalf("typed=%+v legacy=%+v", typed, legacy)
			}
			for offset, index := range []int{0, 2} {
				got, want := typed.Results[index], legacy.Results[offset]
				if got.Error != nil || want.Error != nil || got.Index == nil || *got.Index != index {
					t.Fatalf("sibling %d: %+v", index, got)
				}
				got.Index = nil
				got.ProcessingTimeMs, want.ProcessingTimeMs = 0, 0
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("typed projection differs: %+v != %+v", got, want)
				}
			}
		})
	}
}
