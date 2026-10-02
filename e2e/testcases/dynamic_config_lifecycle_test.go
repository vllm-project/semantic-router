package testcases

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func lifecycleTestObject(kind, name string, generation int64) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "vllm.ai/v1alpha1", "kind": kind,
		"spec": map[string]interface{}{"decisions": []interface{}{}, "signals": map[string]interface{}{}},
	}}
	obj.SetName(name)
	obj.SetNamespace(lifecycleNamespace)
	obj.SetUID(types.UID(name + "-uid"))
	obj.SetGeneration(generation)
	setLifecycleTestStatus(obj, "True", "Ready", "")
	return obj
}

func setLifecycleTestStatus(obj *unstructured.Unstructured, ready, reason, message string) {
	obj.Object["status"] = map[string]interface{}{
		"observedGeneration": obj.GetGeneration(),
		"conditions": []interface{}{map[string]interface{}{
			"type": "Ready", "status": ready, "reason": reason, "message": message,
			"observedGeneration": obj.GetGeneration(),
		}},
	}
}

func TestLifecycleRequiresExactGenerationAndStatus(t *testing.T) {
	obj := lifecycleTestObject("IntelligentRoute", lifecycleRouteName, 4)
	valid := readLifecycleStatus(obj)
	if !lifecycleStatusMatches(valid, obj, "True", "Ready") {
		t.Fatal("current-generation Ready was rejected")
	}
	for name, mutate := range map[string]func(*lifecycleStatus){
		"stale condition": func(s *lifecycleStatus) { s.ReadyGeneration-- },
		"stale status":    func(s *lifecycleStatus) { s.ObservedGeneration-- },
		"new generation":  func(s *lifecycleStatus) { s.Generation++ },
		"replacement":     func(s *lifecycleStatus) { s.UID = "replacement" },
		"not ready":       func(s *lifecycleStatus) { s.Ready = "False" },
		"wrong reason":    func(s *lifecycleStatus) { s.Reason = "Activating" },
	} {
		t.Run(name, func(t *testing.T) {
			status := valid
			mutate(&status)
			if lifecycleStatusMatches(status, obj, "True", "Ready") {
				t.Fatal("mismatched status was accepted")
			}
		})
	}
}

func TestLifecycleResponseRequiresEveryContract(t *testing.T) {
	valid := func() *fixtures.HTTPResponse {
		return &fixtures.HTTPResponse{StatusCode: http.StatusOK, Headers: http.Header{
			"X-Vsr-Selected-Decision": {"lifecycle_recovered"}, "X-Vsr-Selected-Model": {lifecycleAdapter},
			"X-Vsr-Response-Path": {"upstream"},
		}, Body: []byte(`{"choices":[{"message":{"content":"Hello"}}]}`)}
	}
	if _, err := checkLifecycleResponse(valid(), "lifecycle_recovered"); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*fixtures.HTTPResponse){
		"failed backend": func(r *fixtures.HTTPResponse) { r.StatusCode = http.StatusBadGateway },
		"old decision":   func(r *fixtures.HTTPResponse) { r.Headers.Set("x-vsr-selected-decision", "lifecycle_active") },
		"wrong model":    func(r *fixtures.HTTPResponse) { r.Headers.Set("x-vsr-selected-model", "other") },
		"cache hit":      func(r *fixtures.HTTPResponse) { r.Headers.Set("x-vsr-response-path", "cache") },
		"empty choices":  func(r *fixtures.HTTPResponse) { r.Body = []byte(`{"choices":[]}`) },
		"empty content":  func(r *fixtures.HTTPResponse) { r.Body = []byte(`{"choices":[{"message":{}}]}`) },
		"invalid JSON":   func(r *fixtures.HTTPResponse) { r.Body = []byte(`not a completion`) },
	} {
		t.Run(name, func(t *testing.T) {
			response := valid()
			mutate(response)
			if _, err := checkLifecycleResponse(response, "lifecycle_recovered"); err == nil {
				t.Fatal("invalid routed response was accepted")
			}
		})
	}
}

// This HTTP API fixture tests the real dynamic client's serialization and the
// E2E driver's acceptance/restoration. It does not simulate runtime activation;
// the real controller and routed responses are exercised by the live profile.
func newLifecycleTestDriver(t *testing.T, admissionReject bool) *dynamicConfigLifecycle {
	t.Helper()
	pool := lifecycleTestObject("IntelligentPool", lifecyclePoolName, 1)
	route := lifecycleTestObject("IntelligentRoute", lifecycleRouteName, 1)
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			if strings.HasSuffix(r.URL.Path, "/intelligentpools/"+lifecyclePoolName) {
				_ = json.NewEncoder(w).Encode(pool)
				return
			}
			_ = json.NewEncoder(w).Encode(route)
			return
		}
		if r.Method != http.MethodPut || !strings.HasSuffix(r.URL.Path, "/intelligentroutes/"+lifecycleRouteName) {
			http.Error(w, "unexpected API request", http.StatusNotFound)
			return
		}
		candidate := &unstructured.Unstructured{}
		if err := json.NewDecoder(r.Body).Decode(candidate); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		decisions, _, _ := unstructured.NestedSlice(candidate.Object, "spec", "decisions")
		ready, reason, message := "True", "Ready", ""
		if len(decisions) > 0 {
			refs, _, _ := unstructured.NestedSlice(decisions[len(decisions)-1].(map[string]interface{}), "modelRefs")
			if refs[0].(map[string]interface{})["model"] == lifecycleMissing {
				if admissionReject {
					w.WriteHeader(http.StatusUnprocessableEntity)
					_, _ = w.Write([]byte(`{"kind":"Status","apiVersion":"v1","status":"Failure","message":"API rejected the spec","reason":"Invalid","code":422}`))
					return
				}
				ready, reason, message = "False", "ValidationFailed", "decision lifecycle_rejected references unknown model: "+lifecycleMissing
			}
		}
		candidate.SetGeneration(route.GetGeneration() + 1)
		setLifecycleTestStatus(candidate, ready, reason, message)
		setLifecycleTestStatus(pool, ready, reason, message)
		route = candidate
		_ = json.NewEncoder(w).Encode(candidate)
	}))
	t.Cleanup(server.Close)
	client, err := dynamic.NewForConfig(&rest.Config{Host: server.URL, QPS: 1000, Burst: 1000})
	if err != nil {
		t.Fatal(err)
	}
	poolGVR := schema.GroupVersionResource{Group: "vllm.ai", Version: "v1alpha1", Resource: "intelligentpools"}
	routeGVR := schema.GroupVersionResource{Group: "vllm.ai", Version: "v1alpha1", Resource: "intelligentroutes"}
	return &dynamicConfigLifecycle{
		pool: client.Resource(poolGVR).Namespace(lifecycleNamespace), route: client.Resource(routeGVR).Namespace(lifecycleNamespace),
		interval: time.Millisecond, timeout: time.Second,
		probe: func(_ context.Context, decision string, _ int) (lifecycleResponse, error) {
			return lifecycleResponse{Status: 200, Decision: decision, Model: lifecycleAdapter, Path: "upstream"}, nil
		},
	}
}

func TestLifecycleRecordsAllGenerationsAndRestoresOriginalSpec(t *testing.T) {
	lifecycle := newLifecycleTestDriver(t, false)
	original, err := lifecycle.route.Get(context.Background(), lifecycleRouteName, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err = lifecycle.run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(lifecycle.phases) != 5 {
		t.Fatalf("phases = %+v", lifecycle.phases)
	}
	for i, phase := range lifecycle.phases {
		if phase.Route.Generation != int64(i+1) || phase.Route.ReadyGeneration != int64(i+1) {
			t.Fatalf("phase %d lost generation evidence: %+v", i, phase)
		}
		if i > 0 && i < 4 && len(phase.Responses) != lifecycleSamples {
			t.Fatalf("phase %s did not require every routed sample", phase.Name)
		}
	}
	if phase := lifecycle.phases[2]; phase.Route.Ready != "False" || phase.Responses[0].Decision != "lifecycle_active" {
		t.Fatalf("rejection did not verify previous routing: %+v", phase)
	}
	restored, err := lifecycle.route.Get(context.Background(), lifecycleRouteName, metav1.GetOptions{})
	if err != nil || !reflect.DeepEqual(restored.Object["spec"], original.Object["spec"]) {
		t.Fatalf("original spec not restored: %v", err)
	}
}

func TestLifecycleDoesNotProbeUntilExpectedGenerationIsReady(t *testing.T) {
	lifecycle := newLifecycleTestDriver(t, false)
	lifecycle.timeout = 100 * time.Millisecond
	pool, err := lifecycle.pool.Get(context.Background(), lifecyclePoolName, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	route, err := lifecycle.route.Get(context.Background(), lifecycleRouteName, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	route.SetGeneration(route.GetGeneration() + 1)
	probed := false
	lifecycle.probe = func(context.Context, string, int) (lifecycleResponse, error) {
		probed = true
		return lifecycleResponse{}, nil
	}
	err = lifecycle.observe(context.Background(), "stale_ready", pool, route, "True", "Ready", "lifecycle_recovered")
	if err == nil || probed || len(lifecycle.phases) != 1 || lifecycle.phases[0].Error == "" {
		t.Fatalf("stale Ready was not a recorded failure: error=%v probed=%t phases=%+v", err, probed, lifecycle.phases)
	}
}

func TestLifecycleRejectsAdmissionFailureAndStillRestores(t *testing.T) {
	lifecycle := newLifecycleTestDriver(t, true)
	if err := lifecycle.run(context.Background()); err == nil || !strings.Contains(err.Error(), "API did not accept") {
		t.Fatalf("admission rejection incorrectly counted as controller rejection: %v", err)
	}
	if last := lifecycle.phases[len(lifecycle.phases)-1]; last.Name != "restored_original" || last.Error != "" {
		t.Fatalf("failure did not restore original: %+v", last)
	}
}

func TestLifecycleProbeFailureCannotPassAndRestoresAfterCancellation(t *testing.T) {
	lifecycle := newLifecycleTestDriver(t, false)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	lifecycle.probe = func(context.Context, string, int) (lifecycleResponse, error) {
		calls++
		if calls == 2 {
			cancel()
			return lifecycleResponse{Status: 502}, errors.New("routed request failed")
		}
		return lifecycleResponse{Status: 200}, nil
	}
	if err := lifecycle.run(ctx); err == nil || !strings.Contains(err.Error(), "routed sample 2") {
		t.Fatalf("partial request success incorrectly passed: %v", err)
	}
	if last := lifecycle.phases[len(lifecycle.phases)-1]; last.Name != "restored_original" || last.Error != "" {
		t.Fatalf("cancellation prevented restoration: %+v", last)
	}
}
