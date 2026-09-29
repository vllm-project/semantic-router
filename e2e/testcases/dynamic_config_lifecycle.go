package testcases

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	lifecycleNamespace = "vllm-semantic-router-system"
	lifecyclePoolName  = "ai-gateway-pool"
	lifecycleRouteName = "ai-gateway-route"
	lifecycleKeyword   = "lifecyclemarker"
	lifecycleModel     = "base-model"
	lifecycleAdapter   = "general-expert"
	lifecycleMissing   = "missing-lifecycle-model"
	lifecycleSamples   = 3
)

func init() {
	pkgtestcases.Register("dynamic-config-generation-lifecycle", pkgtestcases.TestCase{
		Description: "Require current-generation readiness, retain active routing after a rejected reference, and activate a recovered generation",
		Tags:        []string{"dynamic-config", "kubernetes", "readiness", "routing"},
		Fn:          testDynamicConfigGenerationLifecycle,
	})
}

type lifecycleStatus struct {
	UID                string `json:"uid"`
	Generation         int64  `json:"generation"`
	ObservedGeneration int64  `json:"observed_generation"`
	ReadyGeneration    int64  `json:"ready_observed_generation"`
	Ready              string `json:"ready"`
	Reason             string `json:"reason"`
	Message            string `json:"message"`
}

type lifecycleResponse struct {
	Status   int    `json:"http_status"`
	Decision string `json:"selected_decision"`
	Model    string `json:"selected_model"`
	Path     string `json:"response_path"`
}

type lifecyclePhase struct {
	Name               string              `json:"phase"`
	ExpectedGeneration int64               `json:"expected_route_generation"`
	Pool               lifecycleStatus     `json:"pool"`
	Route              lifecycleStatus     `json:"route"`
	Responses          []lifecycleResponse `json:"responses,omitempty"`
	Error              string              `json:"error,omitempty"`
}

type dynamicConfigLifecycle struct {
	pool, route dynamic.ResourceInterface
	interval    time.Duration
	timeout     time.Duration
	probe       func(context.Context, string, int) (lifecycleResponse, error)
	phases      []lifecyclePhase
}

func testDynamicConfigGenerationLifecycle(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.RestConfig == nil {
		return fmt.Errorf("dynamic-config lifecycle requires a Kubernetes REST config")
	}
	dynamicClient, err := dynamic.NewForConfig(opts.RestConfig)
	if err != nil {
		return err
	}
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	chat := fixtures.NewChatCompletionsClient(session, 30*time.Second)
	lifecycle := dynamicConfigLifecycle{
		pool: dynamicClient.Resource(schema.GroupVersionResource{
			Group: "vllm.ai", Version: "v1alpha1", Resource: "intelligentpools",
		}).Namespace(lifecycleNamespace),
		route: dynamicClient.Resource(schema.GroupVersionResource{
			Group: "vllm.ai", Version: "v1alpha1", Resource: "intelligentroutes",
		}).Namespace(lifecycleNamespace),
		interval: 2 * time.Second,
		timeout:  10 * time.Minute,
		probe: func(ctx context.Context, decision string, sample int) (lifecycleResponse, error) {
			response, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
				Model: "MoM",
				Messages: []fixtures.ChatMessage{{Role: "user", Content: fmt.Sprintf(
					"Please say hello for %s, phase %s, sample %d.", lifecycleKeyword, decision, sample)}},
			}, nil)
			if err != nil {
				return lifecycleResponse{}, err
			}
			return checkLifecycleResponse(response, decision)
		},
	}
	defer func() {
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"phases": lifecycle.phases, "required_routed_samples_per_phase": lifecycleSamples,
				"rejection_contract": "ValidationFailed: ordinary unknown model reference; not a warmup failure",
			})
		}
	}()
	return lifecycle.run(ctx)
}

func (l *dynamicConfigLifecycle) run(ctx context.Context) (result error) {
	pool, err := l.pool.Get(ctx, lifecyclePoolName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	original, err := l.route.Get(ctx, lifecycleRouteName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	originalSpec, found, err := unstructured.NestedMap(original.Object, "spec")
	if err != nil {
		return fmt.Errorf("read original route spec: %w", err)
	}
	if !found {
		return fmt.Errorf("original route has no spec")
	}
	if err = l.observe(ctx, "initial", pool, original, "True", "Ready", ""); err != nil {
		return err
	}
	// Always restore the original spec, including after a failed assertion or
	// canceled testcase. Cleanup itself must reach current-generation Ready.
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), l.timeout)
		defer cancel()
		restored, restoreErr := l.updateSpec(cleanupCtx, original, originalSpec)
		if restoreErr == nil {
			restoreErr = l.observe(cleanupCtx, "restored_original", pool, restored, "True", "Ready", "")
		}
		if restoreErr != nil {
			result = errors.Join(result, fmt.Errorf("restore original route: %w", restoreErr))
		}
	}()
	validSpec, err := lifecycleRouteSpec(originalSpec, "lifecycle_active", lifecycleModel)
	if err != nil {
		return err
	}
	active, err := l.updateSpec(ctx, original, validSpec)
	if err != nil {
		return fmt.Errorf("publish valid route: %w", err)
	}
	if err = l.observe(ctx, "valid_active", pool, active, "True", "Ready", "lifecycle_active"); err != nil {
		return err
	}
	invalidSpec, err := lifecycleRouteSpec(originalSpec, "lifecycle_rejected", lifecycleMissing)
	if err != nil {
		return err
	}
	// Update must succeed at the API before a current-generation controller
	// rejection counts. A schema/admission rejection is not this contract.
	rejected, err := l.updateSpec(ctx, active, invalidSpec)
	if err != nil {
		return fmt.Errorf("API did not accept the ordinary reference error: %w", err)
	}
	if err = l.observe(ctx, "rejected_retains_active", pool, rejected, "False", "ValidationFailed", "lifecycle_active"); err != nil {
		return err
	}
	recoveredSpec, err := lifecycleRouteSpec(originalSpec, "lifecycle_recovered", lifecycleModel)
	if err != nil {
		return err
	}
	recovered, err := l.updateSpec(ctx, rejected, recoveredSpec)
	if err != nil {
		return fmt.Errorf("publish recovered route: %w", err)
	}
	return l.observe(ctx, "recovered_active", pool, recovered, "True", "Ready", "lifecycle_recovered")
}

func lifecycleRouteSpec(original map[string]interface{}, decision, model string) (map[string]interface{}, error) {
	obj := (&unstructured.Unstructured{Object: map[string]interface{}{"spec": original}}).DeepCopy()
	keywords, _, err := unstructured.NestedSlice(obj.Object, "spec", "signals", "keywords")
	if err != nil {
		return nil, err
	}
	keywords = append(keywords, map[string]interface{}{
		"name": lifecycleKeyword, "operator": "OR", "keywords": []interface{}{lifecycleKeyword}, "caseSensitive": false,
	})
	if err = unstructured.SetNestedSlice(obj.Object, keywords, "spec", "signals", "keywords"); err != nil {
		return nil, err
	}
	decisions, _, err := unstructured.NestedSlice(obj.Object, "spec", "decisions")
	if err != nil {
		return nil, err
	}
	decisions = append(decisions, map[string]interface{}{
		"name": decision, "priority": int64(1000),
		"signals": map[string]interface{}{"operator": "OR", "conditions": []interface{}{
			map[string]interface{}{"type": "keyword", "name": lifecycleKeyword},
		}},
		"modelRefs": []interface{}{map[string]interface{}{"model": model, "loraName": lifecycleAdapter, "useReasoning": false}},
		// Every assertion must reach routing/backend execution, not a previously
		// cached completion from the valid generation.
		"plugins": []interface{}{map[string]interface{}{"type": "response_cache", "configuration": map[string]interface{}{"enabled": false}}},
	})
	if err = unstructured.SetNestedSlice(obj.Object, decisions, "spec", "decisions"); err != nil {
		return nil, err
	}
	spec, _, err := unstructured.NestedMap(obj.Object, "spec")
	return spec, err
}

func (l *dynamicConfigLifecycle) updateSpec(ctx context.Context, previous *unstructured.Unstructured, spec map[string]interface{}) (*unstructured.Unstructured, error) {
	var updated *unstructured.Unstructured
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current, err := l.route.Get(ctx, lifecycleRouteName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if current.GetUID() != previous.GetUID() {
			return fmt.Errorf("route was replaced during the lifecycle")
		}
		if err = unstructured.SetNestedMap(current.Object, spec, "spec"); err != nil {
			return err
		}
		updated, err = l.route.Update(ctx, current, metav1.UpdateOptions{})
		return err
	})
	if err != nil {
		return nil, err
	}
	if updated.GetGeneration() <= previous.GetGeneration() {
		return nil, fmt.Errorf("API update did not advance route generation: %d -> %d", previous.GetGeneration(), updated.GetGeneration())
	}
	return updated, nil
}

func (l *dynamicConfigLifecycle) observe(ctx context.Context, name string, pool, route *unstructured.Unstructured, ready, reason, decision string) (result error) {
	phase := lifecyclePhase{Name: name, ExpectedGeneration: route.GetGeneration()}
	defer func() {
		if result != nil {
			phase.Error = result.Error()
		}
		l.phases = append(l.phases, phase)
	}()
	read := func(ctx context.Context) (bool, error) {
		p, err := l.pool.Get(ctx, lifecyclePoolName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		r, err := l.route.Get(ctx, lifecycleRouteName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		phase.Pool, phase.Route = readLifecycleStatus(p), readLifecycleStatus(r)
		return lifecycleStatusMatches(phase.Pool, pool, ready, reason) && lifecycleStatusMatches(phase.Route, route, ready, reason), nil
	}
	if err := wait.PollUntilContextTimeout(ctx, l.interval, l.timeout, true, read); err != nil {
		return fmt.Errorf("%s: current-generation %s/%s not reached (pool=%+v route=%+v): %w", name, ready, reason, phase.Pool, phase.Route, err)
	}
	if reason == "ValidationFailed" && !strings.Contains(phase.Route.Message, "references unknown model: "+lifecycleMissing) {
		return fmt.Errorf("%s: rejection was not the expected model reference error: %s", name, phase.Route.Message)
	}
	if decision == "" {
		return nil
	}
	for sample := 1; sample <= lifecycleSamples; sample++ {
		response, err := l.probe(ctx, decision, sample)
		phase.Responses = append(phase.Responses, response)
		if err != nil {
			return fmt.Errorf("%s: routed sample %d: %w", name, sample, err)
		}
	}
	// Do not accept a successful response if readiness/generation changed
	// while it was in flight (especially a rejected CR becoming Ready).
	matched, err := read(ctx)
	if err != nil {
		return fmt.Errorf("%s: read status after routed probes: %w", name, err)
	}
	if !matched {
		return fmt.Errorf("%s: status changed during routed probes: pool=%+v route=%+v", name, phase.Pool, phase.Route)
	}
	return nil
}

func readLifecycleStatus(obj *unstructured.Unstructured) lifecycleStatus {
	status := lifecycleStatus{UID: string(obj.GetUID()), Generation: obj.GetGeneration()}
	status.ObservedGeneration, _, _ = unstructured.NestedInt64(obj.Object, "status", "observedGeneration")
	conditions, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	for _, value := range conditions {
		condition, ok := value.(map[string]interface{})
		if !ok || condition["type"] != "Ready" {
			continue
		}
		status.Ready, _, _ = unstructured.NestedString(condition, "status")
		status.Reason, _, _ = unstructured.NestedString(condition, "reason")
		status.Message, _, _ = unstructured.NestedString(condition, "message")
		status.ReadyGeneration, _, _ = unstructured.NestedInt64(condition, "observedGeneration")
	}
	return status
}

func lifecycleStatusMatches(status lifecycleStatus, expected *unstructured.Unstructured, ready, reason string) bool {
	return status.UID == string(expected.GetUID()) && status.Generation > 0 &&
		status.Generation == expected.GetGeneration() && status.ObservedGeneration == status.Generation &&
		status.ReadyGeneration == status.Generation && status.Ready == ready && status.Reason == reason
}

func checkLifecycleResponse(response *fixtures.HTTPResponse, decision string) (lifecycleResponse, error) {
	actual := lifecycleResponse{
		Status: response.StatusCode, Decision: response.Headers.Get("x-vsr-selected-decision"),
		Model: response.Headers.Get("x-vsr-selected-model"),
		Path:  response.Headers.Get("x-vsr-response-path"),
	}
	if actual.Status != http.StatusOK || actual.Decision != decision || actual.Model != lifecycleAdapter || actual.Path != "upstream" {
		return actual, fmt.Errorf("want HTTP 200, decision %q, model %q, upstream response; got %+v", decision, lifecycleAdapter, actual)
	}
	var completion struct {
		Choices []struct {
			Message struct{ Content string }
		}
	}
	if err := json.Unmarshal(response.Body, &completion); err != nil {
		return actual, fmt.Errorf("decode routed completion: %w", err)
	}
	if len(completion.Choices) == 0 || strings.TrimSpace(completion.Choices[0].Message.Content) == "" {
		return actual, fmt.Errorf("routed completion has no assistant content")
	}
	return actual, nil
}
