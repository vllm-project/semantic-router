package testcases

import (
	"context"
	"fmt"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-bundles", pkgtestcases.TestCase{
		Description: "One request reaches each runtime model once per request stage: the signal-stage questions of each attached model arrive as one task of a /v1/bundle, never as single calls, and the Router records each call's transport and the runtime's own time",
		Tags:        []string{"model-runtime", "bundles", "performance"},
		Fn:          testModelRuntimeBundles,
	})
}

const (
	runtimeRequestsMetric    = "vllm_srun_requests_total"
	runtimeBundleTasksMetric = "vllm_srun_bundle_tasks"
	mrBundleRequests         = 3
)

// mrSignalStageModels are the two models of the attached runtime process;
// each answers only signal-stage questions of every request.
var mrSignalStageModels = []string{mrAttachedDecisions, mrAttachedVela2}

// testModelRuntimeBundles sends requests and requires the attached process,
// which serves two signal-stage models, to receive each model's questions of
// a request as one task in /v1/bundle calls, one per request or one per model,
// and no single decisions calls.
func testModelRuntimeBundles(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrSignalStageModels...); err != nil {
		return err
	}
	runtime := session.attachedRuntime()
	before, err := runtime.Metrics(ctx)
	if err != nil {
		return err
	}
	routerBefore, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	for index := 0; index < mrBundleRequests; index++ {
		if _, err = session.chat(ctx, freshPrompt("Explain how a hash map handles collisions.")); err != nil {
			return err
		}
	}
	routerAfter, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	details := map[string]interface{}{"requests": mrBundleRequests}
	if err = checkRuntimeTiming(routerBefore, routerAfter, details); err != nil {
		return err
	}
	after, err := runtime.Metrics(ctx)
	if err != nil {
		return err
	}
	bundles := countDelta(before, after, runtimeRequestsMetric, map[string]string{"endpoint": "/v1/bundle"})
	single := countDelta(before, after, runtimeRequestsMetric, map[string]string{"endpoint": "/v1/decisions"})
	tasks := after.Sum(runtimeBundleTasksMetric+"_sum", nil) - before.Sum(runtimeBundleTasksMetric+"_sum", nil)
	details[mrAttachedService] = map[string]interface{}{"bundles": bundles, "decisions_calls": single, "bundled_tasks": tasks}
	want := float64(mrBundleRequests * len(mrSignalStageModels))
	if single != 0 || bundles < mrBundleRequests || bundles > want {
		return fmt.Errorf("%d requests reached %s as %v bundles and %v single decisions calls, want one bundle per request or per model", mrBundleRequests, mrAttachedService, bundles, single)
	}
	if tasks != want {
		return fmt.Errorf("%s's bundles carried %v tasks, want one per model and request, %v", mrAttachedService, tasks, want)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	return nil
}

func countDelta(before, after modelruntime.Metrics, name string, labels map[string]string) float64 {
	return after.Sum(name, labels) - before.Sum(name, labels)
}

// checkRuntimeTiming requires every signal-stage call to have recorded its
// transport and the runtime's Server-Timing: each deployment's calls in the
// window record a transport sample each, and their forwards took time.
func checkRuntimeTiming(before, after modelruntime.Metrics, details map[string]interface{}) error {
	for _, deployment := range mrSignalStageModels {
		calls := map[string]string{"deployment": deployment}
		timed := countDelta(before, after, modelruntime.RouterTransportMetric+"_count", calls)
		requests := countDelta(before, after, "vsr_model_runtime_request_duration_seconds_count", calls)
		forward := countDelta(before, after, modelruntime.RouterServerMetric+"_sum", map[string]string{"deployment": deployment, "phase": "forward"})
		transport := countDelta(before, after, modelruntime.RouterTransportMetric+"_sum", calls)
		details[deployment] = map[string]interface{}{"calls": requests, "timed_calls": timed, "forward_seconds": forward, "transport_seconds": transport}
		if requests < mrBundleRequests || timed != requests || forward <= 0 {
			return fmt.Errorf("%s: %v of %v runtime calls recorded a transport, with %v s of forward; want every call timed by the runtime", deployment, timed, requests, forward)
		}
	}
	return nil
}
