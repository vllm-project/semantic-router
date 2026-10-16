package testcases

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-bundles", pkgtestcases.TestCase{
		Description: "One request reaches each runtime process once per request stage: a process's domain, PII, guard and modality tasks of the signal stage arrive as one /v1/bundle, and the Router records each call's transport and the runtime's own time",
		Tags:        []string{"model-runtime", "bundles", "performance"},
		Fn:          testModelRuntimeBundles,
	})
}

const (
	runtimeRequestsMetric    = "vllm_srun_requests_total"
	runtimeBundleTasksMetric = "vllm_srun_bundle_tasks"
	mrBundleRequests         = 3
)

// mrSignalStageModels answer the signal stage of every request.
var mrSignalStageModels = []string{mrDomainDeployment, mrGuardDeployment, mrModalityDeployment, mrPIIDeployment}

// signalProcess is one managed process serving signal-stage models.
type signalProcess struct {
	socket string
	client *modelruntime.Client
	tasks  int
	before modelruntime.Metrics
}

// testModelRuntimeBundles sends requests and requires every process that
// serves signal-stage models, however the Router spread them, to receive one
// /v1/bundle per request carrying all of its tasks, and no single calls.
func testModelRuntimeBundles(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrManagedDeployments...); err != nil {
		return err
	}
	processes, err := signalProcesses(ctx, session)
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
	for _, process := range processes {
		after, err := process.client.Metrics(ctx)
		if err != nil {
			return err
		}
		bundles := countDelta(process.before, after, runtimeRequestsMetric, map[string]string{"endpoint": "/v1/bundle"})
		classify := countDelta(process.before, after, runtimeRequestsMetric, map[string]string{"endpoint": "/v1/classify"})
		tasks := after.Sum(runtimeBundleTasksMetric+"_sum", nil) - process.before.Sum(runtimeBundleTasksMetric+"_sum", nil)
		details[process.socket] = map[string]interface{}{"bundles": bundles, "classify_calls": classify, "bundled_tasks": tasks}
		if bundles != mrBundleRequests || classify != 0 {
			return fmt.Errorf("%d requests reached %s as %v bundles and %v single classify calls, want one bundle each", mrBundleRequests, process.socket, bundles, classify)
		}
		if tasks != float64(mrBundleRequests*process.tasks) {
			return fmt.Errorf("%s's bundles carried %v tasks, want %d per request", process.socket, tasks, process.tasks)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	return nil
}

func signalProcesses(ctx context.Context, session *modelRuntimeSession) ([]signalProcess, error) {
	runtimes, err := session.pod.ManagedRuntimes(ctx, mrSocketDir)
	if err != nil {
		return nil, err
	}
	var processes []signalProcess
	found := 0
	for socket, models := range runtimes {
		tasks := 0
		for _, model := range models {
			if slices.Contains(mrSignalStageModels, model) {
				tasks++
			}
		}
		if tasks == 0 {
			continue
		}
		client := modelruntime.NewClient(modelruntime.SocketTransport{Target: session.pod, Socket: socket})
		before, err := client.Metrics(ctx)
		if err != nil {
			return nil, err
		}
		found += tasks
		processes = append(processes, signalProcess{socket: filepath.Base(socket), client: client, tasks: tasks, before: before})
	}
	if found != len(mrSignalStageModels) {
		return nil, fmt.Errorf("managed processes %v serve %d of the signal-stage models %v", runtimes, found, mrSignalStageModels)
	}
	return processes, nil
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
