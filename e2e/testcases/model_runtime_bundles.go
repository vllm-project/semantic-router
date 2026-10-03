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
		Description: "One request reaches each runtime process once per request stage: the domain, PII and guard tasks of the signal stage arrive as one /v1/bundle",
		Tags:        []string{"model-runtime", "bundles", "performance"},
		Fn:          testModelRuntimeBundles,
	})
}

const (
	runtimeRequestsMetric    = "vllm_sr_runtime_requests_total"
	runtimeBundleTasksMetric = "vllm_sr_runtime_bundle_tasks"
	// The device group answers the domain, PII and guard signals.
	mrDeviceGroupTasks = 3
	mrBundleRequests   = 3
)

func testModelRuntimeBundles(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err := session.waitReady(ctx, mrManagedDeployments...); err != nil {
		return err
	}
	device, _, err := session.managed(ctx, mrDomainDeployment)
	if err != nil {
		return err
	}
	before, err := device.Metrics(ctx)
	if err != nil {
		return err
	}
	for index := 0; index < mrBundleRequests; index++ {
		if _, err := session.chat(ctx, freshPrompt("Explain how a hash map handles collisions.")); err != nil {
			return err
		}
	}
	after, err := device.Metrics(ctx)
	if err != nil {
		return err
	}

	bundles := countDelta(before, after, runtimeRequestsMetric, map[string]string{"endpoint": "/v1/bundle"})
	classify := countDelta(before, after, runtimeRequestsMetric, map[string]string{"endpoint": "/v1/classify"})
	tasks := after.Sum(runtimeBundleTasksMetric+"_sum", nil) - before.Sum(runtimeBundleTasksMetric+"_sum", nil)
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"requests": mrBundleRequests, "bundles": bundles, "classify_calls": classify, "bundled_tasks": tasks})
	}
	if bundles != mrBundleRequests || classify != 0 {
		return fmt.Errorf("%d requests reached the device group as %v bundles and %v single classify calls, want one bundle each", mrBundleRequests, bundles, classify)
	}
	if tasks != mrBundleRequests*mrDeviceGroupTasks {
		return fmt.Errorf("the bundles carried %v tasks, want %d per request (domain, PII, guard)", tasks, mrDeviceGroupTasks)
	}
	return nil
}

func countDelta(before, after modelruntime.Metrics, name string, labels map[string]string) float64 {
	return after.Sum(name, labels) - before.Sum(name, labels)
}
