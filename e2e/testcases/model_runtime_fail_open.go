package testcases

import (
	"context"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-fail-open", pkgtestcases.TestCase{
		Description: "A deployment whose runtime never answers leaves its signal unknown: requests succeed, its route never matches, and the Router counts the unknown answers",
		Tags:        []string{"model-runtime", "fail-open"},
		Fn:          testModelRuntimeFailOpen,
	})
	pkgtestcases.Register("model-runtime-supervision", pkgtestcases.TestCase{
		Description: "A crashed managed runtime is restarted by the Router; requests succeed meanwhile and the other independent workers stay ready",
		Tags:        []string{"model-runtime", "supervision", "fail-open"},
		// It kills a runtime process, so a retry would not start from a clean state.
		MutatesClusterState: true,
		Fn:                  testModelRuntimeSupervision,
	})
}

const mrFailOpenRequests = 3

func testModelRuntimeFailOpen(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	before, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	for index := 0; index < mrFailOpenRequests; index++ {
		response, chatErr := session.chat(ctx, freshPrompt("Is this request about software?"))
		if chatErr != nil {
			return chatErr
		}
		if decision := response.Headers.Get("x-vsr-selected-decision"); decision == "offline_route" {
			return fmt.Errorf("offline_route matched: an unanswered decision signal must not match")
		}
		if headerItems(response.Headers, "x-vsr-matched-decision-model")["offline"] {
			return fmt.Errorf("the offline deployment's signal matched")
		}
	}
	after, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	labels := map[string]string{"deployment": mrOfflineDeployment}
	unknown := countDelta(before, after, modelruntime.RouterUnknownMetric, labels)
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"requests": mrFailOpenRequests, "unknown_answers": unknown})
	}
	if unknown < mrFailOpenRequests {
		return fmt.Errorf("%d requests left %v unknown answers for %s, want one each", mrFailOpenRequests, unknown, mrOfflineDeployment)
	}
	return nil
}

func testModelRuntimeSupervision(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrManagedDeployments...); err != nil {
		return err
	}
	before, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	_, socket, err := session.managed(ctx, mrDecisionDeployment)
	if err != nil {
		return err
	}
	killed, err := session.pod.KillRuntime(ctx, socket)
	if err != nil {
		return err
	}
	started := time.Now()

	// Until the restart is ready, every request must still succeed, and the
	// other deployment workers must stay ready.
	var requests int
	err = modelruntime.Eventually(ctx, mrReadyTimeout, func(ctx context.Context) error {
		if _, chatErr := session.chat(ctx, freshPrompt("Write a Go function that merges two sorted slices.")); chatErr != nil {
			return modelruntime.Stop(chatErr)
		}
		requests++
		metrics, metricsErr := session.routerMetrics(ctx)
		if metricsErr != nil {
			return metricsErr
		}
		for _, name := range mrDeviceGroup {
			if !metrics.DeploymentReady(name) {
				return modelruntime.Stop(fmt.Errorf("%s became unavailable when another process crashed", name))
			}
		}
		if metrics.DeploymentRestarts(mrDecisionDeployment) <= before.DeploymentRestarts(mrDecisionDeployment) {
			return fmt.Errorf("no restart of the decision worker recorded yet")
		}
		if !metrics.DeploymentReady(mrDecisionDeployment) {
			return fmt.Errorf("%s is restarting", mrDecisionDeployment)
		}
		return nil
	})
	if err != nil {
		return err
	}
	response, err := session.chat(ctx, freshPrompt("Suggest a name for a friendly golden retriever."))
	if err != nil {
		return err
	}
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != "selector_route" {
		return fmt.Errorf("after the restart the selected decision is %q, want selector_route", decision)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"killed_pid": killed, "recovered_seconds": time.Since(started).Seconds(), "requests_during_recovery": requests,
		})
	}
	return nil
}
