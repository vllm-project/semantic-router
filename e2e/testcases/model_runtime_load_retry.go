package testcases

import (
	"context"
	"fmt"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-load-retry", pkgtestcases.TestCase{
		Description: "A managed runtime whose every model failed to load is restarted with back-off until the model loads, with no configuration reload; requests succeed meanwhile",
		Tags:        []string{"model-runtime", "supervision", "fail-open"},
		// It hides a fixture file and kills a runtime process.
		MutatesClusterState: true,
		Fn:                  testModelRuntimeLoadRetry,
	})
}

// mrDecisionPackage is the decision fixture's directory in the Router pod;
// the fixture wrapper rewrites a missing package, so the case hides one file.
const mrDecisionPackage = "/tmp/vsr-fixtures/decision"

const hideFixtureWeights = `import pathlib, sys
root = pathlib.Path(sys.argv[1])
weights = sorted(root.glob("*.safetensors"))
if not weights:
    sys.exit("no weights under %s" % root)
weights[0].rename(weights[0].with_name(weights[0].name + ".hidden"))
print(weights[0].name)
`

const restoreFixtureWeights = `import pathlib, sys
root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
hidden = root / (name + ".hidden")
if hidden.exists():
    hidden.rename(root / name)
`

func testModelRuntimeLoadRetry(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrDecisionDeployment); err != nil {
		return err
	}
	before, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	output, err := session.pod.Exec(ctx, []string{"python3", "-c", hideFixtureWeights, mrDecisionPackage}, nil)
	if err != nil {
		return fmt.Errorf("hide the decision fixture's weights: %w", err)
	}
	hidden := strings.TrimSpace(string(output))
	restored := false
	restore := func() error {
		if restored {
			return nil
		}
		if _, restoreErr := session.pod.Exec(ctx, []string{"python3", "-c", restoreFixtureWeights, mrDecisionPackage, hidden}, nil); restoreErr != nil {
			return fmt.Errorf("restore the decision fixture's weights: %w", restoreErr)
		}
		restored = true
		return nil
	}
	defer func() { _ = restore() }()
	_, socket, err := session.managed(ctx, mrDecisionDeployment)
	if err != nil {
		return err
	}
	if _, err = session.pod.KillRuntime(ctx, socket); err != nil {
		return err
	}

	// The restarted process cannot load the package. The Router recycles it
	// (a restart beyond the one for the kill) while every request succeeds.
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
		restarts := metrics.DeploymentRestarts(mrDecisionDeployment) - before.DeploymentRestarts(mrDecisionDeployment)
		if restarts >= 1 && metrics.DeploymentReady(mrDecisionDeployment) {
			return modelruntime.Stop(fmt.Errorf("%s became ready without its weights", mrDecisionDeployment))
		}
		if restarts < 2 {
			return fmt.Errorf("%v restarts so far; waiting for a recycle after the failed load", restarts)
		}
		return nil
	})
	if err != nil {
		return err
	}
	failing, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	if err = restore(); err != nil {
		return err
	}
	started := time.Now()
	if err = session.waitReady(ctx, mrDecisionDeployment); err != nil {
		return fmt.Errorf("the deployment did not recover once its package loads: %w", err)
	}
	response, err := session.chat(ctx, freshPrompt("Suggest a name for a friendly golden retriever."))
	if err != nil {
		return err
	}
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != "selector_route" {
		return fmt.Errorf("after the recovery the selected decision is %q, want selector_route", decision)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"hidden_file":            hidden,
			"restarts_while_failing": failing.DeploymentRestarts(mrDecisionDeployment) - before.DeploymentRestarts(mrDecisionDeployment),
			"requests_while_failing": requests,
			"recovered_seconds":      time.Since(started).Seconds(),
		})
	}
	return nil
}
