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
	pkgtestcases.Register("model-runtime-load-retry-isolation", pkgtestcases.TestCase{
		Description: "A managed model recovers after a temporary weight-read failure; independent workers continue serving without restarts and no configuration reload is needed",
		Tags:        []string{"model-runtime", "supervision", "fail-open"},
		// It locks a fixture file and kills a runtime process.
		MutatesClusterState: true,
		Fn:                  testModelRuntimeLoadRetryIsolation,
	})
}

// mrModalityPackage is the modality fixture's directory in the Router pod.
const mrModalityPackage = "/tmp/vsr-fixtures/modality"

// lockFixtureWeights makes a package's weights unreadable. The load then fails
// with an error the runtime retries; a missing file instead fails the package
// for good.
const lockFixtureWeights = `import pathlib, sys
weights = sorted(pathlib.Path(sys.argv[1]).glob("*.safetensors"))
if not weights:
    sys.exit("no weights under %s" % sys.argv[1])
weights[0].chmod(0)
print(weights[0].name)
`

const unlockFixtureWeights = `import pathlib, sys
(pathlib.Path(sys.argv[1]) / sys.argv[2]).chmod(0o644)
`

func testModelRuntimeLoadRetryIsolation(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrDeviceGroup...); err != nil {
		return err
	}
	before, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	output, err := session.pod.Exec(ctx, []string{"python3", "-c", lockFixtureWeights, mrModalityPackage}, nil)
	if err != nil {
		return fmt.Errorf("lock the modality fixture's weights: %w", err)
	}
	locked := strings.TrimSpace(string(output))
	unlocked := false
	unlock := func() error {
		if unlocked {
			return nil
		}
		if _, unlockErr := session.pod.Exec(ctx, []string{"python3", "-c", unlockFixtureWeights, mrModalityPackage, locked}, nil); unlockErr != nil {
			return fmt.Errorf("unlock the modality fixture's weights: %w", unlockErr)
		}
		unlocked = true
		return nil
	}
	defer func() { _ = unlock() }()
	_, socket, err := session.managed(ctx, mrModalityDeployment)
	if err != nil {
		return err
	}
	if _, err = session.pod.KillRuntime(ctx, socket); err != nil {
		return err
	}

	// Only the affected logical worker restarts. Other workers keep serving
	// while that worker retries loading its temporarily unreadable weights.
	var siblings []string
	for _, name := range mrDeviceGroup {
		if name != mrModalityDeployment {
			siblings = append(siblings, name)
		}
	}
	var requests int
	var retrying modelruntime.ModelCard
	err = modelruntime.Eventually(ctx, mrReadyTimeout, func(ctx context.Context) error {
		if _, chatErr := session.chat(ctx, freshPrompt("Write a Go function that merges two sorted slices.")); chatErr != nil {
			return modelruntime.Stop(chatErr)
		}
		requests++
		metrics, metricsErr := session.routerMetrics(ctx)
		if metricsErr != nil {
			return metricsErr
		}
		for _, name := range siblings {
			if !metrics.DeploymentReady(name) {
				return modelruntime.Stop(fmt.Errorf("unrelated worker %s became unavailable", name))
			}
			if metrics.DeploymentRestarts(name) != before.DeploymentRestarts(name) {
				return modelruntime.Stop(fmt.Errorf("unrelated worker %s restarted", name))
			}
		}
		if metrics.DeploymentReady(mrModalityDeployment) {
			return modelruntime.Stop(fmt.Errorf("%s became ready with unreadable weights", mrModalityDeployment))
		}
		runtime, _, clientErr := session.managed(ctx, mrModalityDeployment)
		if clientErr != nil {
			return clientErr
		}
		card, cardErr := runtime.Model(ctx, mrModalityDeployment)
		if cardErr != nil {
			return cardErr
		}
		if (card.Status != "loading" && card.Status != "failed") || card.Reason == "" {
			return fmt.Errorf("%s reports %s (%s), waiting for a retry", mrModalityDeployment, card.Status, card.Reason)
		}
		retrying = card
		return nil
	})
	if err != nil {
		return err
	}
	if err = unlock(); err != nil {
		return err
	}
	started := time.Now()
	if err = session.waitReady(ctx, mrModalityDeployment); err != nil {
		return fmt.Errorf("the model did not recover once its weights are readable: %w", err)
	}
	recovered := time.Since(started)
	after, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	if restarts := after.DeploymentRestarts(mrModalityDeployment) - before.DeploymentRestarts(mrModalityDeployment); restarts < 1 {
		return fmt.Errorf("the Router restarted %s's process %v times, want at least the deliberately killed worker to restart", mrModalityDeployment, restarts)
	}
	for _, name := range siblings {
		if after.DeploymentRestarts(name) != before.DeploymentRestarts(name) || !after.DeploymentReady(name) {
			return fmt.Errorf("unrelated worker %s did not remain ready without restart", name)
		}
	}
	response, err := session.chat(ctx, freshPrompt("Suggest a name for a friendly golden retriever."))
	if err != nil {
		return err
	}
	if matched := headerItems(response.Headers, "x-vsr-matched-modality"); len(matched) != 1 {
		return fmt.Errorf("after the recovery x-vsr-matched-modality is %v, want the modality model's one choice", matched)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"locked_file": locked, "retrying_reason": retrying.Reason, "requests_while_retrying": requests,
			"recovered_seconds": recovered.Seconds(),
		})
	}
	return nil
}
