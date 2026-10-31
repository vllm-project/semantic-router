package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("shadow-dispatch-multi-arm-completes", pkgtestcases.TestCase{
		Description: "Two shadow arms receive the same approved request under one aggregate budget and the replay record keeps one bounded outcome per arm",
		Tags:        []string{"router-replay", "shadow-dispatch", "multi-arm"},
		Fn:          testShadowDispatchMultiArmCompletes,
	})
}

// testShadowDispatchMultiArmCompletes pins the multi-arm observation contract
// (issue #3376): both configured arms dispatch for one approved request, the
// primary response stays byte-identical to the client baseline, and the replay
// record carries one bounded outcome per arm with its own shadow identity.
func testShadowDispatchMultiArmCompletes(
	ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions,
) (result error) {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()
	sessionID, err := run.primary(ctx, "vllm-sr/shadow-multiarm", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	record, err := run.replay(ctx, sessionID, 2)
	if err != nil {
		return err
	}
	return requireShadowOutcomes(record,
		shadowOutcomeWant{model: "openai/shadow-candidate", verdict: "completed", reason: "completed"},
		shadowOutcomeWant{model: "openai/shadow-arm-b", verdict: "completed", reason: "completed"},
	)
}
