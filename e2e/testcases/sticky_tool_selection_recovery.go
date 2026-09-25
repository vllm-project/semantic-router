package testcases

import (
	"context"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const stickyToolSelectionRecoveryDecision = "sticky_tool_selection_decision"

func init() {
	pkgtestcases.Register("sticky-tool-selection-recovery", pkgtestcases.TestCase{
		Description: "Verify local sticky state loss on router restart falls back to stateless tool selection",
		Tags:        []string{"kubernetes", "plugin", "tool-selection", "sticky", "restart", "recovery"},
		Fn:          testStickyToolSelectionRecovery,
	})
}

func testStickyToolSelectionRecovery(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] sticky tool-selection restart recovery")
	}

	sessions, err := openStickySessionPair(ctx, client, opts)
	if err != nil {
		return err
	}

	sessionID := fmt.Sprintf("sticky-restart-%d", time.Now().UnixNano())
	headers := stickyRequestHeaders(sessionID, true)
	tools := stickyContractTools(1, "")
	_, retained, err := runStickyTrustedTurns(ctx, sessions, sessionID, headers, tools)
	sessions.gateway.Close()
	sessions.backend.Close()
	if err != nil {
		return err
	}
	if len(retained.Tools) != stickyToolSelectionMaxTools {
		return fmt.Errorf("restart recovery precondition retained %d tools, want %d", len(retained.Tools), stickyToolSelectionMaxTools)
	}

	if restartErr := restartStickySemanticRouterContainer(ctx, client, opts); restartErr != nil {
		return restartErr
	}
	if readinessErr := waitForSemanticRouterReady(ctx, client, opts); readinessErr != nil {
		return readinessErr
	}

	sessions, err = openStickySessionPair(ctx, client, opts)
	if err != nil {
		return err
	}
	defer sessions.gateway.Close()
	defer sessions.backend.Close()

	request := stickyNormalRequest("__STICKY_TOOL_SELECTION__ Search recent weather reports.", tools)
	afterRestart, err := runStickyTurn(ctx, sessions, sessionID, request, headers, "post-restart trusted turn")
	if err != nil {
		return err
	}

	baselineID := sessionID + "-baseline"
	baseline, err := runStickyTurn(
		ctx,
		sessions,
		baselineID,
		request,
		stickyRequestHeaders(baselineID, false),
		"post-restart stateless baseline",
	)
	if err != nil {
		return err
	}
	if err := assertStickySnapshotsEqual(afterRestart, baseline); err != nil {
		return fmt.Errorf("router restart reused local sticky state: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"decision":              stickyToolSelectionRecoveryDecision,
			"pre_restart_tools":     retained.Names,
			"post_restart_tools":    afterRestart.Names,
			"stateless_baseline":    baseline.Names,
			"state_lost_on_restart": true,
			"fallback_request_ok":   true,
		})
	}
	return nil
}
