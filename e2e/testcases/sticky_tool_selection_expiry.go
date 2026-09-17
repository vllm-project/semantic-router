package testcases

import (
	"context"
	"fmt"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const stickyToolSelectionLocalTTL = 30 * time.Second

func init() {
	pkgtestcases.Register("sticky-tool-selection-expiry", pkgtestcases.TestCase{
		Description: "Verify expired local sticky state falls back to current stateless tool selection",
		Tags:        []string{"kubernetes", "plugin", "tool-selection", "sticky", "ttl", "expiry"},
		Fn:          testStickyToolSelectionExpiry,
	})
}

func testStickyToolSelectionExpiry(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] sticky tool-selection local TTL expiry")
	}

	sessions, err := openStickySessionPair(ctx, client, opts)
	if err != nil {
		return err
	}
	defer sessions.gateway.Close()
	defer sessions.backend.Close()

	sessionID := fmt.Sprintf("sticky-expiry-%d", time.Now().UnixNano())
	headers := stickyRequestHeaders(sessionID, true)
	tools := stickyContractTools(1, "")
	_, retained, err := runStickyTrustedTurns(ctx, sessions, sessionID, headers, tools)
	if err != nil {
		return err
	}
	if len(retained.Tools) != stickyToolSelectionMaxTools {
		return fmt.Errorf("TTL expiry precondition retained %d tools, want %d", len(retained.Tools), stickyToolSelectionMaxTools)
	}

	if err := waitForStickyToolSelectionExpiry(ctx); err != nil {
		return err
	}

	request := stickyNormalRequest("__STICKY_TOOL_SELECTION__ Search recent weather reports.", tools)
	afterExpiry, err := runStickyTurn(ctx, sessions, sessionID, request, headers, "post-expiry trusted turn")
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
		"post-expiry stateless baseline",
	)
	if err != nil {
		return err
	}
	if err := assertStickySnapshotsEqual(afterExpiry, baseline); err != nil {
		return fmt.Errorf("expired local sticky state was reused: %w", err)
	}
	if err := assertStickySnapshotsDifferent(retained, afterExpiry); err != nil {
		return fmt.Errorf("TTL expiry did not discard retained state: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"ttl_seconds":        int(stickyToolSelectionLocalTTL / time.Second),
			"pre_expiry_tools":   retained.Names,
			"post_expiry_tools":  afterExpiry.Names,
			"stateless_baseline": baseline.Names,
		})
	}
	return nil
}

func waitForStickyToolSelectionExpiry(ctx context.Context) error {
	timer := time.NewTimer(stickyToolSelectionLocalTTL + 2*time.Second)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
