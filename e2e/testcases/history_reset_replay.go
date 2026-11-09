package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("history-reset-replay-receipts", pkgtestcases.TestCase{
		Description: "Router Replay records the content-free history_reset receipt for an applied change and a kept continuation",
		Tags:        []string{"router-replay", "functional", "history-reset"},
		Fn:          testHistoryResetReplayReceipts,
	})
}

const historyResetReplayModel = "vllm-sr/history-reset"

type historyResetReplayCase struct {
	name string
	live string
	want historyResetReplayReceipt
	// removes reports whether the marked prior turn is removed. Removed
	// content must appear nowhere in the record; a retained turn is still
	// part of the request, which Replay may record like any other.
	removes bool
}

func historyResetReplayCases() []historyResetReplayCase {
	base := historyResetReplayReceipt{Signal: "topic_boundary", Scope: "eligible_history", ExaminedMessages: 3}
	change := base
	change.TriggerClass, change.Outcome, change.Reason = "change", "applied", "applied"
	change.RetainedMessages, change.RemovedMessages, change.RemovedTurns = 1, 2, 1
	change.RecoveryStatus = "not_required"
	continuation := base
	continuation.TriggerClass, continuation.Outcome, continuation.Reason = "continuation", "skipped", "evidence_continuation"
	continuation.RetainedMessages = 3
	return []historyResetReplayCase{
		{name: "change", live: "Unrelated question: how do I renew a passport?", want: change, removes: true},
		{name: "continuation", live: "As you said earlier, it is sunny in Paris; should I bring a jacket tonight?", want: continuation},
	}
}

func testHistoryResetReplayReceipts(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()
	apiSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open Router management API session: %w", err)
	}
	defer apiSession.Close()

	for _, test := range historyResetReplayCases() {
		// Unique content proves the receipt never carries the conversation.
		marker := fmt.Sprintf("zqreplaymarker%d", time.Now().UnixNano())
		sessionID := historyResetObservationID("replay-" + test.name)
		response, err := historyResetPost(ctx, session, "/v1/chat/completions", map[string]any{
			"model": historyResetReplayModel,
			"messages": []any{
				map[string]any{"role": "user", "content": "What is the weather in Paris today? " + marker},
				map[string]any{"role": "assistant", "content": "It is sunny in Paris, 21C."},
				map[string]any{"role": "user", "content": test.live},
			},
		}, map[string]string{"x-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("%s: %w", test.name, err)
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("%s: status %d", test.name, response.StatusCode)
		}
		record, err := historyResetReplayRecord(ctx, apiSession, sessionID, routerReplayDetailToken)
		if err != nil {
			return fmt.Errorf("%s: %w", test.name, err)
		}
		receipt, err := historyResetReceiptFrom(record)
		if err != nil {
			return fmt.Errorf("%s: %w", test.name, err)
		}
		if receipt != test.want {
			return fmt.Errorf("%s: receipt %+v, want %+v", test.name, receipt, test.want)
		}
		if receipt, err := historyResetRawReceipt(record); err != nil || len(historyResetLeakPaths(receipt, marker)) > 0 {
			return fmt.Errorf("%s: the history_reset receipt carries conversation content", test.name)
		}
		if test.removes {
			if paths := historyResetLeakPaths(record, marker); len(paths) > 0 {
				return fmt.Errorf("%s: the replay record contains removed content at %v", test.name, paths)
			}
		}
		if opts.Verbose {
			fmt.Printf("[Test] history reset %s receipt: %+v\n", test.name, receipt)
		}
	}
	return nil
}
