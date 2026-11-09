package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("history-reset-failure-modes", pkgtestcases.TestCase{
		Description: "history_reset fail_closed rejects before dispatch, serves opening turns, and handles unavailable recovery per failure mode",
		Tags:        []string{"plugin", "context", "history-reset"},
		Fn:          testHistoryResetFailureModes,
	})
}

const (
	historyResetFailClosedProbe     = "__HISTORY_RESET_FAIL_CLOSED_PROBE__"
	historyResetRecoveryOpenProbe   = "__HISTORY_RESET_RECOVERY_OPEN_PROBE__"
	historyResetRecoveryClosedProbe = "__HISTORY_RESET_RECOVERY_CLOSED_PROBE__"

	// The envoy-ai-gateway profile's provider mocker.
	historyResetGatewayBackendNamespace = "default"
	historyResetGatewayBackendService   = "vllm-llama3-8b-instruct"
	historyResetGatewayBackendPort      = "8000"
)

type historyResetFailureCase struct {
	name string
	// live is the final user turn; prior turns come first when withPrior is set.
	live      string
	withPrior bool
	// status is the client-visible status; a rejection must never reach the
	// provider, and a served request must reach it with upstream messages.
	status   int
	upstream int
}

// Each probe keyword appears only in the live turn, so it is not shared
// vocabulary with the prior turn.
func historyResetFailureCases() []historyResetFailureCase {
	return []historyResetFailureCase{
		{
			// Unknown evidence on a follow-up with removable history.
			name:      "fail_closed inconclusive follow-up",
			live:      "How do I renew a passport? " + historyResetFailClosedProbe,
			withPrior: true,
			status:    http.StatusServiceUnavailable,
		},
		{
			// Nothing is removable, so the unknown result is never judged.
			name:     "fail_closed opening turn",
			live:     "How do I renew a passport? " + historyResetFailClosedProbe,
			status:   http.StatusOK,
			upstream: 1,
		},
		{
			// The memory-backed cache has no recovery store.
			name:      "fail_open recovery unavailable",
			live:      "Unrelated question: how do I renew a passport? " + historyResetRecoveryOpenProbe,
			withPrior: true,
			status:    http.StatusOK,
			upstream:  3,
		},
		{
			name:      "fail_closed recovery unavailable",
			live:      "Unrelated question: how do I renew a passport? " + historyResetRecoveryClosedProbe,
			withPrior: true,
			status:    http.StatusServiceUnavailable,
		},
	}
}

func testHistoryResetFailureModes(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	backend, err := fixtures.OpenServiceEndpointSession(ctx, client, opts,
		historyResetGatewayBackendNamespace, historyResetGatewayBackendService, historyResetGatewayBackendPort)
	if err != nil {
		return err
	}
	defer backend.Close()

	details := map[string]interface{}{}
	for _, test := range historyResetFailureCases() {
		messages := []map[string]string{}
		if test.withPrior {
			messages = append(messages,
				map[string]string{"role": "user", "content": "What is the weather in Paris today?"},
				map[string]string{"role": "assistant", "content": "It is sunny in Paris, 21C."},
			)
		}
		messages = append(messages, map[string]string{"role": "user", "content": test.live})

		observationID := historyResetObservationID(strings.ReplaceAll(test.name, " ", "-"))
		response, err := historyResetPost(ctx, session, "/v1/chat/completions",
			map[string]interface{}{"model": "vllm-sr/auto", "messages": messages},
			map[string]string{historyResetObservationHeader: observationID})
		if err != nil {
			return fmt.Errorf("%s: %w", test.name, err)
		}
		if response.StatusCode != test.status {
			return fmt.Errorf("%s: status %d, want %d", test.name, response.StatusCode, test.status)
		}
		observed, dispatched, err := historyResetObserve(ctx, backend, observationID)
		if err != nil {
			return fmt.Errorf("%s: %w", test.name, err)
		}
		if test.status != http.StatusOK {
			if dispatched {
				return fmt.Errorf("%s: a rejected request reached the provider", test.name)
			}
			details[test.name] = "rejected before dispatch"
			continue
		}
		if !dispatched {
			return fmt.Errorf("%s: a served request was never dispatched", test.name)
		}
		upstream := historyResetMessages(observed)
		if len(upstream) != test.upstream {
			return fmt.Errorf("%s: provider received %d messages, want %d", test.name, len(upstream), test.upstream)
		}
		if !strings.Contains(historyResetMessageText(upstream[len(upstream)-1]), test.live) {
			return fmt.Errorf("%s: the live turn was not the last provider-bound message", test.name)
		}
		details[test.name] = len(upstream)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	return nil
}
