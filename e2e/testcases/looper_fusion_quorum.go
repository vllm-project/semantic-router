package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	looperFusionQuorumProbeKeyword          = "__LOOPER_FUSION_QUORUM_PROBE__"
	looperFusionFallbackProbeKeyword        = "__LOOPER_FUSION_FALLBACK_PROBE__"
	looperFusionZeroUsableProbeKeyword      = "__LOOPER_FUSION_ZERO_USABLE_PROBE__"
	looperFusionFallbackFailureProbeKeyword = "__LOOPER_FUSION_FALLBACK_FAILURE_PROBE__"
	looperFusionBudgetProbeKeyword          = "__LOOPER_FUSION_BUDGET_PROBE__"
	looperFusionTinyWindowAnswer            = "fusion-tiny-window-dispatched"
	looperFusionDeadlineProbeKeyword        = "__LOOPER_FUSION_DEADLINE_PROBE__"
	looperFusionProtocolProbeKeyword        = "__LOOPER_FUSION_PROTOCOL_PROBE__"
	looperFusionSynthesizedAnswer           = "unexpected-fusion-synthesized-answer"
	looperFusionFallbackAnswer              = "fusion-quorum-fallback-answer"
)

// assertFallbackIsSelectedModel proves the client-visible routing facts name the
// model that actually answered.
//
// Runtime model selection places a panel candidate in the request context before
// the looper runs, and the looper's routing facts must overwrite it before the
// response headers are built. A body-only assertion cannot see that ordering.
func assertFallbackIsSelectedModel(response *localChatCompletionResponse, want, testName string) error {
	selected := response.Headers.Get("x-vsr-selected-model")
	if selected == "" {
		return fmt.Errorf("%s: x-vsr-selected-model header missing", testName)
	}
	if selected != want {
		return fmt.Errorf("%s: x-vsr-selected-model = %q, want %q (a panel candidate must not be advertised as the responder)",
			testName, selected, want)
	}
	return nil
}

// assertNoFusionExtension proves a served fallback returns the fallback target's
// ordinary response and nothing more.
//
// A served fallback bypasses the judge, so there is no deliberation to report.
// Its quorum evidence travels to Replay and metrics instead. Asserting only on
// the answer text and the backend counters would not catch a regression that
// re-attached the trace, and that trace previously carried provider error text.
func assertNoFusionExtension(body []byte, testName string) error {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		return fmt.Errorf("%s: response is not a JSON object: %w", testName, err)
	}
	if _, present := envelope["fusion"]; present {
		return fmt.Errorf("%s: served fallback carried a top-level fusion extension: %s",
			testName, string(body))
	}
	return nil
}

// assertFusionPlanExcludesJudge checks the client-visible routing metadata does
// not advertise the judge.
//
// This is metadata, not an execution trace. The router now builds
// x-vsr-looper-models-used from the panel models it dispatched, but the header
// still cannot prove a call was *not* made: a judge call whose output was
// discarded would leave it unchanged. fusionBackendCounters is the oracle for
// that; this only pins what the client is told.
func assertFusionPlanExcludesJudge(response *localChatCompletionResponse, testName string) error {
	modelsUsed := response.Headers.Get("x-vsr-looper-models-used")
	if modelsUsed == "" {
		return fmt.Errorf("%s: x-vsr-looper-models-used header missing; cannot verify judge suppression", testName)
	}
	for _, model := range strings.Split(modelsUsed, ",") {
		if strings.TrimSpace(model) == looperFusionJudgeModel {
			return fmt.Errorf("%s: judge advertised to the client on a below-quorum panel (models used: %s)", testName, modelsUsed)
		}
	}
	return nil
}

type looperFusionErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func init() {
	pkgtestcases.Register("looper-fusion-usable-quorum", pkgtestcases.TestCase{
		Description: "Reject Fusion synthesis when usable panel responses do not meet quorum",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionUsableQuorum,
	})
	pkgtestcases.Register("looper-fusion-quorum-fallback", pkgtestcases.TestCase{
		Description: "Serve the configured fallback target when the Fusion panel misses quorum",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumFallback,
	})
	pkgtestcases.Register("looper-fusion-quorum-zero-usable", pkgtestcases.TestCase{
		Description: "Serve the fallback target when no Fusion panel response is usable",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumZeroUsable,
	})
	pkgtestcases.Register("looper-fusion-quorum-fallback-failure", pkgtestcases.TestCase{
		Description: "Report a typed failure when the configured Fusion fallback target also fails",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumFallbackFailure,
	})
	pkgtestcases.Register("looper-fusion-quorum-budget-exhausted", pkgtestcases.TestCase{
		Description: "Refuse a Fusion fallback target that no longer fits the remaining context budget",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumBudgetExhausted,
	})
	pkgtestcases.Register("looper-fusion-quorum-deadline-cancellation", pkgtestcases.TestCase{
		Description: "Cancel a Fusion panel at its round deadline and serve the configured fallback",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumDeadlineCancellation,
	})
}

// testLooperFusionQuorumDeadlineCancellation covers the internal round timeout,
// not caller cancellation. The panel's round_timeout_seconds fires while a slow
// model is still in flight, so the panel ends below quorum and the fallback
// answers the still-connected client. The deadline is configuration-driven,
// which makes the branch deterministic.
//
// looper-fusion-quorum-caller-cancellation covers the other contract, where the
// caller disconnects and the fallback must be skipped.
func testLooperFusionQuorumDeadlineCancellation(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionDeadlineProbeKeyword, 60*time.Second)
	if err != nil {
		return fmt.Errorf("fusion deadline request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-quorum-deadline-cancellation")
		return fmt.Errorf("fusion deadline status = %d, want %d", response.StatusCode, http.StatusOK)
	}
	body := string(response.Body)
	if !strings.Contains(body, looperFusionFallbackAnswer) {
		return fmt.Errorf("fallback answer %q missing after panel deadline: %s", looperFusionFallbackAnswer, body)
	}
	// The slow panel model resolves after the deadline; its content must never
	// reach the client, and the judge must not have run.
	if strings.Contains(body, "too late") {
		return fmt.Errorf("a post-deadline panel response was served to the client")
	}
	if strings.Contains(body, looperFusionSynthesizedAnswer) {
		return fmt.Errorf("judge synthesis ran on a cancelled below-quorum panel")
	}
	if err := assertFusionPlanExcludesJudge(response, "looper-fusion-quorum-deadline-cancellation"); err != nil {
		return err
	}
	if err := assertNoFusionExtension(response.Body, "looper-fusion-quorum-deadline-cancellation"); err != nil {
		return err
	}
	// The caller is still connected here, so the fallback must run. This is what
	// separates the internal round timeout from caller cancellation, where the
	// fallback must not run at all.
	return counters.requireCounts(ctx, "looper-fusion-quorum-deadline-cancellation", map[string]int{
		looperFusionSlowPanelModel: 1,
		looperFusionFallbackModel:  1,
		looperFusionJudgeModel:     0,
	})
}

// testLooperFusionQuorumBudgetExhausted covers the budget-exhaustion branch. The
// fallback target declares a context window too small for the request, so the
// stage gate must refuse it rather than dispatching a call that cannot succeed.
func testLooperFusionQuorumBudgetExhausted(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionBudgetProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("fusion budget request failed: %w", err)
	}
	if response.StatusCode != http.StatusInternalServerError {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-quorum-budget-exhausted")
		return fmt.Errorf("fusion budget status = %d, want %d", response.StatusCode, http.StatusInternalServerError)
	}
	body := string(response.Body)
	if strings.Contains(body, looperFusionSynthesizedAnswer) {
		return fmt.Errorf("budget exhaustion unexpectedly returned judge synthesis")
	}
	// The tiny-window backend answers successfully by design. Seeing its answer
	// means the stage budget gate did not refuse the dispatch, which a bare
	// 500-plus-absent-strings assertion would not have caught.
	if strings.Contains(body, looperFusionTinyWindowAnswer) {
		return fmt.Errorf("budget gate regressed: fallback target was dispatched")
	}
	// The body check above can only see a fallback whose answer was returned.
	// The counter proves the call was never made at all.
	return counters.requireCounts(ctx, "looper-fusion-quorum-budget-exhausted", map[string]int{
		looperFusionTinyWindowModel: 0,
		looperFusionJudgeModel:      0,
	})
}

// testLooperFusionQuorumZeroUsable covers the zero-usable-response branch: every
// panel member fails or returns an empty payload, so there is nothing to
// synthesize from and the fallback must answer instead.
func testLooperFusionQuorumZeroUsable(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionZeroUsableProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("fusion zero-usable request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-quorum-zero-usable")
		return fmt.Errorf("fusion zero-usable status = %d, want %d", response.StatusCode, http.StatusOK)
	}
	body := string(response.Body)
	if !strings.Contains(body, looperFusionFallbackAnswer) {
		return fmt.Errorf("fallback answer %q missing from response: %s", looperFusionFallbackAnswer, body)
	}
	if strings.Contains(body, looperFusionSynthesizedAnswer) {
		return fmt.Errorf("zero-usable response unexpectedly contains judge synthesis %q", looperFusionSynthesizedAnswer)
	}
	if err := assertFusionPlanExcludesJudge(response, "looper-fusion-quorum-zero-usable"); err != nil {
		return err
	}
	if err := assertNoFusionExtension(response.Body, "looper-fusion-quorum-zero-usable"); err != nil {
		return err
	}

	return counters.requireCounts(ctx, "looper-fusion-quorum-zero-usable", map[string]int{
		looperFusionFallbackModel: 1,
		looperFusionJudgeModel:    0,
	})
}

// testLooperFusionQuorumFallbackFailure covers the fallback-failure branch. The
// request must surface a typed error rather than degrading to judge synthesis
// over the under-strength panel.
func testLooperFusionQuorumFallbackFailure(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionFallbackFailureProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("fusion fallback-failure request failed: %w", err)
	}
	if response.StatusCode != http.StatusInternalServerError {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-quorum-fallback-failure")
		return fmt.Errorf("fusion fallback-failure status = %d, want %d", response.StatusCode, http.StatusInternalServerError)
	}

	var errorResponse looperFusionErrorResponse
	if err := json.Unmarshal(response.Body, &errorResponse); err != nil {
		return fmt.Errorf("decode fusion fallback-failure error: %w", err)
	}
	if errorResponse.Error.Type != "server_error" {
		return fmt.Errorf("error.type = %q, want server_error: %s", errorResponse.Error.Type, string(response.Body))
	}
	if strings.Contains(string(response.Body), looperFusionSynthesizedAnswer) {
		return fmt.Errorf("fallback failure unexpectedly returned synthesized answer %q", looperFusionSynthesizedAnswer)
	}
	if strings.Contains(string(response.Body), looperFusionFallbackAnswer) {
		return fmt.Errorf("fallback failure unexpectedly returned a fallback answer")
	}
	// The broken target must have been attempted exactly once, and the judge not
	// at all: a failed fallback must not silently degrade into synthesis.
	return counters.requireCounts(ctx, "looper-fusion-quorum-fallback-failure", map[string]int{
		looperFusionBrokenFallbackModel: 1,
		looperFusionJudgeModel:          0,
	})
}

// testLooperFusionQuorumFallback covers the opposite branch of the same panel
// shape as looper-fusion-usable-quorum: one usable response against a quorum of
// two, but with quorum_failure_policy=fallback configured. The request must
// succeed with the fallback target's answer, and must never contain judge
// synthesis over the under-strength panel.
func testLooperFusionQuorumFallback(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionFallbackProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("fusion quorum fallback request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-quorum-fallback")
		return fmt.Errorf("fusion quorum fallback status = %d, want %d", response.StatusCode, http.StatusOK)
	}
	body := string(response.Body)
	if !strings.Contains(body, looperFusionFallbackAnswer) {
		return fmt.Errorf("fallback answer %q missing from response: %s", looperFusionFallbackAnswer, body)
	}
	if strings.Contains(body, looperFusionSynthesizedAnswer) {
		return fmt.Errorf("fallback response unexpectedly contains judge synthesis %q", looperFusionSynthesizedAnswer)
	}
	if err := assertFusionPlanExcludesJudge(response, "looper-fusion-quorum-fallback"); err != nil {
		return err
	}
	if err := assertNoFusionExtension(response.Body, "looper-fusion-quorum-fallback"); err != nil {
		return err
	}
	if err := assertFallbackIsSelectedModel(
		response, "fusion-fallback-target", "looper-fusion-quorum-fallback"); err != nil {
		return err
	}

	return counters.requireCounts(ctx, "looper-fusion-quorum-fallback", map[string]int{
		looperFusionFallbackModel: 1,
		looperFusionJudgeModel:    0,
	})
}

func testLooperFusionUsableQuorum(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionQuorumProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("fusion quorum request failed: %w", err)
	}
	if response.StatusCode != http.StatusInternalServerError {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-usable-quorum")
		return fmt.Errorf("fusion quorum status = %d, want %d", response.StatusCode, http.StatusInternalServerError)
	}

	var errorResponse looperFusionErrorResponse
	if err := json.Unmarshal(response.Body, &errorResponse); err != nil {
		return fmt.Errorf("decode fusion quorum error: %w", err)
	}
	if errorResponse.Error.Type != "server_error" {
		return fmt.Errorf("error.type = %q, want server_error: %s", errorResponse.Error.Type, string(response.Body))
	}
	if strings.TrimSpace(errorResponse.Error.Message) == "" {
		return fmt.Errorf("error.message is empty: %s", string(response.Body))
	}
	if strings.Contains(string(response.Body), looperFusionSynthesizedAnswer) {
		return fmt.Errorf("fusion quorum error unexpectedly returned synthesized answer %q", looperFusionSynthesizedAnswer)
	}
	// This decision configures no fallback, so a below-quorum panel must dispatch
	// neither the judge nor any fallback target.
	return counters.requireCounts(ctx, "looper-fusion-usable-quorum", map[string]int{
		looperFusionJudgeModel:    0,
		looperFusionFallbackModel: 0,
	})
}
