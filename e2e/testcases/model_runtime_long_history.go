package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-long-history", pkgtestcases.TestCase{
		Description: "A conversation whose history has more PII pieces than a runtime takes tasks in one bundle by default is answered with no task refused and every piece reaching the PII model; the Router's managed runtimes advertise a bundle cap that holds every history piece and take them in one bundle, and the PII signal over the whole history follows the runtime's spans",
		Tags:        []string{"model-runtime", "bundles", "pii", "history"},
		Fn:          testModelRuntimeLongHistory,
	})
}

const (
	// More distinct assistant turns than the runtime's default of 64 tasks per
	// bundle; the PII rule reads each of them (include_history in values.yaml).
	mrHistoryTurns = 80
	// vllm-srun's --max-bundle-tasks default.
	mrDefaultBundleTasks = 64
	// One lookup per input window a model is asked, hit or miss. Bundle tasks
	// don't count pieces: the Router fuses a stage's calls to one model and
	// head into one task of many inputs.
	runtimeResultCacheMetric = "vllm_srun_result_cache_total"
)

func testModelRuntimeLongHistory(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrManagedDeployments...); err != nil {
		return err
	}
	pii, socket, err := session.managed(ctx, mrPIIDeployment)
	if err != nil {
		return err
	}
	messages, pieces := longHistory(fmt.Sprintf("%d", time.Now().UnixNano()))
	if len(pieces) <= mrDefaultBundleTasks {
		return fmt.Errorf("the conversation has %d PII pieces; it must exceed %d", len(pieces), mrDefaultBundleTasks)
	}

	before, err := pii.Metrics(ctx)
	if err != nil {
		return err
	}
	response, err := sendLocalChatConversation(ctx, session.gatewayPort, "vllm-sr/auto", messages, mrRequestTimeout)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("long-history chat: %s", formatUnexpectedChatCompletionStatus(response))
	}
	after, err := pii.Metrics(ctx)
	if err != nil {
		return err
	}
	bundles := countDelta(before, after, runtimeRequestsMetric, map[string]string{"endpoint": "/v1/bundle"})
	refused := countDelta(before, after, runtimeRequestsMetric, map[string]string{"status": "413"})
	routed := after.Sum(runtimeBundleTasksMetric+"_sum", nil) - before.Sum(runtimeBundleTasksMetric+"_sum", nil)
	asked := countDelta(before, after, runtimeResultCacheMetric, map[string]string{"model": mrPIIDeployment})
	if refused != 0 {
		return fmt.Errorf("%s refused %v of the Router's requests as too large", socket, refused)
	}
	if bundles < 1 || asked < float64(len(pieces)) {
		return fmt.Errorf("%s received %v bundles carrying %v tasks and asked %s %v inputs, want the %d PII pieces at least",
			socket, bundles, routed, mrPIIDeployment, asked, len(pieces))
	}

	preview, err := session.previewConversation(ctx, messages)
	if err != nil {
		return err
	}
	if err = onlyOfflineSignalError(preview.SignalErrors); err != nil {
		return err
	}

	// How the Router's flushes group a stage depends on timing, so one bundle
	// of every piece checks the cap the Router starts its runtimes with.
	served, err := pii.Models(ctx)
	if err != nil {
		return err
	}
	if served.Limits == nil || served.Limits.MaxBundleTasks < len(pieces) {
		return fmt.Errorf("%s advertises limits %+v; a managed process must take the %d pieces in one bundle", socket, served.Limits, len(pieces))
	}
	threshold := mrSignalThreshold
	options := &modelruntime.ClassifyOptions{
		Overflow: "window", MaxTokens: mrWindowBudget, Threshold: &threshold,
		Window: &modelruntime.WindowOptions{Tokens: mrWindowTokens, Overlap: mrWindowOverlap},
	}
	tasks := make([]modelruntime.BundleTask, len(pieces))
	for index, piece := range pieces {
		tasks[index] = modelruntime.BundleTask{
			ID:       fmt.Sprintf("piece-%d", index),
			Classify: &modelruntime.ClassifyRequest{Model: mrPIIDeployment, Input: []string{piece}, Options: options},
		}
	}
	bundled, err := pii.Bundle(ctx, modelruntime.BundleRequest{Tasks: tasks})
	if err != nil {
		return fmt.Errorf("one /v1/bundle of %d PII tasks to %s: %w", len(tasks), socket, err)
	}
	if len(bundled.Results) != len(tasks) {
		return fmt.Errorf("%s answered %d of %d bundled tasks", socket, len(bundled.Results), len(tasks))
	}
	withSpans := 0
	for index, result := range bundled.Results {
		if result.Status != http.StatusOK || result.Classify == nil || len(result.Classify.Results) != 1 || result.Classify.Results[0].Error != "" {
			return fmt.Errorf("%s bundled task %d answered status %d: %+v", socket, index, result.Status, result.Error)
		}
		if len(result.Classify.Results[0].Spans) > 0 {
			withSpans++
		}
	}
	if matched := headerItems(response.Headers, "x-vsr-matched-pii")["personal_data"]; matched != (withSpans > 0) {
		return fmt.Errorf("personal_data matched=%v, the PII model found spans in %d of %d pieces", matched, withSpans, len(pieces))
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"pii_pieces": len(pieces), "pieces_with_spans": withSpans, "router_bundles": bundles,
			"router_bundled_tasks": routed, "pii_inputs_asked": asked, "direct_bundle_tasks": len(tasks),
			"advertised_max_bundle_tasks": served.Limits.MaxBundleTasks, "decision": response.Headers.Get("x-vsr-selected-decision"),
		})
	}
	return nil
}

// longHistory returns a conversation of mrHistoryTurns distinct assistant
// turns and a closing user question, and the texts the PII rule classifies.
func longHistory(nonce string) ([]map[string]string, []string) {
	var messages []map[string]string
	var pieces []string
	for turn := 1; turn <= mrHistoryTurns; turn++ {
		content := fmt.Sprintf("Note %d (%s): ticket %d is assigned to desk B-%d.", turn, nonce, 4100+turn, turn)
		messages = append(messages, map[string]string{"role": "assistant", "content": content})
		pieces = append(pieces, content)
	}
	question := fmt.Sprintf("Which desk handles ticket 4142? (%s)", nonce)
	messages = append(messages, map[string]string{"role": "user", "content": question})
	return messages, append([]string{question}, pieces...)
}
