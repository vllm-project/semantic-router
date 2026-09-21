package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("hallucination-streaming-observation", pkgtestcases.TestCase{
		Description: "Verify a streamed answer is checked by the hallucination rule and still delivered in full, because a streamed answer is observed and never enforced",
		Tags:        []string{"kubernetes", "hallucination", "streaming"},
		Fn:          testHallucinationStreamingObservation,
	})
}

// testHallucinationStreamingObservation pins the streaming half of the
// hallucination contract: the rule checks the answer the stream reconstructed
// and the check is counted, while the plugin takes no action on it. The answer
// arrives in full and carries no warning header, because its bytes were with
// the client before the answer existed as a whole.
func testHallucinationStreamingObservation(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing that a streamed answer is checked for hallucination and still passes through")
	}

	testCases, err := loadHallucinationCases("e2e/testcases/testdata/hallucination_detection_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}
	if len(testCases) == 0 {
		return fmt.Errorf("no hallucination test cases to stream")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open metrics session: %w", err)
	}
	defer metricsSession.Close()

	checkedBefore, err := readSignalExtractionCount(ctx, metricsSession, "hallucination", hallucinationRuleName)
	if err != nil {
		return fmt.Errorf("read %s before the streamed request: %w", signalExtractionMetric, err)
	}

	resp, err := sendHallucinationChatCompletion(ctx, localPort, testCases[0], 60*time.Second, true)
	if err != nil {
		return fmt.Errorf("streaming request failed: %w", err)
	}
	content, frames := chatStreamedContent(resp.Body)
	warnings := resp.Headers.Get(responseWarningsHeader)

	checkedAfter, err := readSignalExtractionCount(ctx, metricsSession, "hallucination", hallucinationRuleName)
	if err != nil {
		return fmt.Errorf("read %s after the streamed request: %w", signalExtractionMetric, err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":    resp.StatusCode,
			"frames":         frames,
			"content_bytes":  len(content),
			"warnings":       warnings,
			"checked_before": checkedBefore,
			"checked_after":  checkedAfter,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] status=%d frames=%d content_bytes=%d warnings=%q checked=%v->%v\n",
			resp.StatusCode, frames, len(content), warnings, checkedBefore, checkedAfter)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected the streamed answer to be delivered, got %d: %s", resp.StatusCode, truncateString(string(resp.Body), 400))
	}
	if content == "" {
		return fmt.Errorf("the streamed answer arrived empty over %d frames: there was nothing to check", frames)
	}
	if checkedAfter < checkedBefore+1 {
		return fmt.Errorf("the streamed answer was not checked by rule %q: %s went %v to %v",
			hallucinationRuleName, signalExtractionMetric, checkedBefore, checkedAfter)
	}
	if strings.Contains(warnings, hallucinationWarningCode) {
		return fmt.Errorf("a streamed answer carried %s = %q; the bytes were already delivered, so the plugin cannot have acted on the detection",
			responseWarningsHeader, warnings)
	}

	return nil
}
