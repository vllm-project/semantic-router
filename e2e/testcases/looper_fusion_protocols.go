package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The Fusion quorum fallback is a routing decision, not a protocol feature, so
// it must behave identically on every inbound surface the repository declares.
// These cases drive the same below-quorum panel through Anthropic Messages and
// OpenAI Responses; looper_fusion_quorum.go covers Chat Completions.

func init() {
	pkgtestcases.Register("looper-fusion-quorum-fallback-anthropic", pkgtestcases.TestCase{
		Description: "Serve the Fusion fallback for an Anthropic Messages request",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumFallbackAnthropic,
	})
	pkgtestcases.Register("looper-fusion-quorum-fallback-responses", pkgtestcases.TestCase{
		Description: "Serve the Fusion fallback for an OpenAI Responses request",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionQuorumFallbackResponses,
	})
}

// testLooperFusionQuorumFallbackAnthropic drives the same below-quorum fallback
// through the Anthropic Messages surface, so the behaviour is not verified for
// OpenAI Chat Completions alone.
func testLooperFusionQuorumFallbackAnthropic(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	response, err := sendAnthropicMessagesRequest(ctx, anthropicMessagesRequestBody{
		Model:     "MoM",
		MaxTokens: 64,
		Messages: []anthropicMessage{
			{Role: "user", Content: looperFusionProtocolProbeKeyword},
		},
	}, localPort)
	if err != nil {
		return fmt.Errorf("fusion anthropic fallback request failed: %w", err)
	}
	defer response.Body.Close()

	payload, err := io.ReadAll(response.Body)
	if err != nil {
		return fmt.Errorf("read anthropic fallback response: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("anthropic fallback status = %d, want %d: %s",
			response.StatusCode, http.StatusOK, string(payload))
	}
	body := string(payload)
	if !strings.Contains(body, looperFusionFallbackAnswer) {
		return fmt.Errorf("anthropic fallback answer missing: %s", body)
	}
	if strings.Contains(body, looperFusionSynthesizedAnswer) {
		return fmt.Errorf("anthropic fallback unexpectedly contains judge synthesis")
	}

	return counters.requireCounts(ctx, "looper-fusion-quorum-fallback-anthropic", map[string]int{
		looperFusionFallbackModel: 1,
		looperFusionJudgeModel:    0,
	})
}

// testLooperFusionQuorumFallbackResponses completes the supported-protocol
// matrix. The repository declares Chat Completions, OpenAI Responses, and
// Anthropic Messages, so the fallback must behave identically on all three.
func testLooperFusionQuorumFallbackResponses(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	payload, err := json.Marshal(map[string]any{
		"model": "MoM",
		"input": looperFusionProtocolProbeKeyword,
	})
	if err != nil {
		return fmt.Errorf("marshal responses request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("create responses request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-debug", "true")

	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return fmt.Errorf("fusion responses fallback request failed: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read responses fallback body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("responses fallback status = %d, want %d: %s",
			resp.StatusCode, http.StatusOK, string(responseBody))
	}
	body := string(responseBody)
	if err := assertResponsesEnvelope(responseBody); err != nil {
		return err
	}
	if !strings.Contains(body, looperFusionFallbackAnswer) {
		return fmt.Errorf("responses fallback answer missing: %s", body)
	}
	if strings.Contains(body, looperFusionSynthesizedAnswer) {
		return fmt.Errorf("responses fallback unexpectedly contains judge synthesis")
	}
	if strings.Contains(body, `"chat.completion"`) {
		return fmt.Errorf("responses fallback leaked a chat completion object: %s", body)
	}

	return counters.requireCounts(ctx, "looper-fusion-quorum-fallback-responses", map[string]int{
		looperFusionFallbackModel: 1,
		looperFusionJudgeModel:    0,
	})
}

// assertResponsesEnvelope decodes the body as a Responses object rather than
// string-matching it. A chat completion leaking through would still carry the
// fallback text, so only the typed fields establish which surface produced this.
func assertResponsesEnvelope(responseBody []byte) error {
	var decoded fixtures.ResponseAPIResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return fmt.Errorf("decode responses fallback envelope: %w: %s", err, string(responseBody))
	}
	if decoded.Object != "response" {
		return fmt.Errorf("responses fallback object = %q, want %q: %s",
			decoded.Object, "response", string(responseBody))
	}
	if decoded.Status != "completed" {
		return fmt.Errorf("responses fallback status = %q, want %q: %s",
			decoded.Status, "completed", string(responseBody))
	}
	if len(decoded.Output) == 0 {
		return fmt.Errorf("responses fallback returned an empty output array: %s", string(responseBody))
	}
	return nil
}
