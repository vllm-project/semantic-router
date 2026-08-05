package testcases

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("agentgateway-full-duplex-multiturn", pkgtestcases.TestCase{
		Description: "Verify agentgateway FullDuplexStreamed preserves a chunked multi-turn request body",
		Tags:        []string{"agentgateway", "gateway", "streaming", "multi-turn"},
		Fn:          testAgentGatewayFullDuplexMultiturn,
	})
}

type boundedChunkReader struct {
	reader *strings.Reader
	max    int
}

func (r *boundedChunkReader) Read(p []byte) (int, error) {
	if len(p) > r.max {
		p = p[:r.max]
	}
	return r.reader.Read(p)
}

func testAgentGatewayFullDuplexMultiturn(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	const (
		namespace = "agentgateway-system"
		svcName   = "agentgateway-proxy"
		localPort = "8081"
		attempts  = 12
	)

	stop, err := helpers.StartPortForward(ctx, client, opts.RestConfig, namespace, svcName, localPort+":80", opts.Verbose)
	if err != nil {
		return fmt.Errorf("start agentgateway port-forward: %w", err)
	}
	defer stop()
	time.Sleep(2 * time.Second)

	// Keep the payload in the size range from issue #2486 and force HTTP/1.1
	// chunked transfer with small reader chunks. Every attempt must reach the
	// mock LLM as valid JSON; an empty intermediate mutation produces a 503.
	history := strings.Repeat("Earlier context that must remain intact across the streamed request. ", 32)
	payload := fmt.Sprintf(`{"model":"auto","messages":[{"role":"user","content":%q},{"role":"assistant","content":%q},{"role":"user","content":"What is the derivative of x cubed?"}],"max_tokens":64,"temperature":0}`,
		history, "I retained the earlier context and am ready for the next question.")
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	for attempt := 1; attempt <= attempts; attempt++ {
		body := &boundedChunkReader{reader: strings.NewReader(payload), max: 47}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
		if err != nil {
			return fmt.Errorf("attempt %d: create request: %w", attempt, err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.ContentLength = -1

		resp, err := httpClient.Do(req)
		if err != nil {
			return fmt.Errorf("attempt %d: send request: %w", attempt, err)
		}
		responseBody, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return fmt.Errorf("attempt %d: read response: %w", attempt, readErr)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("attempt %d: expected HTTP 200, got %d: %s", attempt, resp.StatusCode, responseBody)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"attempts":         attempts,
			"request_bytes":    len(payload),
			"reader_chunk_max": 47,
		})
	}
	return nil
}
