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
)

const localChatCompletionsPath = "/v1/chat/completions"

type localChatCompletionResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

func sendLocalChatCompletion(
	ctx context.Context,
	localPort string,
	model string,
	prompt string,
	timeout time.Duration,
) (*localChatCompletionResponse, error) {
	requestBody := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s%s", localPort, localChatCompletionsPath)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	return &localChatCompletionResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		Body:       bodyBytes,
	}, nil
}

func formatUnexpectedChatCompletionStatus(response *localChatCompletionResponse) string {
	var errorMsg strings.Builder
	errorMsg.WriteString(fmt.Sprintf("Unexpected status code: %d\n", response.StatusCode))
	errorMsg.WriteString(fmt.Sprintf("Response body: %s\n", string(response.Body)))
	errorMsg.WriteString("Response headers:\n")
	errorMsg.WriteString(formatResponseHeaders(response.Headers))
	return errorMsg.String()
}

func logUnexpectedChatCompletionStatus(
	verbose bool,
	response *localChatCompletionResponse,
	subject string,
	detailLines ...string,
) {
	if !verbose {
		return
	}

	fmt.Printf("[Test] ✗ HTTP %d Error for %s\n", response.StatusCode, subject)
	for _, detail := range detailLines {
		fmt.Printf("  %s\n", detail)
	}
	fmt.Printf("  Response Headers:\n%s", formatResponseHeaders(response.Headers))
	fmt.Printf("  Response Body: %s\n", string(response.Body))
}
