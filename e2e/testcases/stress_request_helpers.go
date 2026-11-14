package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// StressTestResult tracks the result of a single request of the progressive
// stress case.
type StressTestResult struct {
	RequestID    int
	Success      bool
	StatusCode   int
	Duration     time.Duration
	ErrorMessage string
	// ResponsePath is the Router's x-vsr-response-path, which every response
	// it processed carries, and Decision its x-vsr-selected-decision, if any.
	ResponsePath string
	Decision     string
}

func sendSingleRequest(ctx context.Context, requestID int, localPort string, timeout time.Duration) StressTestResult {
	result := StressTestResult{
		RequestID: requestID,
		Success:   false,
	}

	start := time.Now()

	// Prepare request body with random content
	requestBody := map[string]interface{}{
		"model": "vllm-sr/auto",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": generateRandomContent(requestID),
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("marshal error: %v", err)
		result.Duration = time.Since(start)
		return result
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("create request error: %v", err)
		result.Duration = time.Since(start)
		return result
	}

	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{
		Timeout: timeout,
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("send request error: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	defer resp.Body.Close()
	result.ResponsePath = resp.Header.Get("x-vsr-response-path")
	result.Decision = resp.Header.Get("x-vsr-selected-decision")

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("read response error: %v", err)
		result.Duration = time.Since(start)
		result.StatusCode = resp.StatusCode
		return result
	}

	result.Duration = time.Since(start)
	result.StatusCode = resp.StatusCode

	if resp.StatusCode == http.StatusOK {
		result.Success = true
	} else {
		result.ErrorMessage = fmt.Sprintf("status %d: %s", resp.StatusCode, string(body))
	}

	return result
}
