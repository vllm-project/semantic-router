/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const (
	llamaStackMaxResponseBodyBytes = 16 << 20
	llamaStackMaxErrorBodyBytes    = 4 << 10
	llamaStackErrorPreviewBytes    = 500
)

// doRequest is the shared HTTP helper for all Llama Stack API calls.
func (l *LlamaStackBackend) doRequest(
	ctx context.Context, method, path string, body interface{},
) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonBytes)
	}

	url := l.endpoint + path
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if l.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+l.authToken)
	}

	resp, err := l.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request to Llama Stack failed (%s %s): %w", method, path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _, readErr := readLimitedBody(resp.Body, llamaStackMaxErrorBodyBytes)
		if readErr != nil {
			return nil, fmt.Errorf("failed to read Llama Stack response body: %w", readErr)
		}
		errMsg := string(respBody)
		if len(errMsg) > llamaStackErrorPreviewBytes {
			errMsg = errMsg[:llamaStackErrorPreviewBytes] + "..."
		}
		return nil, fmt.Errorf("llama stack API error: %s %s returned status %d: %s",
			method, path, resp.StatusCode, errMsg)
	}

	respBody, exceeded, err := readLimitedBody(resp.Body, llamaStackMaxResponseBodyBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read Llama Stack response body: %w", err)
	}
	if exceeded {
		return nil, fmt.Errorf(
			"llama stack response body exceeded %d bytes for %s %s",
			llamaStackMaxResponseBodyBytes,
			method,
			path,
		)
	}

	return respBody, nil
}

func readLimitedBody(r io.Reader, maxBytes int64) ([]byte, bool, error) {
	body, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) > maxBytes {
		return body[:maxBytes], true, nil
	}
	return body, false, nil
}
