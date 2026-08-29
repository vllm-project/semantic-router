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

package looper

import (
	"fmt"
	"net/http"

	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// maxErrorBodyBytes bounds how much of a non-2xx upstream response is read for
// the diagnostic error message. Error bodies are never parsed, so an oversized
// one is truncated rather than treated as a failure.
const maxErrorBodyBytes int64 = 8 * 1024

// readResponseBody reads and bounds the body of a model-call HTTP response.
// A non-2xx response yields content-free size/truncation diagnostics; a
// success body is read in full up to the configured ceiling and errors
// (rather than silently truncating) when oversized.
func (c *Client) readResponseBody(resp *http.Response) ([]byte, error) {
	if resp.StatusCode != http.StatusOK {
		errBody, truncated := httputil.ReadTruncatedBody(resp.Body, maxErrorBodyBytes)
		return nil, fmt.Errorf(
			"request failed with status %d (error_body_bytes=%d, truncated=%t)",
			resp.StatusCode,
			len(errBody),
			truncated,
		)
	}

	respBody, err := httputil.ReadLimitedBody(resp.Body, c.maxResponseBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	return respBody, nil
}
