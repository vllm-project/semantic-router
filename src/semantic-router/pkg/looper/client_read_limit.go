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
	"io"
	"net/http"
)

// maxErrorBodyBytes bounds how much of a non-2xx upstream response is read for
// the diagnostic error message. Error bodies are never parsed, so an oversized
// one is truncated rather than treated as a failure.
const maxErrorBodyBytes int64 = 8 * 1024

// readLimitedBody reads at most maxBytes from r. If the stream exceeds
// maxBytes it returns an error rather than a silently truncated body, so an
// oversized or malicious upstream response cannot be mis-parsed or exhaust
// memory (amplified N-fold by the parallel fan-out algorithms).
//
// maxBytes must be positive; a non-positive ceiling is a caller bug (callers
// resolve their default via config.LooperConfig.GetMaxResponseBytes) and is
// rejected explicitly so it can never silently disable the guard or overflow
// the maxBytes+1 below.
func readLimitedBody(r io.Reader, maxBytes int64) ([]byte, error) {
	if maxBytes <= 0 {
		return nil, fmt.Errorf("read limit must be positive, got %d bytes", maxBytes)
	}
	// Read one byte past the cap so an exactly-at-cap body is accepted while an
	// over-cap body is detectable.
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("response body exceeds limit of %d bytes", maxBytes)
	}
	return data, nil
}

// readResponseBody reads and bounds the body of a model-call HTTP response.
// A non-2xx response yields an error carrying a small, truncation-marked
// diagnostic prefix; a success body is read in full up to the configured
// ceiling and errors (rather than silently truncating) when oversized.
func (c *Client) readResponseBody(resp *http.Response) ([]byte, error) {
	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes+1))
		msg := string(errBody)
		if int64(len(errBody)) > maxErrorBodyBytes {
			msg = string(errBody[:maxErrorBodyBytes]) + "...(truncated)"
		}
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, msg)
	}

	respBody, err := readLimitedBody(resp.Body, c.maxResponseBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	return respBody, nil
}
