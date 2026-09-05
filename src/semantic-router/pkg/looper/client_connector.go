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
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

type modelConnector interface {
	DoWithHeaders(context.Context, connector.Operation, []byte, http.Header) ([]byte, error)
	Close() error
}

// maxErrorBodyBytes limits the diagnostic body retained for non-2xx responses.
const maxErrorBodyBytes int64 = 8 * 1024

var chatCompletionOperation = connector.Operation{
	Name:              "looper_chat_completion",
	Method:            http.MethodPost,
	Path:              "",
	SuccessStatusCode: http.StatusOK,
	RetrySafe:         false,
}

// NewConnectorClient creates a Looper client backed by the shared remote
// connector. Generative calls are intentionally configured without retries.
func NewConnectorClient(cfg *config.LooperConfig) (*Client, error) {
	if cfg == nil {
		return nil, fmt.Errorf("looper config is required")
	}
	// Looper has no separate request-size setting. Reuse its existing response
	// ceiling to bound the other buffered side without expanding config schema.
	bodyLimit := cfg.GetMaxResponseBytes()
	remote, err := connector.New(cfg.Endpoint, nil, connector.Options{
		AttemptTimeout:   time.Duration(cfg.GetTimeout()) * time.Second,
		MaxRetries:       0,
		MaxRequestBytes:  bodyLimit,
		MaxResponseBytes: bodyLimit,
		MaxErrorBytes:    maxErrorBodyBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("create Looper connector: %w", err)
	}
	return &Client{
		connector: remote,
		endpoint:  cfg.Endpoint,
		headers:   cfg.Headers,
	}, nil
}

func (c *Client) callModelThroughConnector(
	ctx context.Context,
	body []byte,
	headers http.Header,
) ([]byte, error) {
	responseBody, err := c.connector.DoWithHeaders(ctx, chatCompletionOperation, body, headers)
	if err != nil {
		return nil, formatLooperConnectorError(err)
	}
	return responseBody, nil
}

func formatLooperConnectorError(err error) error {
	var connectorErr *connector.Error
	if !errors.As(err, &connectorErr) {
		return fmt.Errorf("request failed: %w", err)
	}
	switch connectorErr.Kind {
	case connector.KindStatus:
		body, truncated := connectorErr.ResponseBody()
		return fmt.Errorf(
			"request failed with status %d (error_body_bytes=%d, truncated=%t): %w",
			connectorErr.StatusCode,
			len(body),
			truncated,
			connectorErr,
		)
	case connector.KindResponse:
		return fmt.Errorf("failed to read response: %w", connectorErr)
	default:
		return fmt.Errorf("request failed: %w", connectorErr)
	}
}
