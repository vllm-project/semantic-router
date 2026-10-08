package looper

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// NewHopClient returns a Client whose model calls are hops that hops sends:
// a request-graph run (*graph.Exec), which accounts for them in its budget,
// spans and evidence, or a caller such as graph.SessionCaller for a one-off
// helper call. The routing core serves each hop in process, with its
// decision's plugin chain and the upstream layer, so no request goes on the
// wire and no internal credential is needed.
func NewHopClient(cfg *config.LooperConfig, hops graph.Caller) *Client {
	return &Client{
		hops:             hops,
		headers:          cfg.Headers,
		hopTimeout:       time.Duration(cfg.GetTimeout()) * time.Second,
		maxResponseBytes: cfg.GetMaxResponseBytes(),
	}
}

// callModelAsHop sends one model call as a hop, bounded as the HTTP connector
// bounds it: the Looper's timeout per call, its body limit both ways, and 200
// as the only success. A failure reads as the connector's would, since traces
// that reach the client quote it.
func (c *Client) callModelAsHop(
	ctx context.Context,
	body []byte,
	header http.Header,
	target ModelTarget,
	options CallOptions,
) ([]byte, error) {
	if int64(len(body)) > c.maxResponseBytes {
		return nil, fmt.Errorf("request failed: %w", hopError(fmt.Errorf("request body exceeds %d bytes", c.maxResponseBytes)))
	}
	ctx, cancel := context.WithTimeout(ctx, c.hopTimeout)
	defer cancel()
	resp, err := c.hops.Call(ctx, &graph.HopRequest{
		Hop: routing.Hop{
			Decision:  options.DecisionName,
			Recipe:    string(routingRecipeFromContext(ctx)),
			Iteration: options.Iteration,
			// A Looper's calls never fell back across models; the
			// algorithm owns its candidates.
			Fallback: fallback.Disabled(),
		},
		Model:   target.Name,
		Request: routing.Request{Header: hopRequestHeader(header, len(body)), Body: body},
	})
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", hopError(err))
	}
	recordAttemptFirstByte(ctx)
	if resp.Status != http.StatusOK {
		return nil, hopStatusError(resp)
	}
	if int64(len(resp.Body)) > c.maxResponseBytes {
		return nil, fmt.Errorf("failed to read response: %w", hopError(fmt.Errorf("response body exceeds %d bytes", c.maxResponseBytes)))
	}
	return resp.Body, nil
}

// hopRequestHeader is the header of a hop's chat completions request.
func hopRequestHeader(header http.Header, bodyBytes int) routing.Header {
	out := graph.ChatHeader()
	for name, values := range header {
		name = strings.ToLower(name)
		if name == "content-type" {
			continue
		}
		for _, value := range values {
			out.Add(name, value)
		}
	}
	out.Set("content-length", strconv.Itoa(bodyBytes))
	return out
}

// hopOperationError is a hop failure worded as the HTTP connector words a
// failed looper_chat_completion operation.
type hopOperationError struct {
	status int
	cause  error
}

func hopError(cause error) error { return &hopOperationError{cause: cause} }

func (e *hopOperationError) Error() string {
	prefix := fmt.Sprintf("connector operation %q failed on attempt 1", chatCompletionOperation.Name)
	if e.status != 0 {
		return fmt.Sprintf("%s with HTTP status %d", prefix, e.status)
	}
	return fmt.Sprintf("%s: %v", prefix, e.cause)
}

func (e *hopOperationError) Unwrap() error { return e.cause }

// hopStatusError fails a hop answered with a status other than 200, keeping
// as much of the error body as the connector keeps.
func hopStatusError(resp *graph.HopResponse) error {
	kept, truncated := len(resp.Body), false
	if int64(kept) > maxErrorBodyBytes {
		kept, truncated = int(maxErrorBodyBytes), true
	}
	return fmt.Errorf(
		"request failed with status %d (error_body_bytes=%d, truncated=%t): %w",
		resp.Status, kept, truncated, &hopOperationError{status: resp.Status},
	)
}

// modelFailureReason is how a failed model call reads in a trace that
// reaches the client: the status it was answered with, or what ended it,
// never transport or provider text.
func modelFailureReason(err error) string {
	var hopErr *hopOperationError
	var connectorErr *connector.Error
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "timed out"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.As(err, &hopErr) && hopErr.status != 0:
		return fmt.Sprintf("answered %d", hopErr.status)
	case errors.As(err, &connectorErr) && connectorErr.StatusCode != 0:
		return fmt.Sprintf("answered %d", connectorErr.StatusCode)
	case attemptReasonFromError(err) == AttemptReasonInvalidResponse:
		return "invalid response"
	default:
		return "failed"
	}
}
