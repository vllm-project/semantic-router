package fallback

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"syscall"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type customTimeoutError struct{}

func (e *customTimeoutError) Error() string   { return "custom network timeout" }
func (e *customTimeoutError) Timeout() bool   { return true }
func (e *customTimeoutError) Temporary() bool { return true }

func TestClassifier(t *testing.T) {
	policy := DefaultPolicy()
	c := NewClassifier(policy)

	t.Run("success status codes are not retryable", func(t *testing.T) {
		for _, code := range []int{200, 201, 204} {
			res := c.Classify(nil, code)
			if res.Retryable {
				t.Errorf("status %d should not be retryable", code)
			}
			if res.TriggerClass != TriggerClassNone {
				t.Errorf("status %d trigger class = %s, want %s", code, res.TriggerClass, TriggerClassNone)
			}
			if res.Reason != "success" {
				t.Errorf("status %d reason = %s, want success", code, res.Reason)
			}
		}
	})

	t.Run("retryable 5xx status codes", func(t *testing.T) {
		tests := []struct {
			code        int
			wantTrigger TriggerClass
			wantReason  string
		}{
			{502, TriggerClass5xx, "bad_gateway"},
			{503, TriggerClass5xx, "service_unavailable"},
			{504, TriggerClassTimeout, "gateway_timeout"},
		}
		for _, tt := range tests {
			res := c.Classify(nil, tt.code)
			if !res.Retryable {
				t.Errorf("status %d should be retryable", tt.code)
			}
			if res.TriggerClass != tt.wantTrigger {
				t.Errorf("status %d trigger class = %s, want %s", tt.code, res.TriggerClass, tt.wantTrigger)
			}
			if res.Reason != tt.wantReason {
				t.Errorf("status %d reason = %s, want %s", tt.code, res.Reason, tt.wantReason)
			}
		}
	})

	t.Run("non-allowlisted 5xx codes are non-retryable by default", func(t *testing.T) {
		for _, code := range []int{500, 501, 505} {
			res := c.Classify(nil, code)
			if res.Retryable {
				t.Errorf("status %d should not be retryable", code)
			}
		}
	})

	t.Run("client 4xx errors are strictly non-retryable", func(t *testing.T) {
		for _, code := range []int{400, 401, 403, 404, 422} {
			res := c.Classify(nil, code)
			if res.Retryable {
				t.Errorf("status %d should not be retryable", code)
			}
		}
	})

	t.Run("client 4xx errors with connection keywords in error message are strictly non-retryable", func(t *testing.T) {
		err := fmt.Errorf("upstream returned status 400: {\"error\": \"peer connection reset by peer\"}")
		res := c.Classify(err, 400)
		if res.Retryable {
			t.Errorf("status 400 with connection keywords in message must not be retryable")
		}
		if res.TriggerClass != TriggerClassNone {
			t.Errorf("expected TriggerClassNone, got %s", res.TriggerClass)
		}
		if res.Reason != "bad_request" {
			t.Errorf("expected bad_request, got %s", res.Reason)
		}
	})

	t.Run("rate limit 429 configurable in policy", func(t *testing.T) {
		resDefault := c.Classify(nil, 429)
		if resDefault.Retryable {
			t.Errorf("status 429 should not be retryable by default")
		}

		policyWith429 := DefaultPolicy()
		policyWith429.RetryableStatusCodes = append(policyWith429.RetryableStatusCodes, 429)
		cWith429 := NewClassifier(policyWith429)
		resWith429 := cWith429.Classify(nil, 429)
		if !resWith429.Retryable {
			t.Errorf("status 429 should be retryable when included in policy")
		}
		if resWith429.TriggerClass != TriggerClassRateLimit {
			t.Errorf("expected TriggerClassRateLimit, got %s", resWith429.TriggerClass)
		}
	})

	t.Run("context cancellation is never retryable", func(t *testing.T) {
		res := c.Classify(context.Canceled, 0)
		if res.Retryable {
			t.Errorf("context.Canceled must never be retryable")
		}
		if res.Reason != "client_canceled" {
			t.Errorf("expected client_canceled reason, got %s", res.Reason)
		}
	})

	t.Run("context deadline exceeded is retryable timeout", func(t *testing.T) {
		res := c.Classify(context.DeadlineExceeded, 0)
		if !res.Retryable {
			t.Errorf("context.DeadlineExceeded must be retryable")
		}
		if res.TriggerClass != TriggerClassTimeout {
			t.Errorf("expected TriggerClassTimeout, got %s", res.TriggerClass)
		}
	})

	t.Run("net.Error timeout is retryable", func(t *testing.T) {
		netTimeout := &net.DNSError{IsTimeout: true}
		res := c.Classify(netTimeout, 0)
		if !res.Retryable {
			t.Errorf("net timeout must be retryable")
		}
		if res.TriggerClass != TriggerClassTimeout {
			t.Errorf("expected TriggerClassTimeout, got %s", res.TriggerClass)
		}

		custom := &customTimeoutError{}
		res2 := c.Classify(custom, 0)
		if !res2.Retryable || res2.TriggerClass != TriggerClassTimeout {
			t.Errorf("custom net.Error timeout must be retryable")
		}
	})

	t.Run("connection failures are retryable", func(t *testing.T) {
		connErrors := []error{
			syscall.ECONNREFUSED,
			syscall.ECONNRESET,
			syscall.EPIPE,
			io.EOF,
			io.ErrUnexpectedEOF,
			fmt.Errorf("dial tcp 127.0.0.1:8000: connect: connection refused"),
			fmt.Errorf("read: connection reset by peer"),
			fmt.Errorf("write: broken pipe"),
		}
		for _, err := range connErrors {
			res := c.Classify(err, 0)
			if !res.Retryable {
				t.Errorf("error %v must be retryable", err)
			}
			if res.TriggerClass != TriggerClassConnection {
				t.Errorf("error %v expected TriggerClassConnection, got %s", err, res.TriggerClass)
			}
		}
	})

	t.Run("non-connection non-timeout unknown error is non-retryable", func(t *testing.T) {
		res := c.Classify(errors.New("unrecognized internal failure"), 0)
		if res.Retryable {
			t.Errorf("unknown error should not be retryable")
		}
		if res.TriggerClass != TriggerClassNone {
			t.Errorf("expected TriggerClassNone, got %s", res.TriggerClass)
		}
	})

	t.Run("error with 2xx status code is treated as failure and not success", func(t *testing.T) {
		connErr := errors.New("read error: unexpected EOF")
		res := c.Classify(connErr, 200)
		if res.Reason == "success" {
			t.Fatalf("error with 200 status should NOT be classified as success")
		}
		if !res.Retryable {
			t.Errorf("expected connection error with 200 status to be retryable, got %v", res)
		}
		if res.TriggerClass != TriggerClassConnection {
			t.Errorf("expected TriggerClassConnection, got %v", res.TriggerClass)
		}

		arbErr := errors.New("unrecognized internal failure")
		res2 := c.Classify(arbErr, 200)
		if res2.Reason == "success" {
			t.Fatalf("arbitrary error with 200 status should NOT be classified as success")
		}
		if res2.Retryable {
			t.Errorf("arbitrary error with 200 status should not be retryable")
		}
	})

	t.Run("protocol errors with 2xx or 0 status are retryable", func(t *testing.T) {
		protocolCases := []struct {
			name   string
			err    error
			status int
		}{
			{
				name:   "malformed response body string error",
				err:    errors.New("malformed response body"),
				status: 200,
			},
			{
				name:   "decode failed unexpected end of json",
				err:    errors.New("decode neutral response failed: unexpected end of JSON input"),
				status: 200,
			},
			{
				name:   "llmprotocol ProtocolError",
				err:    llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", "upstream response JSON is invalid", nil),
				status: 200,
			},
			{
				name:   "json syntax error",
				err:    &json.SyntaxError{Offset: 12},
				status: 200,
			},
			{
				name:   "translation failure without status",
				err:    errors.New("fallback response translation failed"),
				status: 0,
			},
		}

		for _, tc := range protocolCases {
			t.Run(tc.name, func(t *testing.T) {
				res := c.Classify(tc.err, tc.status)
				if !res.Retryable {
					t.Errorf("expected protocol error to be retryable, got %v", res)
				}
				if res.TriggerClass != TriggerClassProtocolError {
					t.Errorf("expected TriggerClassProtocolError, got %s", res.TriggerClass)
				}
				if res.Reason != "protocol_error" {
					t.Errorf("expected reason protocol_error, got %s", res.Reason)
				}
			})
		}
	})
}
