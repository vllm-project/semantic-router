package fallback

import (
	"context"
	"errors"
	"io"
	"net"
	"slices"
	"strings"
	"syscall"
)

// Classifier determines whether an upstream outcome is retryable and classifies its failure trigger.
type Classifier struct {
	policy FallbackPolicy
}

// NewClassifier creates a new failure classifier with the given policy.
func NewClassifier(policy FallbackPolicy) *Classifier {
	return &Classifier{policy: policy}
}

// ClassificationResult contains the structured evaluation of an upstream error or status.
type ClassificationResult struct {
	Retryable    bool
	TriggerClass TriggerClass
	Reason       string
}

// Classify evaluates an error and HTTP status code according to the fallback policy.
func (c *Classifier) Classify(err error, statusCode int) ClassificationResult {
	// 1. If context was canceled by client, it is never retryable.
	if err != nil && errors.Is(err, context.Canceled) {
		return ClassificationResult{
			Retryable:    false,
			TriggerClass: TriggerClassNone,
			Reason:       "client_canceled",
		}
	}

	// 2. Check context deadline exceeded (timeout).
	if err != nil && errors.Is(err, context.DeadlineExceeded) {
		return ClassificationResult{
			Retryable:    true,
			TriggerClass: TriggerClassTimeout,
			Reason:       "request_deadline_exceeded",
		}
	}

	// 3. Evaluate HTTP status code outcomes when a response was received without error.
	if err == nil && statusCode >= 200 && statusCode < 300 {
		return ClassificationResult{
			Retryable:    false,
			TriggerClass: TriggerClassNone,
			Reason:       "success",
		}
	}

	if slices.Contains(c.policy.RetryableStatusCodes, statusCode) {
		trigger := TriggerClass5xx
		switch statusCode {
		case 504:
			trigger = TriggerClassTimeout
		case 429:
			trigger = TriggerClassRateLimit
		}
		return ClassificationResult{
			Retryable:    true,
			TriggerClass: trigger,
			Reason:       statusCodeReason(statusCode),
		}
	}

	// Client 4xx errors are strictly non-retryable.
	if statusCode >= 400 && statusCode < 500 {
		return ClassificationResult{
			Retryable:    false,
			TriggerClass: TriggerClassNone,
			Reason:       statusCodeReason(statusCode),
		}
	}

	// Non-allowlisted 5xx codes (e.g. 500 Internal Server Error, 501 Not Implemented).
	if statusCode >= 500 {
		return ClassificationResult{
			Retryable:    false,
			TriggerClass: TriggerClass5xx,
			Reason:       statusCodeReason(statusCode),
		}
	}

	// 4. Check network and transport errors when no HTTP response was received or for underlying network issues.
	if err != nil {
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return ClassificationResult{
				Retryable:    true,
				TriggerClass: TriggerClassTimeout,
				Reason:       "network_timeout",
			}
		}

		if isConnectionFailure(err) {
			return ClassificationResult{
				Retryable:    true,
				TriggerClass: TriggerClassConnection,
				Reason:       "connection_failure",
			}
		}
	}

	// Default: if an error occurred that wasn't classified above.
	if err != nil {
		return ClassificationResult{
			Retryable:    false,
			TriggerClass: TriggerClassNone,
			Reason:       err.Error(),
		}
	}

	return ClassificationResult{
		Retryable:    false,
		TriggerClass: TriggerClassNone,
		Reason:       "unknown_outcome",
	}
}

// isConnectionFailure inspects error chains for connection refusal, reset, broken pipe, or unexpected EOF.
func isConnectionFailure(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	if errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.ECONNRESET) ||
		errors.Is(err, syscall.EPIPE) ||
		errors.Is(err, syscall.ENETUNREACH) ||
		errors.Is(err, syscall.EHOSTUNREACH) {
		return true
	}

	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "connection reset") ||
		strings.Contains(msg, "broken pipe") ||
		strings.Contains(msg, "no such host") ||
		strings.Contains(msg, "network is unreachable") ||
		strings.Contains(msg, "unexpected eof") ||
		strings.Contains(msg, "server closed connection")
}

func statusCodeReason(code int) string {
	switch code {
	case 400:
		return "bad_request"
	case 401:
		return "unauthorized"
	case 403:
		return "forbidden"
	case 404:
		return "not_found"
	case 422:
		return "unprocessable_entity"
	case 429:
		return "rate_limit_exceeded"
	case 500:
		return "internal_server_error"
	case 501:
		return "not_implemented"
	case 502:
		return "bad_gateway"
	case 503:
		return "service_unavailable"
	case 504:
		return "gateway_timeout"
	default:
		return "http_status_error"
	}
}
