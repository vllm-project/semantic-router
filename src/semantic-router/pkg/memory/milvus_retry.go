package memory

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// isTransientError checks if an error is transient and should be retried
func isTransientError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Check for common transient error patterns
	transientPatterns := []string{
		"connection",
		"timeout",
		"deadline exceeded",
		"context deadline exceeded",
		"unavailable",
		"temporary",
		"retry",
		"rate limit",
		"too many requests",
		"server error",
		"internal error",
		"service unavailable",
		"network",
		"broken pipe",
		"connection reset",
		"no connection",
		"connection refused",
	}

	for _, pattern := range transientPatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	return false
}

// retryWithBackoff retries an operation with exponential backoff for transient errors
func (m *MilvusStore) retryWithBackoff(ctx context.Context, operation func() error) error {
	var lastErr error

	for attempt := 0; attempt < m.maxRetries; attempt++ {
		lastErr = operation()

		// If no error or non-transient error, return immediately
		if lastErr == nil || !isTransientError(lastErr) {
			return lastErr
		}

		// If this is the last attempt, return the error
		if attempt == m.maxRetries-1 {
			logging.Warnf("MilvusStore: operation failed after %d retries: %v", m.maxRetries, lastErr)
			return lastErr
		}

		// Calculate exponential backoff delay
		// Cap the exponent to avoid overflow (max 30 for safety)
		exponent := attempt
		if exponent < 0 {
			exponent = 0
		} else if exponent > 30 {
			exponent = 30
		}
		delay := m.retryBaseDelay * time.Duration(1<<exponent) // 2^attempt * baseDelay

		logging.Debugf("MilvusStore: transient error on attempt %d/%d, retrying in %v: %v",
			attempt+1, m.maxRetries, delay, lastErr)

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
			// Continue to next retry
		}
	}

	return lastErr
}
