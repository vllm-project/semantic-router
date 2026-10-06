package connector

import (
	"context"
	"errors"
	"net"
	"net/http"
	"time"
)

const maxRetryDelay = time.Second

func retryableStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusRequestTimeout,
		http.StatusTooEarly,
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func retryableTransportError(err error) bool {
	if errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr)
}

func waitBeforeRetry(ctx context.Context, attempt int) error {
	delay := 50 * time.Millisecond * time.Duration(1<<min(attempt-1, 4))
	delay = min(delay, maxRetryDelay)
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
