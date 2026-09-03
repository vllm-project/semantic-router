package cache

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestContextErrorOnFailure(t *testing.T) {
	backendErr := errors.New("backend unavailable")

	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	expired, expire := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer expire()

	tests := []struct {
		name         string
		ctx          context.Context
		operationErr error
		want         error
	}{
		{name: "canceled operation", ctx: canceled, operationErr: backendErr, want: context.Canceled},
		{name: "expired operation", ctx: expired, operationErr: backendErr, want: context.DeadlineExceeded},
		{name: "active operation", ctx: context.Background(), operationErr: backendErr},
		{name: "no operation error", ctx: canceled},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := contextErrorOnFailure(tt.ctx, tt.operationErr); !errors.Is(got, tt.want) {
				t.Fatalf("contextErrorOnFailure() = %v, want %v", got, tt.want)
			}
		})
	}
}
