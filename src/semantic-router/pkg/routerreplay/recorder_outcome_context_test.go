package routerreplay

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type stalledOutcomeStore struct {
	store.Storage
	entered chan struct{}
}

func (s *stalledOutcomeStore) AppendOutcome(ctx context.Context, _ string, _ store.Outcome) error {
	close(s.entered)
	<-ctx.Done()
	return ctx.Err()
}

func TestRecorderAppendOutcomeContextCancellation(t *testing.T) {
	backend := &stalledOutcomeStore{entered: make(chan struct{})}
	recorder := NewRecorder(backend)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- recorder.AppendOutcomeContext(ctx, "receipt", Outcome{}) }()
	select {
	case <-backend.entered:
	case <-time.After(time.Second):
		t.Fatal("receipt write did not start")
	}
	cancel()
	select {
	case err := <-done:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("receipt write ignored dispatcher cancellation")
	}
}
