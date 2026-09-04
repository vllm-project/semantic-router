package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestEventStoreRejectsSequenceAndByteLimitOverflow(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRun(context.Background(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}

	service.store.mu.Lock()
	service.store.sequences[run.ID] = maxEventsPerRun
	service.store.mu.Unlock()
	if _, err := service.store.AppendEvent(Event{
		RunID: run.ID, Type: "progress", Timestamp: time.Now().UTC(), Message: "overflow",
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("AppendEvent overflow error=%v, want ErrInvalid", err)
	}

	eventsPath := filepath.Join(root, "runs", run.ID, eventsFileName)
	if err := os.Truncate(eventsPath, maxEventLogBytes+1); err != nil {
		t.Fatalf("expand event log: %v", err)
	}
	if _, err := service.EventsAfter(run.ID, ""); !errors.Is(err, ErrInvalid) {
		t.Fatalf("EventsAfter oversized log error=%v, want ErrInvalid", err)
	}
}
