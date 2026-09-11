package evaluationplane

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestEventStoreRejectsSequenceAndByteLimitOverflow(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}

	service.store.runIndex.setEventSequence(run.ID, maxEventsPerRun)
	if _, err := service.store.AppendEvent(Event{
		RunID: run.ID, Type: "progress", Timestamp: time.Now().UTC(), Message: "overflow",
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("AppendEvent overflow error=%v, want ErrInvalid", err)
	}

	eventsPath := filepath.Join(root, "runs", run.ID, eventsFileName)
	if err := os.Truncate(eventsPath, maxEventLogBytes+1); err != nil {
		t.Fatalf("expand event log: %v", err)
	}
	if _, err := service.EventsAfterAs(SystemActor(), run.ID, ""); !errors.Is(err, ErrInvalid) {
		t.Fatalf("EventsAfter oversized log error=%v, want ErrInvalid", err)
	}
}

func TestEventStoreRejectsUnknownDurableFieldsOnReadAndAppend(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	eventsPath := filepath.Join(root, "runs", run.ID, eventsFileName)
	initial, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("read initial event: %v", err)
	}
	line := bytes.TrimSpace(initial)
	tampered := append(append([]byte(nil), line[:len(line)-1]...), []byte(`,"forged":true}`)...)
	if err := os.WriteFile(eventsPath, append(tampered, '\n'), 0o600); err != nil {
		t.Fatalf("write tampered event: %v", err)
	}
	if _, err := service.store.EventsAfter(run.ID, 0); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("EventsAfter unknown durable field error=%v", err)
	}
	service.store.runIndex.mu.Lock()
	delete(service.store.runIndex.eventSequences, run.ID)
	service.store.runIndex.mu.Unlock()
	if _, err := service.store.AppendEvent(Event{
		RunID: run.ID, Type: "progress", Timestamp: time.Now().UTC(), Message: "next",
	}); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("AppendEvent unknown durable field error=%v", err)
	}
}

func TestEventSequenceIsSharedAcrossStoresForTheSameRoot(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	peer := newTestPeerStore(t, service.store)
	stores := []*Store{service.store, peer, service.store}
	for index, store := range stores {
		if _, appendErr := store.AppendEvent(Event{
			RunID: run.ID, Type: "progress", Timestamp: time.Now().UTC(),
			Message: fmt.Sprintf("cross-store-%d", index),
		}); appendErr != nil {
			t.Fatalf("AppendEvent %d: %v", index, appendErr)
		}
	}
	events, err := peer.EventsAfter(run.ID, 0)
	if err != nil {
		t.Fatalf("EventsAfter: %v", err)
	}
	if len(events) != 4 {
		t.Fatalf("events=%+v, want snapshot plus three control events", events)
	}
	for index, event := range events {
		if event.ID != fmt.Sprint(index+1) {
			t.Fatalf("event %d id=%q, want %d", index, event.ID, index+1)
		}
	}
}
