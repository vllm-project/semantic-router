package statusstore

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestStorePersistsAcrossRestartAndFillsUnknownGaps(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "status.sqlite")
	firstHour := time.Date(2026, time.January, 1, 12, 0, 0, 0, time.UTC)
	store, openErr := Open(dbPath)
	if openErr != nil {
		t.Fatal(openErr)
	}
	store.now = func() time.Time { return firstHour }
	if _, err := store.Observe(context.Background(), []Observation{
		{Service: "Router", State: StateUnavailable},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, reopenErr := Open(dbPath)
	if reopenErr != nil {
		t.Fatal(reopenErr)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	current := firstHour
	reopened.now = func() time.Time { return current }
	history, err := reopened.Observe(context.Background(), []Observation{
		{Service: "Router", State: StateOperational},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertHourState(t, historyFor(t, history, "Router"), "2026-01-01T12:00:00Z", StateUnavailable)

	current = firstHour.Add(2 * time.Hour)
	history, err = reopened.Observe(context.Background(), []Observation{
		{Service: "Router", State: StateOperational},
	})
	if err != nil {
		t.Fatal(err)
	}

	router := historyFor(t, history, "Router")
	if len(router.Hours) != RetentionHours {
		t.Fatalf("hour count = %d, want %d", len(router.Hours), RetentionHours)
	}
	assertHourState(t, router, "2026-01-01T12:00:00Z", StateUnavailable)
	assertHourState(t, router, "2026-01-01T13:00:00Z", StateUnknown)
	assertHourState(t, router, "2026-01-01T14:00:00Z", StateOperational)
}

func TestStoreKeepsWorstHourlyStateAcrossConcurrentObservations(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "status.sqlite")
	store, openErr := Open(dbPath)
	if openErr != nil {
		t.Fatal(openErr)
	}
	t.Cleanup(func() { _ = store.Close() })
	secondStore, secondOpenErr := Open(dbPath)
	if secondOpenErr != nil {
		t.Fatal(secondOpenErr)
	}
	t.Cleanup(func() { _ = secondStore.Close() })
	clock := func() time.Time {
		return time.Date(2026, time.February, 4, 9, 0, 0, 0, time.UTC)
	}
	store.now = clock
	secondStore.now = clock

	observations := []struct {
		store *Store
		state State
	}{
		{store: store, state: StateOperational},
		{store: secondStore, state: StateStarting},
		{store: store, state: StateUnavailable},
		{store: secondStore, state: StateOperational},
	}
	var wait sync.WaitGroup
	errors := make(chan error, len(observations))
	start := make(chan struct{})
	for _, observation := range observations {
		wait.Add(1)
		go func(observation struct {
			store *Store
			state State
		},
		) {
			defer wait.Done()
			<-start
			_, observeErr := observation.store.Observe(context.Background(), []Observation{
				{Service: "Router", State: observation.state},
			})
			errors <- observeErr
		}(observation)
	}
	close(start)
	wait.Wait()
	close(errors)
	for observeErr := range errors {
		if observeErr != nil {
			t.Fatal(observeErr)
		}
	}

	history, err := store.Observe(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	assertHourState(t, historyFor(t, history, "Router"), "2026-02-04T09:00:00Z", StateUnavailable)
}

func TestStorePrunesOutsideTheBoundedWindow(t *testing.T) {
	store, openErr := Open(filepath.Join(t.TempDir(), "status.sqlite"))
	if openErr != nil {
		t.Fatal(openErr)
	}
	t.Cleanup(func() { _ = store.Close() })

	current := time.Date(2025, time.January, 1, 0, 0, 0, 0, time.UTC)
	store.now = func() time.Time { return current }
	if _, err := store.Observe(context.Background(), []Observation{
		{Service: "Router", State: StateUnavailable},
	}); err != nil {
		t.Fatal(err)
	}
	current = current.Add(time.Duration(RetentionHours) * time.Hour)
	history, err := store.Observe(context.Background(), []Observation{
		{Service: "Router", State: StateOperational},
	})
	if err != nil {
		t.Fatal(err)
	}
	router := historyFor(t, history, "Router")
	if router.Hours[0].Status != StateUnknown {
		t.Fatalf("oldest retained hour = %q, want unknown after pruning", router.Hours[0].Status)
	}
	assertHourState(t, router, current.Format(time.RFC3339), StateOperational)

	var count int
	if err := store.db.QueryRow(`SELECT COUNT(*) FROM service_status_hourly`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("persisted row count = %d, want 1", count)
	}
}

func historyFor(t *testing.T, history History, serviceName string) ServiceHistory {
	t.Helper()
	for _, service := range history.Services {
		if service.Name == serviceName {
			return service
		}
	}
	t.Fatalf("history for %q not found", serviceName)
	return ServiceHistory{}
}

func assertHourState(t *testing.T, history ServiceHistory, observedAt string, want State) {
	t.Helper()
	for _, hour := range history.Hours {
		if hour.ObservedAt == observedAt {
			if hour.Status != want {
				t.Fatalf("%s status = %q, want %q", observedAt, hour.Status, want)
			}
			return
		}
	}
	t.Fatalf("hour %q not found", observedAt)
}
