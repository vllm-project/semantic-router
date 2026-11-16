package backendtelemetry

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

type countingAdapter struct {
	rounds atomic.Int32
	obs    Observation
	err    error
}

func (a *countingAdapter) Engine() Engine { return EngineVLLM }

func (a *countingAdapter) Collect(context.Context) ([]Observation, error) {
	a.rounds.Add(1)
	o := a.obs
	o.CollectedAt = time.Now()
	return []Observation{o}, a.err
}

func TestCollectorRecordsObservationsAndReports(t *testing.T) {
	store := NewStore(time.Minute)
	adapter := &countingAdapter{obs: validObservation(), err: errors.New("gpu-b: refused")}
	reported := make(chan error, 16)
	collector := StartCollector(CollectorOptions{
		Adapters: []Adapter{adapter},
		Store:    store,
		Interval: 10 * time.Millisecond,
		Report:   func(_ Engine, _ []Observation, err error) { reported <- err },
	})
	defer func() { _ = collector.Close(context.Background()) }()

	select {
	case err := <-reported:
		if err == nil {
			t.Fatal("the round's target error was not reported")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no collection round ran")
	}
	if _, ok := store.Fresh(validObservation().Identity); !ok {
		t.Fatal("a partial round's observation was not recorded")
	}
	deadline := time.Now().Add(5 * time.Second)
	for adapter.rounds.Load() < 3 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if adapter.rounds.Load() < 3 {
		t.Fatalf("rounds = %d, want the collector to keep polling", adapter.rounds.Load())
	}
}

func TestCollectorCloseStopsPolling(t *testing.T) {
	adapter := &countingAdapter{obs: validObservation()}
	collector := StartCollector(CollectorOptions{
		Adapters: []Adapter{adapter}, Store: NewStore(0), Interval: time.Millisecond,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := collector.Close(ctx); err != nil {
		t.Fatal(err)
	}
	after := adapter.rounds.Load()
	time.Sleep(20 * time.Millisecond)
	if adapter.rounds.Load() != after {
		t.Fatal("the collector kept polling after Close")
	}
}

func TestCollectorJoinsAdapterAndStoreFailures(t *testing.T) {
	invalid := validObservation()
	invalid.Source = ""
	adapterFailure := errors.New("gpu-b: refused")
	adapter := &countingAdapter{obs: invalid, err: adapterFailure}
	reported := make(chan error, 1)
	collector := StartCollector(CollectorOptions{
		Adapters: []Adapter{adapter}, Store: NewStore(0), Interval: time.Hour,
		Report: func(_ Engine, _ []Observation, err error) { reported <- err },
	})
	defer func() { _ = collector.Close(context.Background()) }()
	select {
	case err := <-reported:
		if !errors.Is(err, adapterFailure) || !errors.Is(err, ErrInvalidObservation) {
			t.Fatalf("lost a collection failure: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no collection result")
	}
}

func TestCollectorWithoutAdaptersIsIdle(t *testing.T) {
	collector := StartCollector(CollectorOptions{Store: NewStore(0), Interval: time.Millisecond})
	if err := collector.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// pinnedAdapter holds its round until finish closes, then records when its
// idle connections are closed.
type pinnedAdapter struct {
	started, finish chan struct{}
	collecting      atomic.Bool
	closedMidRound  atomic.Bool
	idleClosed      chan struct{}
}

func (a *pinnedAdapter) Engine() Engine { return EngineVLLM }

func (a *pinnedAdapter) Collect(context.Context) ([]Observation, error) {
	a.collecting.Store(true)
	close(a.started)
	<-a.finish
	a.collecting.Store(false)
	return nil, nil
}

func (a *pinnedAdapter) CloseIdleConnections() {
	a.closedMidRound.Store(a.collecting.Load())
	close(a.idleClosed)
}

func TestCollectorClosesIdleConnectionsOnlyAfterItsLastRound(t *testing.T) {
	adapter := &pinnedAdapter{started: make(chan struct{}), finish: make(chan struct{}), idleClosed: make(chan struct{})}
	collector := StartCollector(CollectorOptions{Adapters: []Adapter{adapter}, Store: NewStore(time.Minute), Interval: time.Hour})
	<-adapter.started
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := collector.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("bounded Close() = %v", err)
	}
	select {
	case <-adapter.idleClosed:
		t.Fatal("idle connections closed while a scrape was in flight")
	case <-time.After(20 * time.Millisecond):
	}
	close(adapter.finish)
	if err := collector.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-adapter.idleClosed:
	default:
		t.Fatal("idle connections stayed open after the last round")
	}
	if adapter.closedMidRound.Load() {
		t.Fatal("idle connections closed during a round")
	}
}
