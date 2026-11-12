package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// fakeReadinessRouter answers /health and reports ready while ready is true.
func fakeReadinessRouter(t *testing.T, ready *atomic.Bool) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
		case "/ready":
			if !ready.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func TestStatusRequestPublishesRouterReadiness(t *testing.T) {
	var ready atomic.Bool
	router := fakeReadinessRouter(t, &ready)
	monitor := NewStatusMonitor(router.URL, "", t.TempDir(), StackState{}, nil)
	if _, ok := monitor.RouterReadiness(); ok {
		t.Fatal("readiness known before any observation")
	}

	request := func() {
		recorder := httptest.NewRecorder()
		monitor.Handler()(recorder, httptest.NewRequest(http.MethodGet, "/api/status", nil))
		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d", recorder.Code)
		}
	}
	request()
	first, ok := monitor.RouterReadiness()
	if !ok || first.Ready {
		t.Fatalf("readiness = %+v ok=%v, want known and not ready", first, ok)
	}

	ready.Store(true)
	request()
	second, _ := monitor.RouterReadiness()
	if !second.Ready || !second.CheckedAt.After(first.CheckedAt) {
		t.Fatalf("readiness = %+v, want ready and newer than %+v", second, first)
	}
}

func TestStatusMonitorLoopPublishesRouterReadiness(t *testing.T) {
	var ready atomic.Bool
	ready.Store(true)
	monitor := NewStatusMonitor(fakeReadinessRouter(t, &ready).URL, "", t.TempDir(), StackState{}, nil)
	monitor.Start()
	t.Cleanup(func() { _ = monitor.Close() })

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if readiness, ok := monitor.RouterReadiness(); ok {
			if !readiness.Ready {
				t.Fatalf("readiness = %+v, want ready", readiness)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("loop never published readiness")
}

func TestStatusMonitorKeepsTheNewestObservation(t *testing.T) {
	monitor := NewStatusMonitor("", "", t.TempDir(), StackState{}, nil)
	now := time.Now().UTC()
	monitor.publish(RouterReadiness{Ready: true, CheckedAt: now})
	monitor.publish(RouterReadiness{Ready: false, CheckedAt: now.Add(-time.Second)})
	if readiness, _ := monitor.RouterReadiness(); !readiness.Ready || !readiness.CheckedAt.Equal(now) {
		t.Fatalf("readiness = %+v, an older observation replaced a newer one", readiness)
	}
}

func TestCancelledObservationIsNotPublished(t *testing.T) {
	var ready atomic.Bool
	ready.Store(true)
	monitor := NewStatusMonitor(fakeReadinessRouter(t, &ready).URL, "", t.TempDir(), StackState{}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	monitor.observe(ctx)
	if readiness, ok := monitor.RouterReadiness(); ok {
		t.Fatalf("cancelled observation published %+v", readiness)
	}
}

func TestStatusMonitorCloseCancelsInFlightProbe(t *testing.T) {
	started := make(chan struct{}, 1)
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case started <- struct{}{}:
		default:
		}
		<-r.Context().Done()
	}))
	t.Cleanup(router.Close)
	monitor := NewStatusMonitor(router.URL, "", t.TempDir(), StackState{}, nil)
	monitor.Start()
	<-started

	closed := time.Now()
	if err := monitor.Close(); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(closed); elapsed > time.Second {
		t.Fatalf("Close waited %s for an in-flight probe", elapsed)
	}
}

func TestStatusMonitorCloseWithoutStart(t *testing.T) {
	if err := NewStatusMonitor("", "", t.TempDir(), StackState{}, nil).Close(); err != nil {
		t.Fatal(err)
	}
}

type staticReadiness RouterReadiness

func (s staticReadiness) RouterReadiness() (RouterReadiness, bool) {
	return RouterReadiness(s), !s.CheckedAt.IsZero()
}

func TestCapabilityProberReportsSharedRouterReadiness(t *testing.T) {
	checkedAt := time.Now().UTC()
	prober := newCapabilityProber(staticReadiness{Ready: true, CheckedAt: checkedAt}, nil, time.Hour)
	result, ok := prober.Snapshot()[probeRouter]
	if !ok || !result.ok || !result.checkedAt.Equal(checkedAt) {
		t.Fatalf("router result = %+v ok=%v", result, ok)
	}
	if _, ok := newCapabilityProber(staticReadiness{}, nil, time.Hour).Snapshot()[probeRouter]; ok {
		t.Fatal("unknown readiness reported as a result")
	}
}

func TestStatusRequestUpdatesConfigCapability(t *testing.T) {
	var ready atomic.Bool
	ready.Store(true)
	monitor := NewStatusMonitor(fakeReadinessRouter(t, &ready).URL, "", t.TempDir(), StackState{}, nil)
	fixture := newCapabilityFixture(t, true)
	prober := newCapabilityProber(monitor, nil, time.Hour)

	config := func() Capability {
		for _, capability := range collectCapabilities(fixture.cfg, fixture.setup, prober.Snapshot(), map[string]bool{"config.read": true}).Capabilities {
			if capability.ID == "config" {
				return capability
			}
		}
		t.Fatal("config capability missing")
		return Capability{}
	}
	if got := config(); got.Reason != reasonProbePending {
		t.Fatalf("before any status request: %+v", got)
	}
	monitor.Handler()(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/api/status", nil))
	if got := config(); !got.Reachable || got.CheckedAt == nil {
		t.Fatalf("after a status request: %+v", got)
	}
}
