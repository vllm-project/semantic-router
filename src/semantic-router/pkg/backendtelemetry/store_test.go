package backendtelemetry

import (
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func newTestStore(ttl time.Duration) (*Store, *fakeClock) {
	clock := &fakeClock{t: time.Unix(1_000, 0)}
	return newStoreWithClock(ttl, clock.now), clock
}

func TestStoreFreshUntilDefaultTTL(t *testing.T) {
	store, clock := newTestStore(0)
	o := validObservation()
	if err := store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	clock.advance(DefaultTTL - time.Nanosecond)
	if _, ok := store.Fresh(o.Identity); !ok {
		t.Fatal("observation expired before the default TTL")
	}
	clock.advance(time.Nanosecond)
	if _, ok := store.Fresh(o.Identity); ok {
		t.Fatal("observation still fresh at the default TTL")
	}
	if _, ok := store.Get(o.Identity); !ok {
		t.Fatal("Get dropped a stale observation it should keep for diagnostics")
	}
}

func TestStoreObservationTTLOverridesDefault(t *testing.T) {
	store, clock := newTestStore(time.Minute)
	o := validObservation()
	o.TTL = time.Second
	if err := store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	clock.advance(time.Second)
	if _, ok := store.Fresh(o.Identity); ok {
		t.Fatal("observation outlived its own TTL")
	}
}

func TestStoreFutureObservationIsNotFresh(t *testing.T) {
	store, _ := newTestStore(0)
	o := validObservation()
	o.CollectedAt = o.CollectedAt.Add(time.Second)
	if err := store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	if _, ok := store.Fresh(o.Identity); ok {
		t.Fatal("an observation from the future was reported fresh")
	}
}

func TestStoreKeepsNewestObservation(t *testing.T) {
	store, _ := newTestStore(0)
	newer := validObservation()
	newer.QueueDepth = Observed[int64](7)
	older := newer
	older.CollectedAt = newer.CollectedAt.Add(-time.Second)
	older.QueueDepth = Observed[int64](1)
	for _, o := range []Observation{newer, older} {
		if err := store.Upsert(o); err != nil {
			t.Fatal(err)
		}
	}
	got, _ := store.Get(newer.Identity)
	if v, _ := got.QueueDepth.Get(); v != 7 {
		t.Fatalf("queue depth = %d, want the newer observation's 7", v)
	}
}

func TestStoreRejectsInvalidObservation(t *testing.T) {
	store, _ := newTestStore(0)
	o := validObservation()
	o.Source = ""
	if err := store.Upsert(o); err == nil {
		t.Fatal("Upsert accepted an invalid observation")
	}
	if _, ok := store.Get(o.Identity); ok {
		t.Fatal("an invalid observation was stored")
	}
}

func TestStoreUpsertManyKeepsValidObservations(t *testing.T) {
	store, _ := newTestStore(0)
	good := validObservation()
	bad := validObservation()
	bad.Identity.BackendRef = "gpu-b"
	bad.Schema = 0
	err := store.UpsertMany([]Observation{bad, good})
	if err == nil || !strings.Contains(err.Error(), "gpu-b") {
		t.Fatalf("UpsertMany() = %v, want the invalid replica named", err)
	}
	if _, ok := store.Get(good.Identity); !ok {
		t.Fatal("a valid observation was dropped with an invalid sibling")
	}
}

func TestStoreReturnsIndependentCopies(t *testing.T) {
	store, _ := newTestStore(0)
	o := validObservation()
	o.QueueDepth = Observed[int64](3)
	if err := store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	o.QueueDepth = Observed[int64](99)
	got, _ := store.Get(o.Identity)
	got.QueueDepth = Observed[int64](42)
	again, _ := store.Get(o.Identity)
	if v, _ := again.QueueDepth.Get(); v != 3 {
		t.Fatalf("stored queue depth = %d, want 3 after mutating both the argument and a result", v)
	}
	if v, _ := got.QueueDepth.Get(); v != 42 {
		t.Fatalf("result queue depth = %d, want the caller's own 42", v)
	}
}

func TestStoreFreshForModelFiltersAndOrders(t *testing.T) {
	store, clock := newTestStore(0)
	mk := func(model, ref string, at time.Time) Observation {
		o := validObservation()
		o.Identity.Model, o.Identity.BackendRef, o.CollectedAt = model, ref, at
		return o
	}
	start := clock.now()
	for _, o := range []Observation{
		mk("qwen3-8b", "gpu-c", start),
		mk("qwen3-8b", "gpu-a", start),
		mk("other", "gpu-a", start),
		mk("qwen3-8b", "gpu-stale", start.Add(-DefaultTTL)),
	} {
		if err := store.Upsert(o); err != nil {
			t.Fatal(err)
		}
	}
	got := store.FreshForModel("qwen3-8b")
	var refs []string
	for _, o := range got {
		refs = append(refs, o.Identity.BackendRef)
	}
	if strings.Join(refs, ",") != "gpu-a,gpu-c" {
		t.Fatalf("FreshForModel refs = %v, want [gpu-a gpu-c]", refs)
	}
}

func TestStoreRemove(t *testing.T) {
	store, _ := newTestStore(0)
	o := validObservation()
	if err := store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	store.Remove(o.Identity)
	if _, ok := store.Get(o.Identity); ok {
		t.Fatal("Remove kept the observation")
	}
}

func TestStoreConcurrentUse(t *testing.T) {
	store, clock := newTestStore(0)
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range 200 {
				o := validObservation()
				o.CollectedAt = clock.now()
				o.ActiveRequests = Observed(int64(i*1000 + j))
				_ = store.Upsert(o)
				_ = store.FreshForModel(o.Identity.Model)
				clock.advance(time.Microsecond)
			}
		}()
	}
	wg.Wait()
	if _, ok := store.Get(validObservation().Identity); !ok {
		t.Fatal("no observation after concurrent writes")
	}
}
