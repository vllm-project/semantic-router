package store

import (
	"context"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

// backingArrayPtrs returns all *Record pointers in the slice's backing array,
// from index 0 up to capacity. It is the only reliable way to verify that
// trailing slots beyond the slice length were actually nil-out.
func backingArrayPtrs(t *testing.T, s []*Record) []*Record {
	t.Helper()
	capacity := cap(s)
	if capacity == 0 {
		return nil
	}
	// unsafe.Slice produces a []*Record view of the full backing-array
	// capacity.  This is the vet-safe way to inspect trailing slots; the
	// go vet "possible misuse of unsafe.Pointer" rule would flag manual
	// pointer arithmetic above.
	full := unsafe.Slice(unsafe.SliceData(s), capacity)
	out := make([]*Record, capacity)
	copy(out, full)
	return out
}

func ttlRecord(id string, ts time.Time) Record {
	rec := Record{ID: id, Timestamp: ts, LifecycleState: LifecycleCompleted}
	return rec
}

func TestMemoryStoreTTLListElidesExpired(t *testing.T) {
	m := NewMemoryStore(100, 10) // retention: 10s
	now := time.Now().UTC()
	fresh := ttlRecord("fresh", now)
	stale := ttlRecord("stale", now.Add(-20*time.Second))

	if _, err := m.Add(context.Background(), fresh); err != nil {
		t.Fatalf("Add fresh: %v", err)
	}
	if _, err := m.Add(context.Background(), stale); err != nil {
		t.Fatalf("Add stale: %v", err)
	}

	got, err := m.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 live record, got %d: %+v", len(got), got)
	}
	if got[0].ID != "fresh" {
		t.Fatalf("expected live record to be 'fresh', got %q", got[0].ID)
	}
}

func TestMemoryStoreTTLZeroKeepsAll(t *testing.T) {
	m := NewMemoryStore(100, 0) // no time-based expiry
	now := time.Now().UTC()
	if _, err := m.Add(context.Background(), ttlRecord("old", now.Add(-1*time.Hour))); err != nil {
		t.Fatalf("Add old: %v", err)
	}
	if _, err := m.Add(context.Background(), ttlRecord("new", now)); err != nil {
		t.Fatalf("Add new: %v", err)
	}

	got, err := m.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 records with ttl=0, got %d", len(got))
	}
}

func TestMemoryStoreTTLPurgesOnAdd(t *testing.T) {
	m := NewMemoryStore(100, 10)
	now := time.Now().UTC()
	if _, err := m.Add(context.Background(), ttlRecord("ancient", now.Add(-30*time.Second))); err != nil {
		t.Fatalf("Add ancient: %v", err)
	}
	// A later mutation must purge the expired record while adding a fresh one.
	if _, err := m.Add(context.Background(), ttlRecord("current", now)); err != nil {
		t.Fatalf("Add current: %v", err)
	}

	got, err := m.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 1 || got[0].ID != "current" {
		t.Fatalf("expected only 'current', got %+v", got)
	}
}

func TestMemoryStoreTTLGetElidesExpired(t *testing.T) {
	m := NewMemoryStore(100, 10)
	now := time.Now().UTC()
	stale := ttlRecord("stale", now.Add(-30*time.Second))
	id, err := m.Add(context.Background(), stale)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	_, ok, err := m.Get(context.Background(), id)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ok {
		t.Fatalf("expected expired record to be elided from Get")
	}

	// Verify both logical containers are now empty
	if _, exists := m.byID[id]; exists {
		t.Errorf("byID still contains expired record %q after Get purge", id)
	}
	if len(m.records) != 0 {
		t.Errorf("records slice has %d entries, want 0 after Get purge", len(m.records))
	}
}

// TestMemoryStoreTTLAddThenGetClearsBackingArray is the exact reproduction of
// the scenario the reviewer flagged: "Add followed by Get: both logical
// containers are empty while the backing slot still holds a 1 MiB expired
// payload". This test proves that after Get purges the expired record,
// every slot in the backing array (not just the visible length) is niled.
func TestMemoryStoreTTLAddThenGetClearsBackingArray(t *testing.T) {
	m := NewMemoryStore(100, 5) // retention: 5s
	now := time.Now().UTC()

	// Step 1: Add an expired record (Reviewer's repro step)
	stale := ttlRecord("expired", now.Add(-10*time.Second))
	_, err := m.Add(context.Background(), stale)
	if err != nil {
		t.Fatalf("Add expired record: %v", err)
	}

	// Sanity check before Get — backing array should still hold the stale pointer
	m.mu.Lock()
	slotsBefore := backingArrayPtrs(t, m.records)
	visibleBefore := len(m.records)
	m.mu.Unlock()
	if visibleBefore != 1 {
		t.Fatalf("precondition: expected 1 record before Get, got %d", visibleBefore)
	}

	// Step 2: Get the expired record — this should purge it AND clear backing array
	_, ok, err := m.Get(context.Background(), "expired")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ok {
		t.Fatalf("expected expired record to be elided from Get")
	}

	// Step 3: Verify both logical containers are empty (Reviewer's "both
	// logical containers are empty" assertion)
	if len(m.records) != 0 {
		t.Errorf("after Get purge: records has %d entries, want 0", len(m.records))
	}
	if len(m.byID) != 0 {
		t.Errorf("after Get purge: byID has %d entries, want 0", len(m.byID))
	}

	// Step 4: Verify backing array no longer holds the stale pointer —
	// this is the Reviewer's key concern ("backing slot still holds a 1
	// MiB expired payload"). If we use the same pre-allocated capacity
	// and no later writes occur, stale pointers must be niled.
	m.mu.Lock()
	slotsAfter := backingArrayPtrs(t, m.records)
	m.mu.Unlock()

	// Backing capacity should be unchanged (100 pre-allocated slots)
	if len(slotsAfter) != len(slotsBefore) {
		t.Errorf("backing capacity changed: before=%d after=%d", len(slotsBefore), len(slotsAfter))
	}

	// Every slot in the backing array must be nil — the stale Record
	// pointer must not be reachable anywhere, not just beyond the
	// visible length.
	for idx, ptr := range slotsAfter {
		if ptr != nil {
			t.Errorf("backing array slot[%d] still holds %q — stale pointer leaks past purge", idx, ptr.ID)
		}
	}
}

// TestMemoryStoreTTLPurgeClearsBackingSlots verifies that after purge,
// the backing array does not retain references to expired records.
func TestMemoryStoreTTLPurgeClearsBackingSlots(t *testing.T) {
	m := NewMemoryStore(100, 5) // retention: 5s
	now := time.Now().UTC()

	// Add several records, some fresh, some expired
	freshIDs := []string{"f1", "f2", "f3"}
	staleIDs := []string{"s1", "s2", "s3", "s4"}

	for _, id := range freshIDs {
		if _, err := m.Add(context.Background(), ttlRecord(id, now)); err != nil {
			t.Fatalf("Add fresh %s: %v", id, err)
		}
	}
	for _, id := range staleIDs {
		if _, err := m.Add(context.Background(), ttlRecord(id, now.Add(-10*time.Second))); err != nil {
			t.Fatalf("Add stale %s: %v", id, err)
		}
	}

	// Force purge by adding a new fresh record
	if _, err := m.Add(context.Background(), ttlRecord("trigger", now)); err != nil {
		t.Fatalf("Add trigger: %v", err)
	}

	// Verify byID map has no stale entries
	for _, id := range staleIDs {
		if _, ok := m.byID[id]; ok {
			t.Errorf("byID still contains stale record %q after purge", id)
		}
	}
	if len(m.byID) != len(freshIDs)+1 {
		t.Errorf("byID has %d entries, want %d (3 fresh + 1 trigger)", len(m.byID), len(freshIDs)+1)
	}

	// Verify records slice has no stale entries
	for _, rec := range m.records {
		for _, sid := range staleIDs {
			if rec.ID == sid {
				t.Errorf("records slice still contains stale record %q after purge", sid)
			}
		}
	}
	if len(m.records) != len(freshIDs)+1 {
		t.Errorf("records slice has %d entries, want %d", len(m.records), len(freshIDs)+1)
	}

	// List should return only fresh records
	got, err := m.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != len(freshIDs)+1 {
		t.Errorf("List returned %d records, want %d", len(got), len(freshIDs)+1)
	}

	// Get should elide stale records
	for _, id := range staleIDs {
		_, ok, err := m.Get(context.Background(), id)
		if err != nil {
			t.Fatalf("Get %s: %v", id, err)
		}
		if ok {
			t.Errorf("Get %s unexpectedly returned a record", id)
		}
	}

	// --- Backing-array regression guard ---
	// Walk every slot in the full backing array (beyond the visible length)
	// and assert that no expired *Record pointer is reachable. If a slot
	// still held a stale pointer, GC could not reclaim the expired payloads.
	m.mu.Lock()
	slots := backingArrayPtrs(t, m.records)
	m.mu.Unlock()

	for idx, ptr := range slots {
		if ptr == nil {
			continue
		}
		for _, sid := range staleIDs {
			if ptr.ID == sid {
				t.Errorf("backing array slot[%d] still holds stale record %q — GC retention leak", idx, sid)
			}
		}
	}

	// Trailing slots beyond the visible length must all be nil — no
	// stale pointers may linger after purgeExpiredLocked truncated the
	// in-place slice.
	for i := len(m.records); i < len(slots); i++ {
		if slots[i] != nil {
			t.Errorf("backing array trailing slot[%d] = %q, want nil", i, slots[i].ID)
		}
	}
}

// TestMemoryStoreTTLExpiredRecordsAreGCRetainedDirectly verifies that after
// purge, the backing array no longer holds any reference to expired Record
// payloads — confirmed by runtime.SetFinalizer. This is the reviewer's
// "retention regression": it proves expired payloads are not just absent
// from the visible slice, but are also unreachable from the slice's
// backing array and will be reclaimed by GC.
//
// If purgeExpiredLocked or the ring-buffer eviction ever stops niling out
// a backing-array slot, this test will hang forever waiting for the
// finalizer to fire — an unambiguous signal that the stale pointer is
// still reachable.
func TestMemoryStoreTTLExpiredRecordsAreGCRetainedDirectly(t *testing.T) {
	m := NewMemoryStore(100, 5) // retention: 5s
	now := time.Now().UTC()

	// Create a large expired record with a SetFinalizer so we can detect
	// when GC considers it unreachable.
	stale := ttlRecord("gc_expired", now.Add(-30*time.Second))
	// Give it a non-trivial payload so there is actually something worth
	// reclaiming (reviewer flagged "1 MiB expired payload").
	stale.ResponseBody = string(make([]byte, 1<<20)) // 1 MiB
	stale.RequestBody = string(make([]byte, 1<<20))  // 1 MiB

	// finalizerFired closes when the stale *Record has no more references
	// and GC finally collects it.
	finalizerFired := make(chan struct{})
	// Capture the Record we store via Add (it clones the input, so we need
	// to attach the finalizer to the clone actually held by the store).
	// We'll grab the *Record out of the store right after Add to attach it.
	_, err := m.Add(context.Background(), stale)
	if err != nil {
		t.Fatalf("Add expired record: %v", err)
	}

	// Attach finalizer to the clone held inside m.records — GC only fires
	// it once that exact pointer becomes unreachable.
	m.mu.Lock()
	if len(m.records) == 0 {
		m.mu.Unlock()
		t.Fatal("precondition: expected 1 record after Add")
	}
	staleClone := m.records[0]
	runtime.SetFinalizer(staleClone, func(*Record) {
		close(finalizerFired)
	})
	m.mu.Unlock()

	// Purge via Get — same purge path the reviewer exercised
	_, ok, err := m.Get(context.Background(), "gc_expired")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ok {
		t.Fatalf("expected expired record to be elided from Get")
	}

	// Force aggressive GC to make sure the finalizer runs promptly.
	// We run multiple GCs because finalizers are only guaranteed to run
	// after the object is unreachable, and a single GC may not collect
	// everything.
	for i := 0; i < 10; i++ {
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}

	select {
	case <-finalizerFired:
		// Good — the stale *Record clone is now unreachable; both the
		// byID map and the backing array have released it.
	case <-time.After(2 * time.Second):
		t.Fatalf(
			"GC retention leak: finalizer for stale record never fired. " +
				"The backing array still holds a reference to the expired *Record, " +
				"preventing GC from reclaiming its 2 MiB payload.",
		)
	}
}

// TestMemoryStoreRingBufferEvictReleasesGC verifies that when the ring
// buffer evicts the oldest record (because maxRecords is hit), the evicted
// slot is niled so the evicted *Record becomes GC-reclaimable.
//
// We use ttl=0 (no time-based expiry) so all three records survive Add and
// the eviction path is exercised purely by the ring-buffer overflow, which
// is a distinct code path from purgeExpiredLocked.
func TestMemoryStoreRingBufferEvictReleasesGC(t *testing.T) {
	m := NewMemoryStore(2, 0) // maxRecords=2, no TTL — eviction only by ring buffer
	now := time.Now().UTC()

	rec1 := ttlRecord("r1", now.Add(-30*time.Second)) // old, but ttl=0 so no expiry
	rec2 := ttlRecord("r2", now)
	rec3 := ttlRecord("r3", now.Add(5*time.Second))

	if _, err := m.Add(context.Background(), rec1); err != nil {
		t.Fatalf("Add r1: %v", err)
	}
	if _, err := m.Add(context.Background(), rec2); err != nil {
		t.Fatalf("Add r2: %v", err)
	}

	// At this point ring buffer is full (len=2). records[0]=r1, records[1]=r2.
	// Attach finalizer to r1 which will be evicted by the next Add.
	finalizerFired := make(chan struct{})
	m.mu.Lock()
	if len(m.records) != 2 {
		t.Fatalf("precondition: expected 2 records, got %d", len(m.records))
	}
	r1 := m.records[0]
	if r1.ID != "r1" {
		t.Fatalf("precondition: records[0].ID=%q, want 'r1'", r1.ID)
	}
	runtime.SetFinalizer(r1, func(*Record) { close(finalizerFired) })
	m.mu.Unlock()

	// Add third record → triggers ring-buffer eviction of r1.
	if _, err := m.Add(context.Background(), rec3); err != nil {
		t.Fatalf("Add r3: %v", err)
	}

	// Verify r1 is gone from both logical containers
	m.mu.Lock()
	for _, rec := range m.records {
		if rec != nil && rec.ID == "r1" {
			t.Errorf("r1 still in records slice after eviction")
		}
	}
	if _, exists := m.byID["r1"]; exists {
		t.Errorf("r1 still in byID map after eviction")
	}
	m.mu.Unlock()

	// Verify backing array slot is niled (defense in depth).
	m.mu.Lock()
	slots := backingArrayPtrs(t, m.records)
	m.mu.Unlock()
	for idx, ptr := range slots {
		if ptr != nil && ptr.ID == "r1" {
			t.Errorf("backing array slot[%d] still holds r1 after ring-buffer eviction", idx)
		}
	}

	// Run aggressive GC to make sure finalizer fires
	for i := 0; i < 10; i++ {
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}

	select {
	case <-finalizerFired:
		// Good — r1's payload is GC-reclaimable after ring-buffer eviction.
	case <-time.After(2 * time.Second):
		t.Fatalf(
			"GC retention leak in ring-buffer eviction: finalizer for evicted record never fired. " +
				"The backing array slot was not niled, keeping the evicted *Record reachable.",
		)
	}
}
