package store

import (
	"context"
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
