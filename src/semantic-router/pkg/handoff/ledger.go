/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package handoff

import (
	"container/list"
	"crypto/sha256"
	"sync"
	"time"
)

// Admission is the ledger outcome for one validated envelope.
type Admission string

const (
	// AdmissionNew is the first sighting of an active handoff ID.
	AdmissionNew Admission = "new"
	// AdmissionDuplicate is an idempotent retry of identical contents.
	AdmissionDuplicate Admission = "duplicate"
	// AdmissionConflict reuses an active handoff ID with different contents.
	AdmissionConflict Admission = "conflict"
	// AdmissionCancelRecorded records a cancelled envelope for its handoff ID.
	AdmissionCancelRecorded Admission = "cancel_recorded"
	// AdmissionCancelled is an active envelope for an already cancelled ID.
	AdmissionCancelled Admission = "cancelled"
)

// Ledger is a bounded, process-local record of handoff IDs seen before their
// expiry. It is not durable task state: records disappear on restart, after
// their latest seen expiry, or when capacity forces the oldest record out.
type Ledger struct {
	mu       sync.Mutex
	capacity int
	entries  map[string]*list.Element
	order    *list.List
}

type ledgerEntry struct {
	id        string
	digest    [sha256.Size]byte
	cancelled bool
	expiresAt time.Time
}

// NewLedger returns a ledger holding at most capacity handoff IDs.
func NewLedger(capacity int) *Ledger {
	if capacity < 1 {
		capacity = 1
	}
	return &Ledger{
		capacity: capacity,
		entries:  make(map[string]*list.Element),
		order:    list.New(),
	}
}

// Admit records a validated envelope and reports how it relates to earlier
// envelopes with the same handoff ID.
func (l *Ledger) Admit(envelope *Envelope, now time.Time) Admission {
	digest := envelope.Digest()
	cancelled := envelope.State == StateCancelled

	l.mu.Lock()
	defer l.mu.Unlock()

	entry := l.lookup(envelope.HandoffID, now)
	switch {
	case entry == nil:
		l.insert(&ledgerEntry{
			id:        envelope.HandoffID,
			digest:    digest,
			cancelled: cancelled,
			expiresAt: envelope.ExpiresAt,
		})
		if cancelled {
			return AdmissionCancelRecorded
		}
		return AdmissionNew
	case cancelled:
		entry.cancelled = true
		if envelope.ExpiresAt.After(entry.expiresAt) {
			entry.expiresAt = envelope.ExpiresAt
		}
		return AdmissionCancelRecorded
	case entry.cancelled:
		return AdmissionCancelled
	case entry.digest == digest:
		return AdmissionDuplicate
	default:
		return AdmissionConflict
	}
}

// Len reports the number of retained records, including expired ones that
// have not been looked up since expiry.
func (l *Ledger) Len() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.order.Len()
}

func (l *Ledger) lookup(id string, now time.Time) *ledgerEntry {
	element, ok := l.entries[id]
	if !ok {
		return nil
	}
	entry := element.Value.(*ledgerEntry)
	if !now.Before(entry.expiresAt) {
		l.remove(element)
		return nil
	}
	return entry
}

func (l *Ledger) insert(entry *ledgerEntry) {
	for l.order.Len() >= l.capacity {
		l.remove(l.order.Front())
	}
	l.entries[entry.id] = l.order.PushBack(entry)
}

func (l *Ledger) remove(element *list.Element) {
	delete(l.entries, element.Value.(*ledgerEntry).id)
	l.order.Remove(element)
}
