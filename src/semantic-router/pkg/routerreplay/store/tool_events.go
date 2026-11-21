package store

import (
	"slices"
	"sync"
	"time"
)

// Tool event statuses: the tool ran and returned, it ran and failed, or it
// never ran because the person, a rule or a hook refused it.
const (
	ToolEventSucceeded = "succeeded"
	ToolEventFailed    = "failed"
	ToolEventDenied    = "denied"
)

// ToolEvent is one tool call as the harness that ran it reported it: the
// tool's name, its call id, when it ran and how it ended, and when it was put
// to approval. Router Replay joins it to the run's tool call of the same id in
// the same session. It never holds a tool's input or output.
type ToolEvent struct {
	SessionID  string `json:"session_id"`
	Harness    string `json:"harness,omitempty"`
	ToolCallID string `json:"tool_call_id"`
	Tool       string `json:"tool"`
	Status     string `json:"status"`
	// StartedAt is when the tool began, EndedAt when it ended or was refused,
	// and ApprovalRequestedAt when the harness put the call to approval.
	StartedAt           *time.Time `json:"started_at,omitempty"`
	EndedAt             *time.Time `json:"ended_at,omitempty"`
	ApprovalRequestedAt *time.Time `json:"approval_requested_at,omitempty"`
	// ReceivedAt is when the Router took the event: the store ages and evicts
	// events by it.
	ReceivedAt time.Time `json:"received_at"`
}

type toolEventKey struct{ session, call string }

// ToolEventStore keeps the tool events harness plugins report, one per
// session and call id, at most max of them and none received longer ago than
// ttl. Opened with a path it also keeps them in a file (tool_events_file.go).
type ToolEventStore struct {
	mu     sync.Mutex
	events map[toolEventKey]ToolEvent
	// order holds the keys oldest first by ReceivedAt.
	order []toolEventKey
	max   int
	ttl   time.Duration
	disk  *toolEventFile
	refs  int
}

// NewToolEventStore makes a store of its own, kept in memory.
func NewToolEventStore(maxEvents int, ttl time.Duration) *ToolEventStore {
	store := &ToolEventStore{events: map[toolEventKey]ToolEvent{}}
	store.setLimits(maxEvents, ttl)
	return store
}

func (s *ToolEventStore) setLimits(maxEvents int, ttl time.Duration) {
	s.max, s.ttl = max(maxEvents, 1), ttl
}

// Add keeps the events, each merged into the one already kept for its
// session and call: a later report fills in or replaces what it carries.
func (s *ToolEventStore) Add(events []ToolEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := make([]ToolEvent, 0, len(events))
	for _, event := range events {
		kept = append(kept, s.putLocked(event))
	}
	s.trimLocked(time.Now())
	if s.disk == nil {
		return nil
	}
	return s.disk.append(s, kept)
}

func (s *ToolEventStore) putLocked(event ToolEvent) ToolEvent {
	key := toolEventKey{event.SessionID, event.ToolCallID}
	if old, ok := s.events[key]; ok {
		event = mergeToolEvent(old, event)
		s.order = slices.DeleteFunc(s.order, func(one toolEventKey) bool { return one == key })
	}
	s.events[key] = event
	s.order = append(s.order, key)
	return event
}

func mergeToolEvent(old, event ToolEvent) ToolEvent {
	if event.Harness == "" {
		event.Harness = old.Harness
	}
	if event.Tool == "" {
		event.Tool = old.Tool
	}
	if event.Status == "" {
		event.Status = old.Status
	}
	if event.StartedAt == nil {
		event.StartedAt = old.StartedAt
	}
	if event.EndedAt == nil {
		event.EndedAt = old.EndedAt
	}
	if event.ApprovalRequestedAt == nil {
		event.ApprovalRequestedAt = old.ApprovalRequestedAt
	}
	return event
}

// trimLocked drops the events past the TTL, then the oldest past max.
func (s *ToolEventStore) trimLocked(now time.Time) {
	drop := 0
	for drop < len(s.order) && (len(s.order)-drop > s.max ||
		(s.ttl > 0 && s.events[s.order[drop]].ReceivedAt.Before(now.Add(-s.ttl)))) {
		delete(s.events, s.order[drop])
		drop++
	}
	s.order = s.order[drop:]
}

// Session returns the events kept for a session, by tool call id.
func (s *ToolEventStore) Session(sessionID string) map[string]ToolEvent {
	if s == nil || sessionID == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.trimLocked(time.Now())
	found := map[string]ToolEvent{}
	for key, event := range s.events {
		if key.session == sessionID {
			found[key.call] = event
		}
	}
	return found
}

// Close releases this user's share of a store opened with OpenToolEventStore;
// the last one writes the file whole and closes it. A store of its own has
// nothing to release.
func (s *ToolEventStore) Close() error {
	if s == nil {
		return nil
	}
	openToolEventStores.Lock()
	defer openToolEventStores.Unlock()
	if s.refs == 0 {
		return nil
	}
	if s.refs--; s.refs > 0 {
		return nil
	}
	for path, open := range openToolEventStores.byPath {
		if open == s {
			delete(openToolEventStores.byPath, path)
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.disk == nil {
		return nil
	}
	err := s.disk.close(s)
	s.disk = nil
	return err
}

// openToolEventStores lets every Router generation of a process share one
// store per path, the empty path for the one kept only in memory, so that a
// configuration reload keeps the events as it keeps durable replay records.
var openToolEventStores = struct {
	sync.Mutex
	byPath map[string]*ToolEventStore
}{byPath: map[string]*ToolEventStore{}}

// OpenToolEventStore opens the store kept at path, or in memory for an empty
// path, or shares the one this process has open there.
func OpenToolEventStore(path string, maxEvents int, ttl time.Duration) (*ToolEventStore, error) {
	openToolEventStores.Lock()
	defer openToolEventStores.Unlock()
	if store := openToolEventStores.byPath[path]; store != nil {
		store.mu.Lock()
		store.setLimits(maxEvents, ttl)
		store.mu.Unlock()
		store.refs++
		return store, nil
	}
	store := NewToolEventStore(maxEvents, ttl)
	if path != "" {
		disk, err := openToolEventFile(path, store)
		if err != nil {
			return nil, err
		}
		store.disk = disk
	}
	store.refs = 1
	openToolEventStores.byPath[path] = store
	return store, nil
}
