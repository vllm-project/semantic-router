package store

import (
	"fmt"
	"sync"
)

// openMemoryStores lets every Router generation of a process share the
// memory store of a key, so that a configuration reload keeps its records.
var openMemoryStores = struct {
	sync.Mutex
	byKey map[string]*sharedMemoryEntry
}{byKey: map[string]*sharedMemoryEntry{}}

// sharedMemoryEntry is the memory store of a key and the number of
// generations that hold it.
type sharedMemoryEntry struct {
	*MemoryStore

	key  string
	refs int
}

// SharedMemoryStore is one generation's hold on the memory store of a key,
// which the generations that opened it share; the store is dropped when the
// last of them closes it.
type SharedMemoryStore struct {
	*MemoryStore

	entry      *sharedMemoryEntry
	maxRecords int
	closed     bool
}

// OpenMemoryStore shares the memory store this process keeps under key with
// the same TTL, or opens a new one with maxRecords. A store whose TTL changed
// starts empty. A shared store takes maxRecords only when Activate says the
// generation serves, so a reload that is rejected leaves its records alone.
func OpenMemoryStore(key string, maxRecords, ttlSeconds int) *SharedMemoryStore {
	key = fmt.Sprintf("%s ttl=%d", key, ttlSeconds)
	openMemoryStores.Lock()
	defer openMemoryStores.Unlock()
	entry := openMemoryStores.byKey[key]
	if entry == nil {
		entry = &sharedMemoryEntry{MemoryStore: NewMemoryStore(maxRecords, ttlSeconds), key: key}
		openMemoryStores.byKey[key] = entry
	}
	entry.refs++
	return &SharedMemoryStore{MemoryStore: entry.MemoryStore, entry: entry, maxRecords: maxRecords}
}

// Activate gives the shared store this generation's max_records, as the
// generation starts to serve.
func (s *SharedMemoryStore) Activate() {
	s.SetMaxRecords(s.maxRecords)
}

// Close lets go of this generation's hold, and drops the records with the
// last one.
func (s *SharedMemoryStore) Close() error {
	openMemoryStores.Lock()
	defer openMemoryStores.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.entry.refs--; s.entry.refs == 0 {
		delete(openMemoryStores.byKey, s.entry.key)
	}
	return nil
}
