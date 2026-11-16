package store

import (
	"fmt"
	"sync"
)

// openMemoryStores lets every Router generation of a process share the
// memory store of a key, so that a configuration reload keeps its records.
var openMemoryStores = struct {
	sync.Mutex
	byKey map[string]*SharedMemoryStore
}{byKey: map[string]*SharedMemoryStore{}}

// SharedMemoryStore is a memory store the generations that opened its key
// hold together; it is dropped when the last of them closes it.
type SharedMemoryStore struct {
	*MemoryStore

	key  string
	refs int
}

// OpenMemoryStore shares the memory store this process keeps under key with
// the same TTL, taking maxRecords as its new limit, or opens a new one. A
// store whose TTL changed starts empty.
func OpenMemoryStore(key string, maxRecords, ttlSeconds int) *SharedMemoryStore {
	key = fmt.Sprintf("%s ttl=%d", key, ttlSeconds)
	openMemoryStores.Lock()
	defer openMemoryStores.Unlock()
	if store := openMemoryStores.byKey[key]; store != nil {
		store.refs++
		store.SetMaxRecords(maxRecords)
		return store
	}
	store := &SharedMemoryStore{MemoryStore: NewMemoryStore(maxRecords, ttlSeconds), key: key, refs: 1}
	openMemoryStores.byKey[key] = store
	return store
}

// Close lets go of one generation's hold, and drops the records with the
// last one.
func (s *SharedMemoryStore) Close() error {
	openMemoryStores.Lock()
	defer openMemoryStores.Unlock()
	if s.refs == 0 {
		return nil
	}
	if s.refs--; s.refs == 0 {
		delete(openMemoryStores.byKey, s.key)
	}
	return nil
}
