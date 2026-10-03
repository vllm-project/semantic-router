package embedding

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"sync"
)

// VectorKey addresses one embedding: a representation (model content and
// numerics), an output view and the input bytes, hashed with SHA-256. The
// input itself is never retained.
type VectorKey [sha256.Size]byte

// Input kinds, so equal bytes of different modalities never collide.
const (
	InputText byte = iota + 1
	InputImage
	InputAudio
)

// NewVectorKey derives the key of input under a representation and view.
func NewVectorKey(representation string, options Options, kind byte, input []byte) VectorKey {
	hash := sha256.New()
	header := make([]byte, 0, 4*binary.MaxVarintLen64+1)
	header = binary.AppendVarint(header, int64(len(representation)))
	hash.Write(header)
	hash.Write([]byte(representation))
	header = binary.AppendVarint(header[:0], int64(options.Dimension))
	header = binary.AppendVarint(header, int64(options.Layer))
	header = append(header, kind)
	hash.Write(header)
	hash.Write(input)
	var key VectorKey
	hash.Sum(key[:0])
	return key
}

const vectorCacheShards = 32

// VectorCache is a bounded, sharded LRU of embedding vectors with in-flight
// coalescing: concurrent requests for one key make a single model call, so
// the consumers of one request that embed the same text share one vector.
// An embedding is a pure function of its key, so entries never expire; they
// are only evicted. Vectors are copied in and out, so callers own what they get.
type VectorCache struct {
	shards [vectorCacheShards]vectorShard
}

type vectorShard struct {
	mu       sync.Mutex
	entries  map[VectorKey]*list.Element
	order    *list.List // front is most recent
	bytes    int
	maxBytes int
	flights  map[VectorKey]*vectorFlight
}

type vectorEntry struct {
	key    VectorKey
	vector []float32
}

type vectorFlight struct {
	done   chan struct{}
	vector []float32
	err    error
}

// entryOverhead approximates the per-entry bookkeeping beyond the vector.
const entryOverhead = 128

// NewVectorCache returns a cache holding at most about maxBytes of vectors.
// A non-positive budget disables caching; coalescing still applies.
func NewVectorCache(maxBytes int) *VectorCache {
	c := &VectorCache{}
	for i := range c.shards {
		c.shards[i] = vectorShard{
			entries: make(map[VectorKey]*list.Element), order: list.New(),
			maxBytes: maxBytes / vectorCacheShards, flights: make(map[VectorKey]*vectorFlight),
		}
	}
	return c
}

func (c *VectorCache) shard(key VectorKey) *vectorShard {
	return &c.shards[key[0]%vectorCacheShards]
}

// Resolve returns one vector per key. Cached keys are answered at once; the
// rest are computed by one call of compute with their indexes, unless another
// caller is already computing a key, in which case this call waits for that
// result. If that other computation fails, the key is computed here.
func (c *VectorCache) Resolve(ctx context.Context, keys []VectorKey, compute func(context.Context, []int) ([][]float32, error)) ([][]float32, error) {
	vectors := make([][]float32, len(keys))
	var owned []int
	var flights []*vectorFlight
	type pending struct {
		index  int
		flight *vectorFlight
	}
	var waits []pending
	for i, key := range keys {
		shard := c.shard(key)
		shard.mu.Lock()
		switch vector, flight := shard.get(key); {
		case vector != nil:
			vectors[i] = vector
		case flight != nil:
			waits = append(waits, pending{index: i, flight: flight})
		default:
			flight = &vectorFlight{done: make(chan struct{})}
			shard.flights[key] = flight
			owned = append(owned, i)
			flights = append(flights, flight)
		}
		shard.mu.Unlock()
	}
	if len(owned) > 0 {
		computed, err := compute(ctx, owned)
		if err == nil && len(computed) != len(owned) {
			err = errVectorCount
		}
		for j, i := range owned {
			flight := flights[j]
			shard := c.shard(keys[i])
			shard.mu.Lock()
			delete(shard.flights, keys[i])
			if err == nil {
				flight.vector = shard.put(keys[i], computed[j])
				vectors[i] = copyVector(computed[j])
			} else {
				flight.err = err
			}
			shard.mu.Unlock()
			close(flight.done)
		}
		if err != nil {
			return nil, err
		}
	}
	var retry []int
	for _, wait := range waits {
		select {
		case <-wait.flight.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		if wait.flight.err != nil {
			retry = append(retry, wait.index)
			continue
		}
		vectors[wait.index] = copyVector(wait.flight.vector)
	}
	if len(retry) > 0 {
		computed, err := compute(ctx, retry)
		if err == nil && len(computed) != len(retry) {
			err = errVectorCount
		}
		if err != nil {
			return nil, err
		}
		for j, i := range retry {
			vectors[i] = copyVector(computed[j])
		}
	}
	return vectors, nil
}

// get returns a copy of a cached vector, or the flight computing it. The
// caller holds the lock.
func (s *vectorShard) get(key VectorKey) ([]float32, *vectorFlight) {
	if element, ok := s.entries[key]; ok {
		s.order.MoveToFront(element)
		return copyVector(element.Value.(*vectorEntry).vector), nil
	}
	return nil, s.flights[key]
}

// put stores a private copy of vector and returns it, evicting the least
// recently used entries over budget. The caller holds the lock.
func (s *vectorShard) put(key VectorKey, vector []float32) []float32 {
	stored := copyVector(vector)
	size := 4*len(stored) + entryOverhead
	if size > s.maxBytes {
		return stored
	}
	if element, ok := s.entries[key]; ok {
		s.order.MoveToFront(element)
		return element.Value.(*vectorEntry).vector
	}
	s.entries[key] = s.order.PushFront(&vectorEntry{key: key, vector: stored})
	s.bytes += size
	for s.bytes > s.maxBytes {
		oldest := s.order.Back()
		entry := oldest.Value.(*vectorEntry)
		s.order.Remove(oldest)
		delete(s.entries, entry.key)
		s.bytes -= 4*len(entry.vector) + entryOverhead
	}
	return stored
}

func copyVector(vector []float32) []float32 {
	return append([]float32(nil), vector...)
}

var errVectorCount = errors.New("embedding call returned a different number of vectors")
