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

// Embedded is one input's vector and whether the input was truncated to fit
// the model's input budget.
type Embedded struct {
	Vector    []float32
	Truncated bool
}

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
	key      VectorKey
	embedded Embedded
}

type vectorFlight struct {
	done     chan struct{}
	embedded Embedded
	err      error
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

// Resolve returns one result per key. Cached keys are answered at once; the
// rest are computed by one call of compute with their indexes, unless another
// caller is already computing a key, in which case this call waits for that
// result. If that other computation fails, the key is computed here.
func (c *VectorCache) Resolve(ctx context.Context, keys []VectorKey, compute func(context.Context, []int) ([]Embedded, error)) ([]Embedded, error) {
	return c.resolve(ctx, keys, compute, true)
}

// ResolveAlone is Resolve without waiting on other callers: a key another
// caller is computing is computed again here. A caller inside a request
// bundle uses it, because the bundle sends both calls' inputs in one round
// trip, while a waiting participant would hold the bundle until its window
// ends.
func (c *VectorCache) ResolveAlone(ctx context.Context, keys []VectorKey, compute func(context.Context, []int) ([]Embedded, error)) ([]Embedded, error) {
	return c.resolve(ctx, keys, compute, false)
}

func (c *VectorCache) resolve(ctx context.Context, keys []VectorKey, compute func(context.Context, []int) ([]Embedded, error), wait bool) ([]Embedded, error) {
	vectors := make([]Embedded, len(keys))
	var owned, unshared []int
	var flights []*vectorFlight
	type pending struct {
		index  int
		flight *vectorFlight
	}
	var waits []pending
	for i, key := range keys {
		shard := c.shard(key)
		shard.mu.Lock()
		switch cached, flight := shard.get(key); {
		case cached.Vector != nil:
			vectors[i] = cached
		case flight != nil && wait:
			waits = append(waits, pending{index: i, flight: flight})
		case flight != nil:
			unshared = append(unshared, i)
		default:
			flight = &vectorFlight{done: make(chan struct{})}
			shard.flights[key] = flight
			owned = append(owned, i)
			flights = append(flights, flight)
		}
		shard.mu.Unlock()
	}
	if len(owned)+len(unshared) > 0 {
		computed, err := compute(ctx, append(owned, unshared...))
		if err == nil && len(computed) != len(owned)+len(unshared) {
			err = errVectorCount
		}
		for j, i := range owned {
			flight := flights[j]
			shard := c.shard(keys[i])
			shard.mu.Lock()
			delete(shard.flights, keys[i])
			if err == nil {
				flight.embedded = shard.put(keys[i], computed[j])
				vectors[i] = copyEmbedded(computed[j])
			} else {
				flight.err = err
			}
			shard.mu.Unlock()
			close(flight.done)
		}
		if err != nil {
			return nil, err
		}
		for j, i := range unshared {
			vectors[i] = copyEmbedded(computed[len(owned)+j])
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
		vectors[wait.index] = copyEmbedded(wait.flight.embedded)
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
			vectors[i] = copyEmbedded(computed[j])
		}
	}
	return vectors, nil
}

// get returns a copy of a cached result, or the flight computing it. The
// caller holds the lock.
func (s *vectorShard) get(key VectorKey) (Embedded, *vectorFlight) {
	if element, ok := s.entries[key]; ok {
		s.order.MoveToFront(element)
		return copyEmbedded(element.Value.(*vectorEntry).embedded), nil
	}
	return Embedded{}, s.flights[key]
}

// put stores a private copy of embedded and returns it, evicting the least
// recently used entries over budget. The caller holds the lock.
func (s *vectorShard) put(key VectorKey, embedded Embedded) Embedded {
	stored := copyEmbedded(embedded)
	size := 4*len(stored.Vector) + entryOverhead
	if size > s.maxBytes {
		return stored
	}
	if element, ok := s.entries[key]; ok {
		s.order.MoveToFront(element)
		return element.Value.(*vectorEntry).embedded
	}
	s.entries[key] = s.order.PushFront(&vectorEntry{key: key, embedded: stored})
	s.bytes += size
	for s.bytes > s.maxBytes {
		oldest := s.order.Back()
		entry := oldest.Value.(*vectorEntry)
		s.order.Remove(oldest)
		delete(s.entries, entry.key)
		s.bytes -= 4*len(entry.embedded.Vector) + entryOverhead
	}
	return stored
}

func copyEmbedded(embedded Embedded) Embedded {
	embedded.Vector = append([]float32(nil), embedded.Vector...)
	return embedded
}

var errVectorCount = errors.New("embedding call returned a different number of vectors")
