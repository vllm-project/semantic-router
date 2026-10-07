package modelservice

import (
	"container/list"
	"encoding/binary"
	"hash/maphash"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
)

// ResultCacheEnv sets how many results each served model keeps (0 disables).
const ResultCacheEnv = "VLLM_SRUN_RESULT_CACHE"

const defaultResultCacheEntries = 4096

// ResultCacheEntries is the per-model result cache size from ResultCacheEnv.
func ResultCacheEntries() int {
	if value, err := strconv.Atoi(os.Getenv(ResultCacheEnv)); err == nil && value >= 0 {
		return value
	}
	return defaultResultCacheEntries
}

// cacheKey identifies a request by surface and a 128-bit hash of everything
// that can change its result; deadlines and request IDs are excluded.
type cacheKey struct {
	surface byte
	hi, lo  uint64
}

const (
	surfaceClassify byte = iota + 1
	surfaceDecide
)

var cacheSeeds = [2]maphash.Seed{maphash.MakeSeed(), maphash.MakeSeed()}

// keyWriter feeds two independently seeded hashes with length-prefixed fields.
type keyWriter struct {
	surface byte
	hashes  [2]maphash.Hash
}

func newKeyWriter(surface byte) *keyWriter {
	w := &keyWriter{surface: surface}
	for i := range w.hashes {
		w.hashes[i].SetSeed(cacheSeeds[i])
	}
	return w
}

func (w *keyWriter) string(value string) {
	w.int(len(value))
	for i := range w.hashes {
		_, _ = w.hashes[i].WriteString(value)
	}
}

func (w *keyWriter) int(value int) { w.word(uint64(int64(value))) } //nolint:gosec // a bit pattern for hashing, never converted back

func (w *keyWriter) float(value float64) { w.word(math.Float64bits(value)) }

func (w *keyWriter) word(value uint64) {
	var buffer [8]byte
	binary.LittleEndian.PutUint64(buffer[:], value)
	for i := range w.hashes {
		_, _ = w.hashes[i].Write(buffer[:])
	}
}

func (w *keyWriter) key() cacheKey {
	return cacheKey{surface: w.surface, hi: w.hashes[0].Sum64(), lo: w.hashes[1].Sum64()}
}

func classifyKey(request ClassifyRequest) cacheKey {
	w := newKeyWriter(surfaceClassify)
	w.string(request.Head)
	w.string(request.Overflow)
	w.int(request.MaxTokens)
	if request.Window != nil {
		w.int(request.Window.Tokens)
		w.int(request.Window.Overlap)
	} else {
		w.int(-1)
	}
	if request.Threshold != nil {
		w.float(*request.Threshold)
	} else {
		w.int(-1)
	}
	w.int(len(request.Inputs))
	for _, input := range request.Inputs {
		w.string(input.Text)
		w.string(input.TextPair)
		w.string(input.Context)
		w.string(input.Question)
		w.string(input.Answer)
	}
	return w.key()
}

func decideKey(request Request) cacheKey {
	w := newKeyWriter(surfaceDecide)
	w.string(request.State)
	w.int(request.MaxTokens)
	if request.Parts != nil {
		names := make([]string, 0, len(request.Parts))
		for name := range request.Parts {
			names = append(names, name)
		}
		sort.Strings(names)
		w.int(len(names))
		for _, name := range names {
			w.string(name)
			w.string(request.Parts[name])
		}
	} else {
		w.int(-1)
	}
	w.int(len(request.Questions))
	for _, question := range request.Questions {
		w.string(question.ID)
		w.string(question.Type)
		w.string(question.Instructions)
		w.string(question.Preset)
		w.string(question.Head)
		if question.Truncate {
			w.int(1)
		} else {
			w.int(0)
		}
		if question.Threshold != nil {
			w.float(*question.Threshold)
		} else {
			w.int(-1)
		}
		for _, options := range [][]Choice{question.Choices, question.Labels} {
			w.int(len(options))
			for _, option := range options {
				w.string(option.Key)
				w.string(option.Description)
			}
		}
		w.int(len(question.Levels))
		for _, level := range question.Levels {
			w.string(level)
		}
	}
	return w.key()
}

// resultCache is a bounded LRU of complete results for one served model.
// Results are shared read-only with every caller that hits them.
type resultCache struct {
	mu      sync.Mutex
	limit   int
	entries map[cacheKey]*list.Element
	order   *list.List
}

type cacheEntry struct {
	key   cacheKey
	value any
}

func newResultCache(limit int) *resultCache {
	return &resultCache{limit: limit, entries: make(map[cacheKey]*list.Element), order: list.New()}
}

// active returns the cache, or nil when caching is disabled, so callers skip
// hashing the request.
func (c *resultCache) active() *resultCache {
	if c == nil || c.limit == 0 {
		return nil
	}
	return c
}

func (c *resultCache) get(key cacheKey) (any, bool) {
	if c == nil || c.limit == 0 {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	element, ok := c.entries[key]
	if !ok {
		return nil, false
	}
	c.order.MoveToFront(element)
	return element.Value.(*cacheEntry).value, true
}

func (c *resultCache) put(key cacheKey, value any) {
	if c == nil || c.limit == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if element, ok := c.entries[key]; ok {
		element.Value.(*cacheEntry).value = value
		c.order.MoveToFront(element)
		return
	}
	c.entries[key] = c.order.PushFront(&cacheEntry{key: key, value: value})
	for c.order.Len() > c.limit {
		oldest := c.order.Back()
		c.order.Remove(oldest)
		delete(c.entries, oldest.Value.(*cacheEntry).key)
	}
}

// reset drops every result; a restarted process may serve a changed model.
func (c *resultCache) reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = make(map[cacheKey]*list.Element)
	c.order.Init()
}
