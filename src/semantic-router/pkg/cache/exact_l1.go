package cache

import (
	"container/list"
	"sync"
	"time"
)

type exactL1Entry struct {
	key          string
	responseBody []byte
	storedAt     time.Time
	expiresAt    time.Time
}

type exactL1 struct {
	mu         sync.Mutex
	maxEntries int
	maxTTL     time.Duration
	items      map[string]*list.Element
	order      *list.List
}

func newExactL1(maxEntries int, maxTTL time.Duration) *exactL1 {
	if maxEntries < 0 {
		maxEntries = 0
	}
	return &exactL1{
		maxEntries: maxEntries,
		maxTTL:     maxTTL,
		items:      make(map[string]*list.Element, maxEntries),
		order:      list.New(),
	}
}

func (c *exactL1) get(key string, maxAge *time.Duration) (CacheResult, bool) {
	if c == nil || c.maxEntries == 0 {
		return CacheResult{}, false
	}
	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()
	element, ok := c.items[key]
	if !ok {
		return CacheResult{}, false
	}
	entry := element.Value.(*exactL1Entry)
	age := now.Sub(entry.storedAt)
	if (!entry.expiresAt.IsZero() && !now.Before(entry.expiresAt)) ||
		(maxAge != nil && age > *maxAge) {
		c.removeElement(element)
		return CacheResult{}, false
	}
	c.order.MoveToFront(element)
	return CacheResult{
		ResponseBody: append([]byte(nil), entry.responseBody...),
		Found:        true,
		HitKind:      HitKindExact,
		Source:       CacheSourceL1,
		Age:          age,
		AgeKnown:     true,
		ExpiresAt:    entry.expiresAt,
	}, true
}

func (c *exactL1) put(key string, responseBody []byte, ttl TTLPolicy) {
	if c == nil || c.maxEntries == 0 || ttl.NoStore {
		return
	}
	effectiveTTL := ttl.Duration
	if ttl.UseDefault || effectiveTTL <= 0 ||
		(c.maxTTL > 0 && effectiveTTL > c.maxTTL) {
		effectiveTTL = c.maxTTL
	}
	if effectiveTTL <= 0 {
		return
	}
	now := time.Now()
	entry := &exactL1Entry{
		key:          key,
		responseBody: append([]byte(nil), responseBody...),
		storedAt:     now,
		expiresAt:    now.Add(effectiveTTL),
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if existing, ok := c.items[key]; ok {
		existing.Value = entry
		c.order.MoveToFront(existing)
		return
	}
	c.items[key] = c.order.PushFront(entry)
	for c.order.Len() > c.maxEntries {
		c.removeElement(c.order.Back())
	}
}

func (c *exactL1) clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*list.Element, c.maxEntries)
	c.order.Init()
}

func (c *exactL1) len() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}

func (c *exactL1) removeElement(element *list.Element) {
	if element == nil {
		return
	}
	entry := element.Value.(*exactL1Entry)
	delete(c.items, entry.key)
	c.order.Remove(element)
}
