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
	ageKnown     bool
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
	var age time.Duration
	if entry.ageKnown {
		age = now.Sub(entry.storedAt)
	}
	if (!entry.expiresAt.IsZero() && !now.Before(entry.expiresAt)) ||
		(maxAge != nil && (!entry.ageKnown || age > *maxAge)) {
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
		AgeKnown:     entry.ageKnown,
		ExpiresAt:    entry.expiresAt,
	}, true
}

func (c *exactL1) put(key string, result CacheResult, ttl TTLPolicy) {
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
	var storedAt time.Time
	if result.AgeKnown {
		storedAt = now.Add(-max(0, result.Age))
	}
	expiresAt := now.Add(effectiveTTL)
	if !result.ExpiresAt.IsZero() && result.ExpiresAt.Before(expiresAt) {
		expiresAt = result.ExpiresAt
	}
	entry := &exactL1Entry{
		key:          key,
		responseBody: append([]byte(nil), result.ResponseBody...),
		storedAt:     storedAt,
		ageKnown:     result.AgeKnown,
		expiresAt:    expiresAt,
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
