package cache

import (
	"container/heap"
	"sort"
	"sync"
	"time"
)

// EvictionPolicy defines the interface for cache eviction policies
type EvictionPolicy interface {
	SelectVictim(entries []CacheEntry) int
}

// OptimizedEvictionPolicy extends EvictionPolicy with O(1) operations
type OptimizedEvictionPolicy interface {
	EvictionPolicy
	OnInsert(entryIndex int, requestID string)
	OnRemove(entryIndex int, requestID string)
	Evict() int
}

// OptimizedEvictionPolicyWithAccess extends OptimizedEvictionPolicy with access tracking
type OptimizedEvictionPolicyWithAccess interface {
	OptimizedEvictionPolicy
	OnAccess(entryIndex int, requestID string)
}

// =====================================
// Doubly-Linked List (internal)
// =====================================

type dlNode struct {
	entryIndex int
	requestID  string
	prev       *dlNode
	next       *dlNode
	freq       int64
}

type dlList struct {
	head *dlNode
	tail *dlNode
	size int
}

func newDLList() *dlList {
	head := &dlNode{}
	tail := &dlNode{}
	head.next = tail
	tail.prev = head
	return &dlList{head: head, tail: tail, size: 0}
}

func (dl *dlList) addToFront(node *dlNode) {
	node.prev = dl.head
	node.next = dl.head.next
	dl.head.next.prev = node
	dl.head.next = node
	dl.size++
}

func (dl *dlList) addToBack(node *dlNode) {
	node.next = dl.tail
	node.prev = dl.tail.prev
	dl.tail.prev.next = node
	dl.tail.prev = node
	dl.size++
}

func (dl *dlList) remove(node *dlNode) {
	if node.prev == nil || node.next == nil {
		return
	}
	node.prev.next = node.next
	node.next.prev = node.prev
	node.prev = nil
	node.next = nil
	dl.size--
}

func (dl *dlList) moveToFront(node *dlNode) {
	dl.remove(node)
	dl.addToFront(node)
}

func (dl *dlList) removeLast() *dlNode {
	if dl.size == 0 {
		return nil
	}
	node := dl.tail.prev
	dl.remove(node)
	return node
}

func (dl *dlList) removeFirst() *dlNode {
	if dl.size == 0 {
		return nil
	}
	node := dl.head.next
	dl.remove(node)
	return node
}

func (dl *dlList) isEmpty() bool {
	return dl.size == 0
}

// =====================================
// FIFO Policy - O(1) eviction
// =====================================

type FIFOPolicy struct {
	mu       sync.RWMutex
	list     *dlList
	nodeMap  map[string]*dlNode
	indexMap map[int]*dlNode
}

func NewFIFOPolicy() *FIFOPolicy {
	return &FIFOPolicy{
		list:     newDLList(),
		nodeMap:  make(map[string]*dlNode),
		indexMap: make(map[int]*dlNode),
	}
}

func (p *FIFOPolicy) OnInsert(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if oldNode, ok := p.nodeMap[requestID]; ok {
		p.list.remove(oldNode)
		delete(p.indexMap, oldNode.entryIndex)
		delete(p.nodeMap, requestID)
	}

	node := &dlNode{entryIndex: entryIndex, requestID: requestID}
	p.list.addToBack(node)
	p.nodeMap[requestID] = node
	p.indexMap[entryIndex] = node
}

func (p *FIFOPolicy) OnRemove(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, ok := p.nodeMap[requestID]; ok {
		p.list.remove(node)
		delete(p.nodeMap, requestID)
		delete(p.indexMap, node.entryIndex)
	}
}

func (p *FIFOPolicy) SelectVictim(entries []CacheEntry) int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.list == nil || p.list.isEmpty() {
		return selectOldestByTimestamp(entries)
	}

	node := p.list.head.next
	if node == p.list.tail {
		return -1
	}
	return node.entryIndex
}

func (p *FIFOPolicy) Evict() int {
	p.mu.Lock()
	defer p.mu.Unlock()

	node := p.list.removeFirst()
	if node == nil {
		return -1
	}
	delete(p.nodeMap, node.requestID)
	delete(p.indexMap, node.entryIndex)
	return node.entryIndex
}

func (p *FIFOPolicy) UpdateIndex(requestID string, oldIdx, newIdx int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, ok := p.nodeMap[requestID]; ok {
		delete(p.indexMap, node.entryIndex)
		node.entryIndex = newIdx
		p.indexMap[newIdx] = node
	}
}

// =====================================
// LRU Policy - O(1) eviction
// =====================================

type LRUPolicy struct {
	mu       sync.RWMutex
	list     *dlList
	nodeMap  map[string]*dlNode
	indexMap map[int]*dlNode
}

func NewLRUPolicy() *LRUPolicy {
	return &LRUPolicy{
		list:     newDLList(),
		nodeMap:  make(map[string]*dlNode),
		indexMap: make(map[int]*dlNode),
	}
}

func (p *LRUPolicy) OnAccess(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, ok := p.nodeMap[requestID]; ok {
		p.list.moveToFront(node)
		if node.entryIndex != entryIndex {
			delete(p.indexMap, node.entryIndex)
			node.entryIndex = entryIndex
			p.indexMap[entryIndex] = node
		}
	}
}

func (p *LRUPolicy) OnInsert(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if oldNode, ok := p.nodeMap[requestID]; ok {
		p.list.remove(oldNode)
		delete(p.indexMap, oldNode.entryIndex)
		delete(p.nodeMap, requestID)
	}

	node := &dlNode{entryIndex: entryIndex, requestID: requestID}
	p.list.addToFront(node)
	p.nodeMap[requestID] = node
	p.indexMap[entryIndex] = node
}

func (p *LRUPolicy) OnRemove(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, ok := p.nodeMap[requestID]; ok {
		p.list.remove(node)
		delete(p.nodeMap, requestID)
		delete(p.indexMap, node.entryIndex)
	}
}

func (p *LRUPolicy) SelectVictim(entries []CacheEntry) int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.list == nil || p.list.isEmpty() {
		return selectLeastRecentlyUsed(entries)
	}

	node := p.list.tail.prev
	if node == p.list.head {
		return -1
	}
	return node.entryIndex
}

func (p *LRUPolicy) Evict() int {
	p.mu.Lock()
	defer p.mu.Unlock()

	node := p.list.removeLast()
	if node == nil {
		return -1
	}
	delete(p.nodeMap, node.requestID)
	delete(p.indexMap, node.entryIndex)
	return node.entryIndex
}

func (p *LRUPolicy) UpdateIndex(requestID string, oldIdx, newIdx int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, ok := p.nodeMap[requestID]; ok {
		delete(p.indexMap, node.entryIndex)
		node.entryIndex = newIdx
		p.indexMap[newIdx] = node
	}
}

func (p *LRUPolicy) RebuildFromEntries(entries []CacheEntry) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.list = newDLList()
	p.nodeMap = make(map[string]*dlNode)
	p.indexMap = make(map[int]*dlNode)

	if len(entries) == 0 {
		return
	}

	type entryWithIdx struct {
		idx          int
		lastAccessAt time.Time
		requestID    string
	}
	sortedEntries := make([]entryWithIdx, len(entries))
	for i, e := range entries {
		sortedEntries[i] = entryWithIdx{idx: i, lastAccessAt: e.LastAccessAt, requestID: e.RequestID}
	}

	sort.Slice(sortedEntries, func(i, j int) bool {
		return sortedEntries[i].lastAccessAt.Before(sortedEntries[j].lastAccessAt)
	})

	for _, e := range sortedEntries {
		node := &dlNode{entryIndex: e.idx, requestID: e.requestID}
		p.list.addToFront(node)
		p.nodeMap[e.requestID] = node
		p.indexMap[e.idx] = node
	}
}

// =====================================
// LFU Policy - O(1) eviction
// =====================================

type LFUPolicy struct {
	mu         sync.RWMutex
	nodeMap    map[string]*dlNode
	indexMap   map[int]*dlNode
	freqBucket map[int64]*dlList
	minFreq    int64
}

func NewLFUPolicy() *LFUPolicy {
	return &LFUPolicy{
		nodeMap:    make(map[string]*dlNode),
		indexMap:   make(map[int]*dlNode),
		freqBucket: make(map[int64]*dlList),
		minFreq:    0,
	}
}

func (p *LFUPolicy) OnAccess(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	node, ok := p.nodeMap[requestID]
	if !ok {
		return
	}

	oldFreq := node.freq
	if bucket, ok := p.freqBucket[oldFreq]; ok {
		bucket.remove(node)
		if oldFreq == p.minFreq && bucket.isEmpty() {
			p.minFreq++
		}
	}

	node.freq++
	if _, ok := p.freqBucket[node.freq]; !ok {
		p.freqBucket[node.freq] = newDLList()
	}
	p.freqBucket[node.freq].addToFront(node)

	if node.entryIndex != entryIndex {
		delete(p.indexMap, node.entryIndex)
		node.entryIndex = entryIndex
		p.indexMap[entryIndex] = node
	}
}

func (p *LFUPolicy) OnInsert(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if oldNode, ok := p.nodeMap[requestID]; ok {
		if bucket, ok := p.freqBucket[oldNode.freq]; ok {
			bucket.remove(oldNode)
		}
		delete(p.indexMap, oldNode.entryIndex)
		delete(p.nodeMap, requestID)
	}

	node := &dlNode{entryIndex: entryIndex, requestID: requestID, freq: 1}
	if _, ok := p.freqBucket[1]; !ok {
		p.freqBucket[1] = newDLList()
	}
	p.freqBucket[1].addToFront(node)
	p.nodeMap[requestID] = node
	p.indexMap[entryIndex] = node
	p.minFreq = 1
}

func (p *LFUPolicy) OnRemove(entryIndex int, requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, ok := p.nodeMap[requestID]; ok {
		if bucket, ok := p.freqBucket[node.freq]; ok {
			bucket.remove(node)
		}
		delete(p.nodeMap, requestID)
		delete(p.indexMap, node.entryIndex)
	}
}

func (p *LFUPolicy) SelectVictim(entries []CacheEntry) int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.freqBucket == nil {
		return selectLeastFrequentlyUsed(entries)
	}

	bucket, ok := p.freqBucket[p.minFreq]
	if !ok || bucket.isEmpty() {
		found := false
		for f := p.minFreq; f <= p.minFreq+1000; f++ {
			if b, ok := p.freqBucket[f]; ok && !b.isEmpty() {
				bucket = b
				found = true
				break
			}
		}
		if !found || bucket == nil || bucket.isEmpty() {
			return selectLeastFrequentlyUsed(entries)
		}
	}

	node := bucket.tail.prev
	if node == bucket.head {
		return -1
	}
	return node.entryIndex
}

func (p *LFUPolicy) Evict() int {
	p.mu.Lock()
	defer p.mu.Unlock()

	bucket, ok := p.freqBucket[p.minFreq]
	if !ok || bucket.isEmpty() {
		found := false
		for f := p.minFreq; f <= p.minFreq+1000; f++ {
			if b, ok := p.freqBucket[f]; ok && !b.isEmpty() {
				bucket = b
				p.minFreq = f
				found = true
				break
			}
		}
		if !found || bucket == nil || bucket.isEmpty() {
			return -1
		}
	}

	node := bucket.removeLast()
	if node == nil {
		return -1
	}

	delete(p.nodeMap, node.requestID)
	delete(p.indexMap, node.entryIndex)
	return node.entryIndex
}

func (p *LFUPolicy) UpdateIndex(requestID string, oldIdx, newIdx int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node, ok := p.nodeMap[requestID]; ok {
		delete(p.indexMap, node.entryIndex)
		node.entryIndex = newIdx
		p.indexMap[newIdx] = node
	}
}

// =====================================
// Expiration Heap for O(k) TTL Cleanup
// =====================================

type expirationEntry struct {
	expiresAt  time.Time
	requestID  string
	entryIndex int
	index      int
}

type ExpirationHeap struct {
	mu      sync.RWMutex
	heap    []*expirationEntry
	nodeMap map[string]*expirationEntry
}

func NewExpirationHeap() *ExpirationHeap {
	return &ExpirationHeap{
		heap:    make([]*expirationEntry, 0),
		nodeMap: make(map[string]*expirationEntry),
	}
}

func (h *ExpirationHeap) Len() int { return len(h.heap) }

func (h *ExpirationHeap) Less(i, j int) bool {
	return h.heap[i].expiresAt.Before(h.heap[j].expiresAt)
}

func (h *ExpirationHeap) Swap(i, j int) {
	h.heap[i], h.heap[j] = h.heap[j], h.heap[i]
	h.heap[i].index = i
	h.heap[j].index = j
}

func (h *ExpirationHeap) Push(x interface{}) {
	entry := x.(*expirationEntry)
	entry.index = len(h.heap)
	h.heap = append(h.heap, entry)
}

func (h *ExpirationHeap) Pop() interface{} {
	old := h.heap
	n := len(old)
	entry := old[n-1]
	old[n-1] = nil
	entry.index = -1
	h.heap = old[0 : n-1]
	return entry
}

func (h *ExpirationHeap) Add(requestID string, entryIndex int, expiresAt time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if old, ok := h.nodeMap[requestID]; ok {
		heap.Remove(h, old.index)
		delete(h.nodeMap, requestID)
	}

	entry := &expirationEntry{
		expiresAt:  expiresAt,
		requestID:  requestID,
		entryIndex: entryIndex,
	}
	heap.Push(h, entry)
	h.nodeMap[requestID] = entry
}

func (h *ExpirationHeap) Remove(requestID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if entry, ok := h.nodeMap[requestID]; ok {
		if entry.index >= 0 && entry.index < len(h.heap) {
			heap.Remove(h, entry.index)
		}
		delete(h.nodeMap, requestID)
	}
}

func (h *ExpirationHeap) PeekNext() (requestID string, entryIndex int, expiresAt time.Time, ok bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.heap) == 0 {
		return "", -1, time.Time{}, false
	}
	entry := h.heap[0]
	return entry.requestID, entry.entryIndex, entry.expiresAt, true
}

func (h *ExpirationHeap) PopExpired(now time.Time) []string {
	h.mu.Lock()
	defer h.mu.Unlock()

	var expired []string
	for len(h.heap) > 0 && !h.heap[0].expiresAt.After(now) {
		entry := heap.Pop(h).(*expirationEntry)
		delete(h.nodeMap, entry.requestID)
		expired = append(expired, entry.requestID)
	}
	return expired
}

func (h *ExpirationHeap) UpdateExpiration(requestID string, newExpiresAt time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if entry, ok := h.nodeMap[requestID]; ok {
		entry.expiresAt = newExpiresAt
		heap.Fix(h, entry.index)
	}
}

func (h *ExpirationHeap) UpdateIndex(requestID string, newIndex int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if entry, ok := h.nodeMap[requestID]; ok {
		entry.entryIndex = newIndex
	}
}

func (h *ExpirationHeap) Size() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.heap)
}

// =====================================
// Fallback functions for empty policies
// =====================================

func selectOldestByTimestamp(entries []CacheEntry) int {
	if len(entries) == 0 {
		return -1
	}
	oldestIdx := 0
	for i := 1; i < len(entries); i++ {
		if entries[i].Timestamp.Before(entries[oldestIdx].Timestamp) {
			oldestIdx = i
		}
	}
	return oldestIdx
}

func selectLeastRecentlyUsed(entries []CacheEntry) int {
	if len(entries) == 0 {
		return -1
	}
	oldestIdx := 0
	for i := 1; i < len(entries); i++ {
		if entries[i].LastAccessAt.Before(entries[oldestIdx].LastAccessAt) {
			oldestIdx = i
		}
	}
	return oldestIdx
}

func selectLeastFrequentlyUsed(entries []CacheEntry) int {
	if len(entries) == 0 {
		return -1
	}
	victimIdx := 0
	for i := 1; i < len(entries); i++ {
		if entries[i].HitCount < entries[victimIdx].HitCount {
			victimIdx = i
		} else if entries[i].HitCount == entries[victimIdx].HitCount {
			if entries[i].LastAccessAt.Before(entries[victimIdx].LastAccessAt) {
				victimIdx = i
			}
		}
	}
	return victimIdx
}
