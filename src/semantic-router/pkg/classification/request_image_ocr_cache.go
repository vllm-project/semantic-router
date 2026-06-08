package classification

import "sync"

// requestImageOCRCache memoizes OCR extraction for an image within the
// lifetime of one EvaluateAllSignalsWithContext call. Mirrors the embedding
// cache pattern so multiple evaluators that need the same OCR text do not
// independently call the (potentially expensive) OCR backend.
type requestImageOCRCache struct {
	mu      sync.Mutex
	entries map[string]*requestImageOCRCacheEntry
}

type requestImageOCRCacheEntry struct {
	once sync.Once
	text string
	err  error
}

func newRequestImageOCRCache() *requestImageOCRCache {
	return &requestImageOCRCache{
		entries: make(map[string]*requestImageOCRCacheEntry),
	}
}

// resolve returns the OCR text for imageRef, computing it via compute on the
// first call for this imageRef. A nil receiver is treated as cache-disabled
// and compute runs unconditionally.
func (c *requestImageOCRCache) resolve(imageRef string, compute func() (string, error)) (string, error) {
	if c == nil {
		return compute()
	}
	c.mu.Lock()
	entry, ok := c.entries[imageRef]
	if !ok {
		entry = &requestImageOCRCacheEntry{}
		c.entries[imageRef] = entry
	}
	c.mu.Unlock()
	entry.once.Do(func() {
		entry.text, entry.err = compute()
	})
	return entry.text, entry.err
}
