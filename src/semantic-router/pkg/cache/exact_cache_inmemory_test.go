//go:build !windows && cgo

package cache

import "testing"

func TestInMemoryExactCacheIsPartitionedAndHonorsTTL(t *testing.T) {
	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:    true,
		TTLSeconds: 60,
		MaxEntries: 10,
	})
	defer func() { _ = cache.Close() }()

	if err := cache.AddExact("tenant-a", "request", []byte(`{"answer":"a"}`), -1); err != nil {
		t.Fatalf("AddExact: %v", err)
	}
	hit, err := cache.FindExact("tenant-a", "request")
	if err != nil {
		t.Fatalf("FindExact: %v", err)
	}
	if !hit.Found || string(hit.ResponseBody) != `{"answer":"a"}` || hit.Similarity != 1 {
		t.Fatalf("unexpected exact hit: %#v", hit)
	}
	miss, err := cache.FindExact("tenant-b", "request")
	if err != nil {
		t.Fatalf("FindExact other partition: %v", err)
	}
	if miss.Found {
		t.Fatalf("cross-partition exact hit: %#v", miss)
	}
}
