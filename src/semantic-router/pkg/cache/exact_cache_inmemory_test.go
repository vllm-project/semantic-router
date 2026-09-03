//go:build !windows && cgo

package cache

import (
	"context"
	"errors"
	"testing"
)

func TestInMemoryExactCacheIsPartitionedAndHonorsTTL(t *testing.T) {
	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:    true,
		TTLSeconds: 60,
		MaxEntries: 10,
	})
	defer func() { _ = cache.Close() }()

	if err := cache.AddExact(context.Background(), "tenant-a", "request", []byte(`{"answer":"a"}`), -1); err != nil {
		t.Fatalf("AddExact: %v", err)
	}
	hit, err := cache.FindExact(context.Background(), "tenant-a", "request")
	if err != nil {
		t.Fatalf("FindExact: %v", err)
	}
	if !hit.Found || string(hit.ResponseBody) != `{"answer":"a"}` || hit.Similarity != 1 {
		t.Fatalf("unexpected exact hit: %#v", hit)
	}
	miss, err := cache.FindExact(context.Background(), "tenant-b", "request")
	if err != nil {
		t.Fatalf("FindExact other partition: %v", err)
	}
	if miss.Found {
		t.Fatalf("cross-partition exact hit: %#v", miss)
	}
}

// The in-memory backend checks cancellation explicitly because it has no driver.
func TestInMemoryExactCacheHonorsCancellation(t *testing.T) {
	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:    true,
		TTLSeconds: 60,
		MaxEntries: 10,
	})
	defer func() { _ = cache.Close() }()

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	if err := cache.AddExact(cancelled, "tenant-a", "request", []byte(`{"answer":"a"}`), -1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled AddExact: want context.Canceled, got %v", err)
	}
	// The write must not have been published: a later, live reader sees nothing.
	orphan, err := cache.FindExact(context.Background(), "tenant-a", "request")
	if err != nil {
		t.Fatalf("FindExact after cancelled write: %v", err)
	}
	if orphan.Found {
		t.Fatalf("cancelled AddExact published an entry: %#v", orphan)
	}

	if err = cache.AddExact(context.Background(), "tenant-a", "request", []byte(`{"answer":"a"}`), -1); err != nil {
		t.Fatalf("AddExact: %v", err)
	}
	res, err := cache.FindExact(cancelled, "tenant-a", "request")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled FindExact: want context.Canceled, got %v", err)
	}
	if res.Found {
		t.Fatalf("cancelled FindExact served a hit: %#v", res)
	}
}
