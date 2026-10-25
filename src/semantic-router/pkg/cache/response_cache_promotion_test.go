package cache

import (
	"context"
	"testing"
	"time"
)

type promotionTestStore struct {
	*serviceTestStore
	result CacheResult
}

func (s *promotionTestStore) LookupExact(context.Context, ExactLookup) (CacheResult, error) {
	s.exactLookup.Add(1)
	return s.result, nil
}

func TestResponseCacheServicePromotionPreservesFreshness(t *testing.T) {
	tests := []struct {
		name     string
		age      time.Duration
		ageKnown bool
		wantHit  bool
	}{
		{"fresh", 10 * time.Second, true, true},
		{"stale", 10 * time.Minute, true, false},
		{"unknown", 0, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expiresAt := time.Now().Add(time.Hour)
			store := &promotionTestStore{
				serviceTestStore: newServiceTestStore(),
				result: CacheResult{
					Found: true, ResponseBody: []byte("cached"),
					Age: tt.age, AgeKnown: tt.ageKnown, ExpiresAt: expiresAt,
				},
			}
			service := NewResponseCacheService(store, DefaultResponseCacheServiceOptions())
			identity := serviceTestIdentity(tt.name)
			ctx := context.Background()
			promoted, err := service.LookupExact(ctx, ExactLookup{Identity: identity})
			if err != nil || !promoted.Found || promoted.Source != CacheSourceL2 {
				t.Fatalf("promotion = %#v, err = %v", promoted, err)
			}
			warm, err := service.LookupExact(ctx, ExactLookup{Identity: identity})
			if err != nil || !warm.Found || warm.Source != CacheSourceL1 {
				t.Fatalf("warm lookup = %#v, err = %v", warm, err)
			}
			if warm.AgeKnown != tt.ageKnown || (tt.ageKnown && warm.Age < tt.age) {
				t.Fatalf("L1 lost original age: %#v", warm)
			}
			if !tt.ageKnown && warm.Age != 0 {
				t.Fatalf("unknown age must remain unknown: %#v", warm)
			}
			if store.exactLookup.Load() != 1 {
				t.Fatal("warm lookup did not use L1")
			}
			maxAge := time.Minute
			bounded, err := service.LookupExact(ctx, ExactLookup{Identity: identity, MaxAge: &maxAge})
			if err != nil || bounded.Found != tt.wantHit {
				t.Fatalf("max-age lookup = %#v, err = %v, want hit %v", bounded, err, tt.wantHit)
			}
		})
	}
}

func TestExactL1PromotionRetainsBackendExpiry(t *testing.T) {
	cache := newExactL1(1, time.Minute)
	expiresAt := time.Now().Add(10 * time.Second)
	cache.put("key", CacheResult{
		ResponseBody: []byte("cached"), Age: time.Minute, AgeKnown: true, ExpiresAt: expiresAt,
	}, TTL(time.Hour))
	result, found := cache.get("key", nil)
	if !found || !result.ExpiresAt.Equal(expiresAt) {
		t.Fatalf("L1 extended backend expiry: %#v", result)
	}
}

func TestResponseCacheServiceLocalWriteHasKnownAge(t *testing.T) {
	service := NewResponseCacheService(newServiceTestStore(), DefaultResponseCacheServiceOptions())
	identity := serviceTestIdentity("local-write")
	ctx := context.Background()
	if err := service.StoreExact(ctx, CacheWrite{
		Identity: identity, ResponseBody: []byte("new"), TTL: TTL(time.Minute),
	}); err != nil {
		t.Fatal(err)
	}
	maxAge := time.Minute
	result, err := service.LookupExact(ctx, ExactLookup{Identity: identity, MaxAge: &maxAge})
	if err != nil || !result.Found || !result.AgeKnown || result.Source != CacheSourceL1 {
		t.Fatalf("new local entry should be fresh: %#v, err = %v", result, err)
	}
}
