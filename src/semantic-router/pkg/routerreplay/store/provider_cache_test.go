package store

import (
	"context"
	"reflect"
	"testing"
)

func TestProviderCacheAccountingPersistence(t *testing.T) {
	tokens, cost := 0, 0.0
	accounting := &ProviderCacheAccounting{State: "partial", CacheRead: TokenCostComponent{Tokens: &tokens, Cost: &cost, Provenance: "authoritative"}}
	record := Record{ProviderCache: accounting}
	prepared, err := newPostgresInsertRecord(record)
	if err != nil {
		t.Fatal(err)
	}
	row := postgresRecordRow{providerCacheJSON: prepared.providerCacheJSON, signalsJSON: []byte("{}")}
	decoded, err := row.decode()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded.ProviderCache, accounting) {
		t.Fatalf("Postgres JSON roundtrip lost evidence: %#v", decoded.ProviderCache)
	}
	if decoded.ProviderCache.CacheWrite.Tokens != nil {
		t.Fatal("missing writes became zero")
	}
	memory := NewMemoryStore(10, 0)
	id, err := memory.Add(context.Background(), record)
	if err != nil {
		t.Fatal(err)
	}
	if err := memory.UpdateUsageCost(context.Background(), id, UsageCost{ProviderCache: accounting}); err != nil {
		t.Fatal(err)
	}
	*accounting.CacheRead.Tokens = 999
	got, _, err := memory.Get(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if *got.ProviderCache.CacheRead.Tokens != 0 {
		t.Fatal("update retained caller's pointer")
	}
	*got.ProviderCache.CacheRead.Cost = 100
	again, _, err := memory.Get(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if *again.ProviderCache.CacheRead.Cost != 0 {
		t.Fatal("read exposed stored pointer")
	}
}
