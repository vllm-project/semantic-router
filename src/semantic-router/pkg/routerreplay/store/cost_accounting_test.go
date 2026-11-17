package store

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

func costAccountingFixture() *CostAccounting {
	cost := 0.25
	return &CostAccounting{
		PricingSource: PricingSourceCatalogList, CallerLogin: true,
		BaselineSource: BaselineSourceConfigured, BaselineUnavailable: UnpricedCurrencyMismatch,
		RouterCalls: []RouterCallCost{{
			Kind: RouterCallKindLooper, Calls: 3, PromptTokens: 30, CompletionTokens: 9,
			Cost: &cost, Currency: "USD", UnpricedCalls: 1,
			UnpricedReasons: map[string]int{UnpricedPricingNotConfigured: 1},
		}},
	}
}

func TestMemoryStoreKeepsCostAccountingApartFromItsCaller(t *testing.T) {
	ctx := context.Background()
	memory := NewMemoryStore(10, 0)
	id, err := memory.Add(ctx, Record{ID: "cost"})
	if err != nil {
		t.Fatal(err)
	}
	accounting := costAccountingFixture()
	if err = memory.UpdateUsageCost(ctx, id, UsageCost{CostAccounting: accounting}); err != nil {
		t.Fatal(err)
	}
	*accounting.RouterCalls[0].Cost = 9
	accounting.RouterCalls[0].UnpricedReasons[UnpricedNoUsage] = 4
	got, _, err := memory.Get(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.CostAccounting, costAccountingFixture()) {
		t.Fatalf("stored cost accounting followed the caller's copy: %+v", got.CostAccounting)
	}
	if err = memory.UpdateUsageCost(ctx, id, UsageCost{}); err != nil {
		t.Fatal(err)
	}
	if got, _, _ = memory.Get(ctx, id); got.CostAccounting != nil {
		t.Fatalf("a usage update without cost accounting kept the old one: %+v", got.CostAccounting)
	}
}

func TestPostgresCostAccountingRowRoundTrip(t *testing.T) {
	input := postgresMetadataFixture()
	input.CostAccounting = costAccountingFixture()
	insert, err := newPostgresInsertRecord(input)
	if err != nil {
		t.Fatal(err)
	}
	row := postgresRecordRow{signalsJSON: []byte("{}")}
	index := columnIndex(t, "cost_accounting")
	*row.scanDestinations()[index].(*[]byte) = insert.args()[index].([]byte)
	output, err := row.decode()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(output.CostAccounting, input.CostAccounting) {
		t.Fatalf("cost accounting changed in a row: %+v", output.CostAccounting)
	}

	query := postgresCreateTableQuery(DefaultPostgresTableName)
	if !strings.Contains(query, "cost_accounting JSONB,") || !strings.Contains(query, "ADD COLUMN IF NOT EXISTS cost_accounting JSONB;") {
		t.Fatal("fresh and existing replay tables must gain nullable cost accounting")
	}
}
