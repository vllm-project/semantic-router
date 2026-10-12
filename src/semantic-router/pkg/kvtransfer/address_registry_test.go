package kvtransfer

import (
	"context"
	"testing"
	"time"
)

func TestAddressKeyEscapesColons(t *testing.T) {
	key := AddressKey("tenant:1", "recipe::session")
	want := "kv_addr:tenant%3A1:recipe%3A%3Asession"
	if key != want {
		t.Fatalf("AddressKey() = %q, want %q", key, want)
	}
}

func TestAddressKeyDistinguishesLiteralPercent3AFromColon(t *testing.T) {
	withColon := AddressKey("ns", "recipe::a:b")
	withLiteral := AddressKey("ns", "recipe::a%3Ab")
	if withColon == withLiteral {
		t.Fatalf("AddressKey collision: %q == %q", withColon, withLiteral)
	}
	wantColon := "kv_addr:ns:recipe%3A%3Aa%3Ab"
	wantLiteral := "kv_addr:ns:recipe%3A%3Aa%253Ab"
	if withColon != wantColon {
		t.Fatalf("AddressKey(colon) = %q, want %q", withColon, wantColon)
	}
	if withLiteral != wantLiteral {
		t.Fatalf("AddressKey(literal %%3A) = %q, want %q", withLiteral, wantLiteral)
	}
}

func TestMemoryAddressRegistryLiteralPercent3ADoesNotOverwriteColonSession(t *testing.T) {
	reg := NewMemoryAddressRegistry()
	colonRecord := AddressRecord{
		SessionID: "recipe::a:b",
		SourcePod: "10.0.1.5:8000",
		Model:     "qwen3-14b",
		Namespace: "ns",
		TurnCount: 1,
	}
	literalRecord := AddressRecord{
		SessionID: "recipe::a%3Ab",
		SourcePod: "10.0.1.6:8000",
		Model:     "qwen3-14b",
		Namespace: "ns",
		TurnCount: 2,
	}
	if err := reg.Write(context.Background(), colonRecord); err != nil {
		t.Fatalf("Write(colon) error = %v", err)
	}
	if err := reg.Write(context.Background(), literalRecord); err != nil {
		t.Fatalf("Write(literal) error = %v", err)
	}
	gotColon, err := reg.Lookup(context.Background(), "ns", "recipe::a:b")
	if err != nil || gotColon == nil {
		t.Fatalf("Lookup(colon) = (%v, %v), want record", gotColon, err)
	}
	if gotColon.SourcePod != colonRecord.SourcePod || gotColon.TurnCount != 1 {
		t.Fatalf("Lookup(colon) = %+v, want original colon record", gotColon)
	}
	gotLiteral, err := reg.Lookup(context.Background(), "ns", "recipe::a%3Ab")
	if err != nil || gotLiteral == nil {
		t.Fatalf("Lookup(literal) = (%v, %v), want record", gotLiteral, err)
	}
	if gotLiteral.SourcePod != literalRecord.SourcePod || gotLiteral.TurnCount != 2 {
		t.Fatalf("Lookup(literal) = %+v, want original literal record", gotLiteral)
	}
}

func TestMemoryAddressRegistryWriteAndLookup(t *testing.T) {
	reg := NewMemoryAddressRegistry()
	record := AddressRecord{
		SessionID: "recipe::sess-1",
		SourcePod: "10.0.1.5:8000",
		Model:     "qwen3-14b",
		Namespace: "abc123",
		TurnCount: 5,
		UpdatedAt: time.Unix(1_700_000_000, 0).UTC(),
	}
	if err := reg.Write(context.Background(), record); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	got, err := reg.Lookup(context.Background(), "abc123", "recipe::sess-1")
	if err != nil {
		t.Fatalf("Lookup() error = %v", err)
	}
	if got == nil {
		t.Fatal("Lookup() = nil, want record")
	}
	if got.SourcePod != record.SourcePod || got.Model != record.Model || got.TurnCount != 5 {
		t.Fatalf("Lookup() = %+v, want %+v", got, record)
	}
}

func TestMemoryAddressRegistryLookupMissReturnsNil(t *testing.T) {
	reg := NewMemoryAddressRegistry()
	got, err := reg.Lookup(context.Background(), "tenant-a", "missing")
	if err != nil {
		t.Fatalf("Lookup() error = %v", err)
	}
	if got != nil {
		t.Fatalf("Lookup() = %+v, want nil", got)
	}
}

func TestValidateAddressRecordRejectsMissingFields(t *testing.T) {
	err := validateAddressRecord(AddressRecord{
		SessionID: "sess",
		Model:     "qwen3-14b",
		TurnCount: 1,
	})
	if err == nil {
		t.Fatal("validateAddressRecord() = nil, want error")
	}
}

func TestNoopAddressRegistryIsSafe(t *testing.T) {
	var reg NoopAddressRegistry
	if err := reg.Write(context.Background(), AddressRecord{}); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	got, err := reg.Lookup(context.Background(), "tenant", "sess")
	if err != nil || got != nil {
		t.Fatalf("Lookup() = (%v, %v), want (nil, nil)", got, err)
	}
}
