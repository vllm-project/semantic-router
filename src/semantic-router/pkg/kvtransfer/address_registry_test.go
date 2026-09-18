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
