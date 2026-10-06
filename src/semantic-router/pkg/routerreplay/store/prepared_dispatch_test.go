package store

import (
	"encoding/json"
	"testing"
)

func TestPreparedDispatchReceiptRoundTripAndClone(t *testing.T) {
	original := Record{RouteDiagnostics: &RouteDiagnostics{
		PreparedDispatch: &PreparedDispatchReceipt{
			Version:    1,
			WireFormat: "openai.chat.v1",
			SHA256:     "9c90b5338b604509298558ddecb91acee8c3d5e8a21999f818399387388d8b4b",
			ByteLength: 137,
		},
	}}

	encoded, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal record: %v", err)
	}
	var decoded Record
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal record: %v", err)
	}
	if decoded.RouteDiagnostics == nil || decoded.RouteDiagnostics.PreparedDispatch == nil {
		t.Fatal("prepared dispatch receipt was lost during JSON round trip")
	}
	if got := *decoded.RouteDiagnostics.PreparedDispatch; got != *original.RouteDiagnostics.PreparedDispatch {
		t.Fatalf("prepared dispatch receipt changed: got %+v want %+v", got, *original.RouteDiagnostics.PreparedDispatch)
	}

	cloned := cloneRecord(decoded)
	cloned.RouteDiagnostics.PreparedDispatch.SHA256 = "changed"
	if decoded.RouteDiagnostics.PreparedDispatch.SHA256 == "changed" {
		t.Fatal("clone shares the prepared dispatch receipt pointer")
	}
}
