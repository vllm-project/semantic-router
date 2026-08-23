package postgres

import (
	"database/sql"
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestGCRABurstToleranceUsesInt64Storage(t *testing.T) {
	maximum := int64(math.MaxInt64)
	stored, err := encodeRateLimitRule(accesscontrol.RateLimitRule{
		GCRABurstTolerance: &maximum,
	})
	if err != nil {
		t.Fatal(err)
	}
	encoded, ok := stored.gcraBurstTolerance.(int64)
	if !ok || encoded != maximum {
		t.Fatalf("encoded burst tolerance = %#v, want int64(%d)", stored.gcraBurstTolerance, maximum)
	}

	var decoded accesscontrol.RateLimitRule
	if err := (scannedRateLimitRule{
		gcraBurstTolerance: sql.NullInt64{Int64: maximum, Valid: true},
	}).applyGCRA(&decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.GCRABurstTolerance == nil || *decoded.GCRABurstTolerance != maximum {
		t.Fatalf("decoded burst tolerance = %#v, want %d", decoded.GCRABurstTolerance, maximum)
	}
}
