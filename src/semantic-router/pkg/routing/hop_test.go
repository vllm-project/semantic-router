package routing

import (
	"context"
	"testing"
)

func TestHopTravelsInTheContextOnly(t *testing.T) {
	if _, ok := HopFrom(context.Background()); ok {
		t.Fatal("a plain context is not a hop")
	}
	hop := Hop{Decision: "fusion", Recipe: "default", Iteration: 2}
	if got, ok := HopFrom(WithHop(context.Background(), hop)); !ok || got != hop {
		t.Fatalf("HopFrom = %+v, %v", got, ok)
	}
}
