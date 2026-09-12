package candle_binding

import (
	"errors"
	"testing"
)

func TestEmbeddingDimensionStateValidation(t *testing.T) {
	cases := []struct {
		name       string
		state      uint32
		native     int
		dimensions []int
		want       DimensionState
	}{
		{"not loaded", 0, 0, nil, DimensionStateNotLoaded},
		{"model native need not be first", 1, 960, []int{320, 960, 640}, DimensionStateAvailable},
		{"native only", 1, 960, []int{960}, DimensionStateAvailable},
		{"unknown state", 2, 0, nil, ""},
		{"unloaded native width", 0, 960, nil, ""},
		{"unloaded declared widths", 0, 0, []int{960}, ""},
		{"missing native width", 1, 0, []int{960}, ""},
		{"empty is not unrestricted", 1, 960, nil, ""},
		{"native absent from list", 1, 960, []int{320}, ""},
		{"zero width", 1, 960, []int{960, 0}, ""},
		{"negative width", 1, 960, []int{960, -1}, ""},
		{"duplicate width", 1, 960, []int{960, 960}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dimensionStateFromNative(tc.state, tc.native, tc.dimensions)
			if tc.want == "" {
				if !errors.Is(err, ErrMalformedCapabilities) {
					t.Fatalf("got %q, %v; want typed malformed-capabilities error", got, err)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("got %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}
