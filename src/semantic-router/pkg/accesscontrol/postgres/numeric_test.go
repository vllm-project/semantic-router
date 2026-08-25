package postgres

import (
	"math"
	"testing"
)

func TestPositiveUint64Bounds(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		value   int64
		want    uint64
		wantErr bool
	}{
		{name: "negative", value: -1, wantErr: true},
		{name: "zero", value: 0, wantErr: true},
		{name: "maximum signed", value: math.MaxInt64, want: math.MaxInt64},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := positiveUint64(test.value, "value")
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("positiveUint64(%d) = %d, %v; want %d, error=%t", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}

func TestPostgresInt64Bounds(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		value   uint64
		want    int64
		wantErr bool
	}{
		{name: "zero", value: 0, want: 0},
		{name: "maximum signed", value: math.MaxInt64, want: math.MaxInt64},
		{name: "above maximum signed", value: uint64(math.MaxInt64) + 1, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := postgresInt64(test.value, "value")
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("postgresInt64(%d) = %d, %v; want %d, error=%t", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}

func TestNonNegativeUint32Bounds(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		value   int64
		want    uint32
		wantErr bool
	}{
		{name: "negative", value: -1, wantErr: true},
		{name: "zero", value: 0, want: 0},
		{name: "maximum unsigned", value: math.MaxUint32, want: math.MaxUint32},
		{name: "above maximum unsigned", value: int64(math.MaxUint32) + 1, wantErr: true},
		{name: "maximum signed", value: math.MaxInt64, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := nonNegativeUint32(test.value, "value")
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("nonNegativeUint32(%d) = %d, %v; want %d, error=%t", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}
