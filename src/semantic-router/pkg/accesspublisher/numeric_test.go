package accesspublisher

import (
	"math"
	"testing"
	"time"
)

func TestDatabasePositiveUint64Bounds(t *testing.T) {
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
			got, err := databasePositiveUint64(test.value, "value")
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("databasePositiveUint64(%d) = %d, %v; want %d, error=%t", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}

func TestPostgresBigintBounds(t *testing.T) {
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
			got, err := postgresBigint(test.value, "value")
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("postgresBigint(%d) = %d, %v; want %d, error=%t", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}

func fixturePostgresBigint(value uint64) int64 {
	converted, err := postgresBigint(value, "fixture value")
	if err != nil {
		panic(err)
	}
	return converted
}

func fixtureMillisecondOffset(value uint64) time.Duration {
	return time.Duration(fixturePostgresBigint(value)) * time.Millisecond
}
