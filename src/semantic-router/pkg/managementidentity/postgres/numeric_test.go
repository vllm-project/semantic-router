package postgres

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestDatabaseRevisionBounds(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		value   int64
		want    accesscontrol.Revision
		wantErr bool
	}{
		{name: "negative", value: -1, wantErr: true},
		{name: "zero", value: 0, wantErr: true},
		{name: "maximum signed", value: math.MaxInt64, want: math.MaxInt64},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := databaseRevision(test.value, "value")
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("databaseRevision(%d) = %d, %v; want %d, error=%t", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}

func TestDatabaseRevisionUint64Bounds(t *testing.T) {
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
			got, err := databaseRevisionUint64(test.value, "value")
			if (err != nil) != test.wantErr || got != test.want {
				t.Fatalf("databaseRevisionUint64(%d) = %d, %v; want %d, error=%t", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}
