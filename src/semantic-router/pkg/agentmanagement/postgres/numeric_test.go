package postgres

import (
	"errors"
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func TestResourceRevisionUint64Bounds(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		value   int64
		want    uint64
		wantErr error
	}{
		{name: "negative", value: -1, wantErr: agentmanagement.ErrInvalid},
		{name: "zero", value: 0, wantErr: agentmanagement.ErrInvalid},
		{name: "maximum signed", value: math.MaxInt64, want: math.MaxInt64},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := resourceRevisionUint64(test.value)
			if !errors.Is(err, test.wantErr) || got != test.want {
				t.Fatalf("resourceRevisionUint64(%d) = %d, %v; want %d, %v", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}

func TestResourceRevisionInt64Bounds(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		value   uint64
		want    int64
		wantErr error
	}{
		{name: "zero", value: 0, wantErr: agentmanagement.ErrConflict},
		{name: "maximum signed", value: math.MaxInt64, want: math.MaxInt64},
		{name: "above maximum signed", value: uint64(math.MaxInt64) + 1, wantErr: agentmanagement.ErrConflict},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := resourceRevisionInt64(test.value)
			if !errors.Is(err, test.wantErr) || got != test.want {
				t.Fatalf("resourceRevisionInt64(%d) = %d, %v; want %d, %v", test.value, got, err, test.want, test.wantErr)
			}
		})
	}
}
