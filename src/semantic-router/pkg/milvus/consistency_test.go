package milvus

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestParseConsistencyLevel(t *testing.T) {
	cases := []struct {
		name  string
		want  entity.ConsistencyLevel
		valid bool
	}{
		{"Strong", entity.ClStrong, true},
		{"Session", entity.ClSession, true},
		{"Bounded", entity.ClBounded, true},
		{"Eventually", entity.ClEventually, true},
		{"  eVENTUALLY  ", entity.ClEventually, true}, // case-insensitive, trimmed
		{"", entity.ClStrong, false},
		{"   ", entity.ClStrong, false},
		{"customized", entity.ClStrong, false},
	}

	for _, tc := range cases {
		got, ok := ParseConsistencyLevel(tc.name)
		if got != tc.want || ok != tc.valid {
			t.Errorf("ParseConsistencyLevel(%q) = (%v, %v), want (%v, %v)", tc.name, got, ok, tc.want, tc.valid)
		}
	}
}
