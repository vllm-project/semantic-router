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
		{"strong", entity.ClStrong, true},
		{"  session  ", entity.ClSession, true},
		{"EVENTUALLY", entity.ClEventually, true},
		{"", entity.ClStrong, false},
		{"   ", entity.ClStrong, false},
		{"customized", entity.ClStrong, false},
	}

	for _, tc := range cases {
		got, ok := ParseConsistencyLevel(tc.name)
		if ok != tc.valid {
			t.Fatalf("ParseConsistencyLevel(%q) ok = %v, want %v", tc.name, ok, tc.valid)
		}
		if got != tc.want {
			t.Fatalf("ParseConsistencyLevel(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}
