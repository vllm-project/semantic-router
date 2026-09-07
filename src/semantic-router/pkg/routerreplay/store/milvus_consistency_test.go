package store

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestMilvusConsistencyLevel(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  entity.ConsistencyLevel
		valid bool
	}{
		{name: "default", want: entity.ClSession, valid: true},
		{name: "case insensitive", value: "  sTrOnG ", want: entity.ClStrong, valid: true},
		{name: "bounded", value: "bounded", want: entity.ClBounded, valid: true},
		{name: "eventually", value: "eventually", want: entity.ClEventually, valid: true},
		{name: "invalid defaults to session", value: "unknown", want: entity.ClSession, valid: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, valid := milvusConsistencyLevel(test.value)
			if got != test.want || valid != test.valid {
				t.Fatalf("milvusConsistencyLevel(%q) = (%v, %t), want (%v, %t)", test.value, got, valid, test.want, test.valid)
			}
		})
	}
}
