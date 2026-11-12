package store

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// DecisionRanking mirrors decision.RankingTrace by hand, so a field added
// to the trace must be added here too or replay silently drops it.
func TestDecisionRankingMirrorsRankingTrace(t *testing.T) {
	if got, want := jsonTags(reflect.TypeOf(DecisionRanking{})), jsonTags(reflect.TypeOf(decision.RankingTrace{})); !reflect.DeepEqual(got, want) {
		t.Fatalf("replay mirror tags = %v, want %v", got, want)
	}
}

func jsonTags(typ reflect.Type) []string {
	tags := make([]string, 0, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		tags = append(tags, typ.Field(i).Tag.Get("json"))
	}
	return tags
}
