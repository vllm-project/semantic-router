package accesscontrol

import (
	"reflect"
	"testing"
)

func TestEffectiveModelPatternsUsesDirectKeyGrantAsOverride(t *testing.T) {
	got := effectiveModelPatterns([]string{"vllm-sr/mom-v1-lite"}, []string{"*", "vllm-sr/mom-v1-ultra"})
	want := []string{"vllm-sr/mom-v1-lite"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("effectiveModelPatterns() = %#v, want %#v", got, want)
	}
}

func TestEffectiveModelPatternsInheritsWhenKeyHasNoDirectGrant(t *testing.T) {
	got := effectiveModelPatterns(nil, []string{"vllm-sr/*", "vllm-sr/*"})
	want := []string{"vllm-sr/*"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("effectiveModelPatterns() = %#v, want %#v", got, want)
	}
}
