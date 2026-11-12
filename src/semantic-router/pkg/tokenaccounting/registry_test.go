package tokenaccounting

import (
	"errors"
	"strings"
	"testing"
)

type wordTokenizer struct{}

func (wordTokenizer) CountTokens(text string) int { return len(strings.Fields(text)) }

func TestNewRegistryRejectsAmbiguousOrIncompleteFamilies(t *testing.T) {
	cases := []struct {
		name     string
		families []Family
		want     string
	}{
		{"unnamed", []Family{{Models: []string{"m"}, Tokenizer: wordTokenizer{}}}, "has no name"},
		{"twice", []Family{
			{Name: "qwen3", Models: []string{"a"}, Tokenizer: wordTokenizer{}},
			{Name: " qwen3 ", Models: []string{"b"}, Tokenizer: wordTokenizer{}},
		}, "defined twice"},
		{"no tokenizer", []Family{{Name: "qwen3", Models: []string{"a"}}}, "has no tokenizer"},
		{"no models", []Family{{Name: "qwen3", Tokenizer: wordTokenizer{}}}, "lists no models"},
		{"empty model", []Family{{Name: "qwen3", Models: []string{" "}, Tokenizer: wordTokenizer{}}}, "empty model name"},
		{"shared model", []Family{
			{Name: "qwen3", Models: []string{"a"}, Tokenizer: wordTokenizer{}},
			{Name: "llama3", Models: []string{"b", " a"}, Tokenizer: wordTokenizer{}},
		}, `model "a" is in tokenizer families "qwen3" and "llama3"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry(tc.families...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("NewRegistry error = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}

func TestRegistryResolvesExactTrimmedModelNames(t *testing.T) {
	registry, err := NewRegistry(
		Family{Name: "qwen3", Models: []string{" Qwen/Qwen3-8B "}, Tokenizer: wordTokenizer{}},
		Family{Name: "llama3", Models: []string{"meta-llama/Llama-3.1-8B-Instruct"}, Tokenizer: wordTokenizer{}},
	)
	if err != nil {
		t.Fatal(err)
	}
	family, err := registry.Resolve("Qwen/Qwen3-8B ")
	if err != nil || family.Name != "qwen3" {
		t.Fatalf("Resolve = %+v, %v; want qwen3", family, err)
	}
	for _, model := range []string{"Qwen/Qwen3-8b", "Qwen/Qwen3", "", "vllm-sr/auto"} {
		if _, err := registry.Resolve(model); !errors.Is(err, ErrNoTokenizer) {
			t.Fatalf("Resolve(%q) err = %v, want ErrNoTokenizer", model, err)
		}
	}
	var none *Registry
	if _, err := none.Resolve("Qwen/Qwen3-8B"); !errors.Is(err, ErrNoTokenizer) {
		t.Fatalf("nil registry resolved a model: %v", err)
	}
}
