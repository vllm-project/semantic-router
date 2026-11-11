package tokenaccounting

import (
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type countingTokenizer struct{ calls atomic.Int64 }

func (tokenizer *countingTokenizer) CountTokens(text string) int {
	tokenizer.calls.Add(1)
	return len(strings.Fields(text))
}

func historyRequest(texts ...string) *llmprotocol.Request {
	request := &llmprotocol.Request{}
	for _, value := range texts {
		request.Messages = append(request.Messages, llmprotocol.Message{
			Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{text(value)},
		})
	}
	return request
}

func longText(word string) string { return strings.Repeat(word+" ", cacheMinBytes) }

func countingCounter(t *testing.T, tokenizer Tokenizer, options ...Option) *Counter {
	t.Helper()
	registry, err := NewRegistry(Family{Name: "qwen3", Models: []string{testModel}, Tokenizer: tokenizer})
	if err != nil {
		t.Fatal(err)
	}
	return NewCounter(registry, options...)
}

func TestCacheCountsAResentHistoryOnce(t *testing.T) {
	tokenizer := &countingTokenizer{}
	counter := countingCounter(t, tokenizer)
	turn1 := historyRequest(longText("first"))
	turn2 := historyRequest(longText("first"), longText("second"))

	one := counter.Count(testModel, turn1, nil)
	two := counter.Count(testModel, turn2, nil)

	if calls := tokenizer.calls.Load(); calls != 2 {
		t.Fatalf("tokenizer ran %d times for two turns sharing a message, want 2", calls)
	}
	if two.Tokens-one.Tokens != cacheMinBytes+llmprotocol.InputMessageFramingTokens {
		t.Fatalf("cached count differs from a fresh one: turn1 %+v, turn2 %+v", one, two)
	}
	if stats := counter.CacheStats(); stats != (CacheStats{Entries: 2, Hits: 1, Misses: 2}) {
		t.Fatalf("CacheStats = %+v", stats)
	}
}

func TestCacheSkipsShortTextsAndEvictsTheLeastRecent(t *testing.T) {
	tokenizer := &countingTokenizer{}
	counter := countingCounter(t, tokenizer, WithCacheEntries(2))
	short := historyRequest("hello there")
	counter.Count(testModel, short, nil)
	counter.Count(testModel, short, nil)
	if calls, stats := tokenizer.calls.Load(), counter.CacheStats(); calls != 2 || stats.Entries != 0 {
		t.Fatalf("short text: %d tokenizer calls, %+v; want 2 calls and no entries", calls, stats)
	}

	a, b, c := historyRequest(longText("a")), historyRequest(longText("b")), historyRequest(longText("c"))
	for _, request := range []*llmprotocol.Request{a, b, a, c} {
		counter.Count(testModel, request, nil)
	}
	// a, b, a (hit), c evicts b, the least recently used.
	before := tokenizer.calls.Load()
	counter.Count(testModel, a, nil)
	if tokenizer.calls.Load() != before {
		t.Fatal("a was evicted although b was used less recently")
	}
	counter.Count(testModel, b, nil)
	if tokenizer.calls.Load() != before+1 {
		t.Fatal("b was not evicted")
	}
	if stats := counter.CacheStats(); stats.Entries != 2 {
		t.Fatalf("cache holds %d entries, want its bound of 2", stats.Entries)
	}
}

func TestCacheKeepsFamiliesApart(t *testing.T) {
	half := tokenizerFunc(func(text string) int { return len(strings.Fields(text)) / 2 })
	registry, err := NewRegistry(
		Family{Name: "words", Models: []string{"m-words"}, Tokenizer: wordTokenizer{}},
		Family{Name: "half", Models: []string{"m-half"}, Tokenizer: half},
	)
	if err != nil {
		t.Fatal(err)
	}
	counter := NewCounter(registry)
	request := historyRequest(longText("shared"))
	words := counter.Count("m-words", request, nil)
	halves := counter.Count("m-half", request, nil)
	if words.Tokens-halves.Tokens != cacheMinBytes/2 {
		t.Fatalf("families shared a cached count: words %+v, half %+v", words, halves)
	}
}

func TestCacheCanBeDisabled(t *testing.T) {
	tokenizer := &countingTokenizer{}
	counter := countingCounter(t, tokenizer, WithCacheEntries(0))
	request := historyRequest(longText("again"))
	counter.Count(testModel, request, nil)
	counter.Count(testModel, request, nil)
	if calls := tokenizer.calls.Load(); calls != 2 {
		t.Fatalf("disabled cache: tokenizer ran %d times, want 2", calls)
	}
	if stats := counter.CacheStats(); stats != (CacheStats{}) {
		t.Fatalf("disabled cache reports %+v", stats)
	}
}

type tokenizerFunc func(string) int

func (f tokenizerFunc) CountTokens(text string) int { return f(text) }
