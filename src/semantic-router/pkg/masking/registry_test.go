package masking

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/maskingregistry"
)

func registryFixture(t *testing.T) (*config.MaskingPluginConfig, *maskingregistry.Identity) {
	t.Helper()
	dir := t.TempDir()
	key := filepath.Join(dir, "key")
	if e := os.WriteFile(key, make([]byte, 32), 0o600); e != nil {
		t.Fatal(e)
	}
	id, e := maskingregistry.Open("test", config.MaskingIdentityNamespace{KeyFile: key, Scope: config.MaskingIdentityScope, Registry: config.MaskingRegistryConfig{Backend: "file", Path: filepath.Join(dir, "log")}})
	if e != nil {
		t.Fatal(e)
	}
	return &config.MaskingPluginConfig{Enabled: true, Signal: "pii", IdentityNamespace: "test"}, id
}

func allocator(t *testing.T, cfg *config.MaskingPluginConfig, id *maskingregistry.Identity, p string) *Allocator {
	t.Helper()
	a, e := NewRegistryAllocator(cfg, id, p, "")
	if e != nil {
		t.Fatal(e)
	}
	return a
}

func textRequest(texts ...string) *llmprotocol.Request {
	r := &llmprotocol.Request{}
	for _, s := range texts {
		r.Messages = append(r.Messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: s}}})
	}
	return r
}

func TestRegistryEchoBeforeValueAcrossRequests(t *testing.T) {
	cfg, id := registryFixture(t)
	r := textRequest("Alice")
	if _, e := Apply(r, allocator(t, cfg, id, "p"), scanForValue("PERSON", "Alice")); e != nil {
		t.Fatal(e)
	}
	if r.Messages[0].Content[0].Text != "[PERSON_0]" {
		t.Fatal(r)
	}
	// A compacted echo arrives before its original, with a detector that also
	// misreads PERSON in the echo. Membership already exists from request one.
	r = textRequest("[PERSON_0]", "Alice")
	scan := func(text string) ([]Span, error) {
		if text == "Alice" {
			return []Span{{EntityType: "PERSON", Start: 0, End: 5}}, nil
		}
		return []Span{{EntityType: "PERSON", Start: 1, End: 7}}, nil
	}
	if _, e := Apply(r, allocator(t, cfg, id, "p"), scan); e != nil {
		t.Fatal(e)
	}
	for _, m := range r.Messages {
		if m.Content[0].Text != "[PERSON_0]" {
			t.Fatal(m)
		}
	}
}

func TestRegistryWholeDispatchReservation(t *testing.T) {
	for _, jsonLeaf := range []bool{false, true} {
		t.Run(fmt.Sprint(jsonLeaf), func(t *testing.T) {
			cfg, id := registryFixture(t)
			r := textRequest("Alice", "[PERSON_0]")
			if jsonLeaf {
				r.Messages[1].Content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{Arguments: `{"later":"[PERSON_0]"}`}}}
			}
			if _, e := Apply(r, allocator(t, cfg, id, "p"), scanForValue("PERSON", "Alice")); e != nil {
				t.Fatal(e)
			}
			if r.Messages[0].Content[0].Text != "[PERSON_1]" {
				t.Fatal("later literal was allocated", r)
			}
			next := allocator(t, cfg, id, "p")
			masked, _, e := MaskText("Bob", []Span{{EntityType: "PERSON", Start: 0, End: 3}}, nil, next)
			if e != nil || masked != "[PERSON_2]" {
				t.Fatalf("permanent reservation: %s %v", masked, e)
			}
		})
	}
}

func TestRegistryConcurrentAllocatorsAndScopes(t *testing.T) {
	cfg, id := registryFixture(t)
	var wg sync.WaitGroup
	for range 15 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a := allocator(t, cfg, id, "A")
			if p := a.Placeholder("PERSON", "Alice"); p != "[PERSON_0]" {
				t.Error(p)
			}
		}()
	}
	wg.Wait()
	b := allocator(t, cfg, id, "B")
	if p := b.Placeholder("PERSON", "Bob"); p != "[PERSON_0]" {
		t.Fatal(p)
	}
	if p := b.Placeholder("PERSON", "Alice"); p != "[PERSON_1]" {
		t.Fatal("scope leaked", p)
	}
}

func TestRegistryUnissuedPlaceholderAndInt64(t *testing.T) {
	cfg, id := registryFixture(t)
	a := allocator(t, cfg, id, "p")
	// A placeholder the registry never issued is text a client wrote: a span
	// over it is masked, and its index is reserved.
	raw := "[PERSON_0] and [PERSON_5]"
	masked, _, e := MaskText(raw, []Span{{EntityType: "PERSON", Start: 1, End: 7}}, nil, a)
	if e != nil || masked != "[[PERSON_1]_0] and [PERSON_5]" {
		t.Fatalf("span over an unissued placeholder: %s %v", masked, e)
	}
	// An issued one exempts it.
	issued := a.Placeholder("PERSON", "Alice")
	if masked, _, e = MaskText(issued, []Span{{EntityType: "PERSON", Start: 1, End: 7}}, nil, a); e != nil || masked != issued {
		t.Fatalf("span over an issued placeholder: %s %v", masked, e)
	}
	a.collect("[PERSON_1234567890123456789] [PERSON_99999999999999999999]")
	if e = a.settle(); e != nil {
		t.Fatal(e)
	}
	scope, _ := id.Scope("p", "")
	issuedIndexes, e := id.Store.Prepare(scope, "pii:PERSON", []int64{1234567890123456789})
	if e != nil || issuedIndexes[1234567890123456789] {
		t.Fatal(issuedIndexes, e)
	}
}

func TestRegistryLooperRedispatchSharedAllocator(t *testing.T) {
	cfg, id := registryFixture(t)
	a := allocator(t, cfg, id, "p")
	r := textRequest("Alice")
	if _, e := Apply(r, a, scanForValue("PERSON", "Alice")); e != nil {
		t.Fatal(e)
	}
	r.Messages = append(r.Messages, textRequest("Alice Bob [PERSON_2]").Messages...)
	scan := func(text string) ([]Span, error) {
		var spans []Span
		for _, v := range []string{"Alice", "Bob"} {
			if at := strings.Index(text, v); at >= 0 {
				spans = append(spans, Span{EntityType: "PERSON", Start: at, End: at + len(v)})
			}
		}
		return spans, nil
	}
	if _, e := Apply(r, a, scan); e != nil {
		t.Fatal(e)
	}
	if got := r.Messages[1].Content[0].Text; got != "[PERSON_0] [PERSON_1] [PERSON_2]" {
		t.Fatal(got)
	}
}
