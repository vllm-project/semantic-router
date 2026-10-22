package extension

import (
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestRegistryKeepsTypesAliasesAndOrder(t *testing.T) {
	r := NewRegistry[int]("test extension")
	r.MustRegister("zeta", 1)
	r.MustRegister("alpha", 2, "a", "first")
	if got := r.Types(); !reflect.DeepEqual(got, []string{"alpha", "zeta"}) {
		t.Fatalf("Types() = %v, want sorted", got)
	}
	if got := r.Entries(); len(got) != 2 || got[0].Type != "zeta" || got[1].Type != "alpha" {
		t.Fatalf("Entries() = %v, want registration order", got)
	}
	if spec, ok := r.Lookup("first"); !ok || spec != 2 {
		t.Fatalf("Lookup(alias) = %d, %v", spec, ok)
	}
	if r.Normalize("a") != "alpha" || r.Normalize("other") != "other" {
		t.Fatal("Normalize resolves aliases only")
	}
	if _, ok := r.Lookup("missing"); ok {
		t.Fatal("an unregistered type resolved")
	}
}

func TestRegistryRefusesConflictingNames(t *testing.T) {
	r := NewRegistry[string]("test extension")
	r.MustRegister("alpha", "x", "a")
	for _, attempt := range []struct {
		typ     string
		aliases []string
	}{
		{"alpha", nil},
		{"a", nil},
		{"beta", []string{"alpha"}},
		{"gamma", []string{"a"}},
		{" ", nil},
	} {
		if err := r.Register(attempt.typ, "y", attempt.aliases...); err == nil {
			t.Fatalf("Register(%q, %v) succeeded", attempt.typ, attempt.aliases)
		} else if !strings.Contains(err.Error(), "test extension") {
			t.Fatalf("the error does not name the kind: %v", err)
		}
	}
	if len(r.Entries()) != 1 {
		t.Fatal("a refused registration left an entry")
	}
}

func TestRegistryIsSafeForConcurrentUse(t *testing.T) {
	r := NewRegistry[int]("test extension")
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = r.Register(strings.Repeat("t", i+1), i)
			r.Types()
			r.Lookup("t")
		}()
	}
	wg.Wait()
	if len(r.Types()) != 16 {
		t.Fatalf("Types() = %v", r.Types())
	}
}
