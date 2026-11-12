// Package extension keeps typed registries of the Router's extensions:
// signals, algorithms, plugins, graph nodes and gateway capabilities each
// register by a type name, the way Envoy extensions are identified by their
// type. Every kind of extension defines its own spec, typically a Go factory,
// a config payload whose shape is the schema, defaults and a validator, and
// its own Registry of that spec. The built-in types register through the same
// call as any other package, so adding a type needs no change to the code that
// consumes the registry.
package extension

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Entry is one registered type.
type Entry[S any] struct {
	Type string
	Spec S
}

// Registry holds the specs of one kind of extension by type name. It is safe
// for concurrent use; types usually register from init functions.
type Registry[S any] struct {
	kind string

	mu      sync.RWMutex
	order   []string
	specs   map[string]S
	aliases map[string]string
}

// NewRegistry returns an empty registry; kind names the extension kind in
// errors, such as "decision plugin".
func NewRegistry[S any](kind string) *Registry[S] {
	return &Registry[S]{kind: kind, specs: map[string]S{}, aliases: map[string]string{}}
}

// Kind names the extension kind.
func (r *Registry[S]) Kind() string { return r.kind }

// Register adds spec under typ. Aliases are other spellings that resolve to
// typ. A name registers once, as a type or as an alias.
func (r *Registry[S]) Register(typ string, spec S, aliases ...string) error {
	if strings.TrimSpace(typ) == "" {
		return fmt.Errorf("%s: a type name is required", r.kind)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, name := range append([]string{typ}, aliases...) {
		if r.takenLocked(name) {
			return fmt.Errorf("%s %q is already registered", r.kind, name)
		}
	}
	r.order = append(r.order, typ)
	r.specs[typ] = spec
	for _, alias := range aliases {
		r.aliases[alias] = typ
	}
	return nil
}

// MustRegister is Register for init functions: a conflicting registration is
// a programming error.
func (r *Registry[S]) MustRegister(typ string, spec S, aliases ...string) {
	if err := r.Register(typ, spec, aliases...); err != nil {
		panic(err)
	}
}

func (r *Registry[S]) takenLocked(name string) bool {
	_, isType := r.specs[name]
	_, isAlias := r.aliases[name]
	return isType || isAlias
}

// Normalize resolves an alias to its type; any other name is returned as is.
func (r *Registry[S]) Normalize(name string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if typ, ok := r.aliases[name]; ok {
		return typ
	}
	return name
}

// Lookup returns the spec of a type or alias.
func (r *Registry[S]) Lookup(name string) (S, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if typ, ok := r.aliases[name]; ok {
		name = typ
	}
	spec, ok := r.specs[name]
	return spec, ok
}

// Types returns the registered type names, sorted.
func (r *Registry[S]) Types() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	types := append([]string(nil), r.order...)
	sort.Strings(types)
	return types
}

// Entries returns the registered types in registration order, which is the
// order catalogs present them in.
func (r *Registry[S]) Entries() []Entry[S] {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entries := make([]Entry[S], 0, len(r.order))
	for _, typ := range r.order {
		entries = append(entries, Entry[S]{Type: typ, Spec: r.specs[typ]})
	}
	return entries
}
