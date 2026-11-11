package systemonecodec

import (
	"fmt"
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
)

var (
	schemaIDPattern    = regexp.MustCompile(`^[a-z][a-z0-9-]{0,63}/[0-9]+\.[0-9]+$`)
	deviationIDPattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
)

// VendoredSpecs are the specifications in config/schemas/systemone and the
// model runtime contract, with the sha256 their files had when the tables
// were generated.
func VendoredSpecs() []SpecRef { return slices.Clone(generatedSpecs) }

// VendoredSpec is the vendored specification of a schema.
func VendoredSpec(id SchemaID) (SpecRef, bool) {
	for _, spec := range generatedSpecs {
		if spec.ID == id {
			return spec, true
		}
	}
	return SpecRef{}, false
}

// Registry resolves schema and deviation IDs to their stateless codecs. It is
// immutable after construction.
type Registry struct {
	codecs     map[SchemaID]SchemaCodec
	deviations map[string]Deviation
}

// NewRegistry validates codecs and deviations against the vendored
// specifications: every ID well formed and registered once, every codec
// stateless and following the vendored file of its schema byte for byte (by
// sha256), and every deviation applying to a registered schema.
func NewRegistry(codecs []SchemaCodec, deviations []Deviation) (*Registry, error) {
	return newRegistry(generatedSpecs, codecs, deviations)
}

func newRegistry(specs []SpecRef, codecs []SchemaCodec, deviations []Deviation) (*Registry, error) {
	registry := &Registry{codecs: map[SchemaID]SchemaCodec{}, deviations: map[string]Deviation{}}
	for index, codec := range codecs {
		if isNil(codec) {
			return nil, fmt.Errorf("schema codec %d is nil", index)
		}
		id := codec.ID()
		if !schemaIDPattern.MatchString(string(id)) {
			return nil, fmt.Errorf("schema codec %d has the invalid ID %q; an ID is <family>/<major>.<minor>, such as systemone/0.2", index, id)
		}
		if !codec.Stateless() {
			return nil, fmt.Errorf("schema codec %q is not stateless; one codec value serves every request", id)
		}
		if _, duplicate := registry.codecs[id]; duplicate {
			return nil, fmt.Errorf("schema codec %q is registered more than once", id)
		}
		vendored, found := specFor(specs, id)
		if !found {
			return nil, fmt.Errorf("schema codec %q has no vendored specification; add it under config/schemas/systemone and run make systemone-schemas-generate", id)
		}
		if spec := codec.Spec(); spec.ID != id || spec.Path != vendored.Path || spec.SHA256 != vendored.SHA256 {
			return nil, fmt.Errorf("schema codec %q follows %s (sha256 %s), not the vendored %s (sha256 %s)", id, spec.Path, spec.SHA256, vendored.Path, vendored.SHA256)
		}
		registry.codecs[id] = codec
	}
	if len(registry.codecs) == 0 {
		return nil, fmt.Errorf("the schema codec registry is empty")
	}
	for index, deviation := range deviations {
		if isNil(deviation) {
			return nil, fmt.Errorf("deviation %d is nil", index)
		}
		id := deviation.ID()
		if !deviationIDPattern.MatchString(id) {
			return nil, fmt.Errorf("deviation %d has the invalid ID %q; an ID is lower_snake_case, such as state_text", index, id)
		}
		if _, duplicate := registry.deviations[id]; duplicate {
			return nil, fmt.Errorf("deviation %q is registered more than once", id)
		}
		applies := false
		for schema := range registry.codecs {
			applies = applies || deviation.AppliesTo(schema)
		}
		if !applies {
			return nil, fmt.Errorf("deviation %q applies to no registered schema (%s)", id, strings.Join(registry.schemaNames(), ", "))
		}
		registry.deviations[id] = deviation
	}
	return registry, nil
}

func specFor(specs []SpecRef, id SchemaID) (SpecRef, bool) {
	for _, spec := range specs {
		if spec.ID == id {
			return spec, true
		}
	}
	return SpecRef{}, false
}

// Resolve is the codec of a schema and the deviations a deployment declares
// for it, in order. An empty schema is DefaultSchema. An unknown ID, a
// repeated deviation or one that does not apply to the schema is an error
// that lists what is supported.
func (r *Registry) Resolve(schema SchemaID, deviations []string) (SchemaCodec, []Deviation, error) {
	if r == nil {
		return nil, nil, fmt.Errorf("no schema codec registry")
	}
	if schema == "" {
		schema = DefaultSchema
	}
	codec, ok := r.codecs[schema]
	if !ok {
		return nil, nil, fmt.Errorf("schema %q is not supported; the schemas are %s", schema, strings.Join(r.schemaNames(), ", "))
	}
	resolved := make([]Deviation, 0, len(deviations))
	for index, id := range deviations {
		deviation, ok := r.deviations[id]
		if !ok {
			return nil, nil, fmt.Errorf("deviation %q is not supported; the deviations are %s", id, strings.Join(r.deviationNames(), ", "))
		}
		if slices.Contains(deviations[:index], id) {
			return nil, nil, fmt.Errorf("deviations lists %q twice", id)
		}
		if !deviation.AppliesTo(schema) {
			return nil, nil, fmt.Errorf("deviation %q does not apply to schema %s", id, schema)
		}
		resolved = append(resolved, deviation)
	}
	return codec, resolved, nil
}

// Schemas are the registered schema IDs, sorted.
func (r *Registry) Schemas() []SchemaID {
	ids := make([]SchemaID, 0, len(r.codecs))
	for id := range r.codecs {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

// Deviations are the registered deviation IDs, sorted.
func (r *Registry) Deviations() []string { return r.deviationNames() }

func (r *Registry) schemaNames() []string {
	names := make([]string, 0, len(r.codecs))
	for id := range r.codecs {
		names = append(names, string(id))
	}
	sort.Strings(names)
	return names
}

func (r *Registry) deviationNames() []string {
	names := make([]string, 0, len(r.deviations))
	for id := range r.deviations {
		names = append(names, id)
	}
	sort.Strings(names)
	return names
}

var builtin = sync.OnceValue(func() *Registry {
	registry, err := NewRegistry(
		[]SchemaCodec{SystemOne02Codec{}, VLLMSR50Codec{}, VLLMSR40Codec{}},
		[]Deviation{StateTextDeviation{}},
	)
	if err != nil {
		panic(err)
	}
	return registry
})

// Builtin is the registry of the schemas and deviations this build speaks:
// systemone/0.2, vllm-sr/5.0, the legacy vllm-sr/4.0 and state_text.
func Builtin() *Registry { return builtin() }

func isNil(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}
