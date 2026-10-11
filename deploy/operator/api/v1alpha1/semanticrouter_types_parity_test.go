package v1alpha1

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The complexity rule is the one signal rule the CRD types by hand instead of
// passing through as JSON, and the controller converts it generically: a CRD
// key the Router does not know is dropped without a word, and a Router key the
// CRD lacks cannot be written at all. Nothing else keeps the two field lists
// together, which is how three of four representations drifted in #3542. This
// test compares the CRD's json keys with the Router's yaml keys, level by
// level, and names every key present on one side only unless it is listed
// below with a reason.
func TestComplexityRulesConfigMirrorsRouterComplexityRule(t *testing.T) {
	// Differences that are deliberate. A path is the dotted key path from the
	// rule; the value is the reason a reader can disagree with. An entry that
	// stops matching a real difference fails the test, so the list cannot
	// outlive the drift it documents.
	known := map[string]string{
		// RuleNode is both leaf and branch, so it carries leaf fields at the
		// composer root. The CRD composer is always a branch.
		"composer.type":       "CRD composer is branch-only; RuleNode doubles as a leaf",
		"composer.name":       "CRD composer is branch-only; RuleNode doubles as a leaf",
		"composer.label":      "CRD composer is branch-only; RuleNode doubles as a leaf",
		"composer.predicate":  "CRD composer is branch-only; RuleNode doubles as a leaf",
		"composer.on_error":   "CRD composer is branch-only; RuleNode doubles as a leaf",
		"composer.on_unknown": "CRD composer is branch-only; RuleNode doubles as a leaf",
		// CRD composer conditions are flat named-signal leaves by design;
		// nesting, predicates and failure policy stay with decisions.
		"composer.conditions.label":      "CRD composer conditions are flat named-signal leaves",
		"composer.conditions.predicate":  "CRD composer conditions are flat named-signal leaves",
		"composer.conditions.on_error":   "CRD composer conditions are flat named-signal leaves",
		"composer.conditions.on_unknown": "CRD composer conditions are flat named-signal leaves",
		"composer.conditions.operator":   "CRD composer conditions are flat named-signal leaves",
		"composer.conditions.conditions": "CRD composer conditions are flat named-signal leaves",
	}

	used := map[string]bool{}
	assertMirroredKeys(t, "", reflect.TypeOf(ComplexityRulesConfig{}), reflect.TypeOf(routerconfig.ComplexityRule{}), known, used)

	for path := range known {
		if !used[path] {
			t.Errorf("known difference %q no longer exists; remove it from the list", path)
		}
	}
}

// assertMirroredKeys compares the serialised keys of two struct types and
// recurses into the fields both sides share. Types are not compared: the CRD
// stores fractional numbers as strings and wraps optional blocks in pointers,
// and the controller's conversion already handles both.
func assertMirroredKeys(t *testing.T, path string, crd, router reflect.Type, known map[string]string, used map[string]bool) {
	t.Helper()
	crd, router = structType(crd), structType(router)
	if crd == nil || router == nil {
		return
	}
	crdKeys := taggedFields(crd, "json")
	routerKeys := taggedFields(router, "yaml")

	report := func(key, message string) {
		full := strings.TrimPrefix(path+"."+key, ".")
		if _, ok := known[full]; ok {
			used[full] = true
			return
		}
		t.Errorf("%s: %s", full, message)
	}

	for _, key := range sortedKeys(crdKeys) {
		routerField, shared := routerKeys[key]
		if !shared {
			report(key, "declared on the CRD but the Router has no such field, so the controller drops it silently")
			continue
		}
		assertMirroredKeys(t, strings.TrimPrefix(path+"."+key, "."), crdKeys[key], routerField, known, used)
	}
	for _, key := range sortedKeys(routerKeys) {
		if _, shared := crdKeys[key]; !shared {
			report(key, "accepted by the Router but missing from the CRD, so an operator user cannot write it")
		}
	}
}

// structType unwraps pointers, slices and maps and returns the struct type
// underneath, or nil when the field is a scalar.
func structType(typ reflect.Type) reflect.Type {
	for typ.Kind() == reflect.Pointer || typ.Kind() == reflect.Slice || typ.Kind() == reflect.Map {
		typ = typ.Elem()
	}
	if typ.Kind() != reflect.Struct {
		return nil
	}
	return typ
}

// taggedFields maps each serialised key of a struct to its field type, using
// the first segment of the named tag and skipping untagged and "-" fields.
func taggedFields(typ reflect.Type, tag string) map[string]reflect.Type {
	fields := map[string]reflect.Type{}
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		key := strings.Split(field.Tag.Get(tag), ",")[0]
		if key == "" || key == "-" {
			continue
		}
		fields[key] = field.Type
	}
	return fields
}

func sortedKeys(fields map[string]reflect.Type) []string {
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
