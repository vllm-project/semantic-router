package v1alpha1

import (
	"encoding/json"
	"strings"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
)

// arrayItem stands for "the first element" in a defaultedScalar path.
const arrayItem = "[]"

// defaultedScalar is a boolean or integer spec property whose CRD default is
// not its zero value, in a schema that still admits that zero.
type defaultedScalar struct {
	path []string
	zero interface{}
	def  interface{}
}

func (f defaultedScalar) String() string {
	return strings.ReplaceAll(strings.Join(f.path, "."), "."+arrayItem, arrayItem)
}

// A controller that reads a SemanticRouter into these Go types and writes it
// back sends whatever encoding/json keeps. A plain bool or int with omitempty
// loses an explicit false or 0, and the API server then applies the CRD default
// over the value the user wrote. Every such property must therefore be a
// pointer, so that only an omitted field takes its default.
func TestExplicitZeroSurvivesTypedRoundTrip(t *testing.T) {
	for _, path := range generatedCRDCopies {
		t.Run(generatedCRDName(path), func(t *testing.T) {
			structural := loadCRDStructural(t, path)
			spec, ok := structural.Properties["spec"]
			if !ok {
				t.Fatal("CRD schema has no spec")
			}
			var fields []defaultedScalar
			collectDefaultedScalars(&spec, nil, &fields)

			seen := map[string]bool{}
			for _, field := range fields {
				seen[field.String()] = true
			}
			for _, want := range []string{
				"config.response_cache.enabled",
				"config.response_cache.ttl_seconds",
				"config.observability.tracing.exporter.insecure",
				"vllmEndpoints[].weight",
			} {
				if !seen[want] {
					t.Errorf("schema walk did not reach %s; found %d properties", want, len(fields))
				}
			}

			for _, field := range fields {
				t.Run(field.String(), func(t *testing.T) {
					for _, tc := range []struct {
						name     string
						provided bool
						want     interface{}
					}{
						{name: "explicit_zero", provided: true, want: field.zero},
						{name: "omitted", want: field.def},
					} {
						object := throughGoTypes(t, nestedValue(field.path, field.zero, tc.provided))
						defaulting.Default(object, structural)
						got, found := lookupValue(object["spec"], field.path)
						if !found {
							t.Fatalf("%s: property is absent after defaulting", tc.name)
						}
						if encodeJSON(t, got) != encodeJSON(t, tc.want) {
							t.Errorf("%s: got %v after a typed round trip and defaulting, want %v", tc.name, got, tc.want)
						}
					}
				})
			}
		})
	}
}

func collectDefaultedScalars(s *schema.Structural, path []string, out *[]defaultedScalar) {
	if def := s.Default.Object; def != nil {
		switch s.Type {
		case "boolean":
			if def == true {
				*out = append(*out, defaultedScalar{path: path, zero: false, def: def})
			}
		case "integer":
			if !isZeroNumber(def) && admitsZero(s.ValueValidation) {
				*out = append(*out, defaultedScalar{path: path, zero: 0, def: def})
			}
		}
	}
	for name, property := range s.Properties {
		collectDefaultedScalars(&property, appendPath(path, name), out)
	}
	if s.Items != nil {
		collectDefaultedScalars(s.Items, appendPath(path, arrayItem), out)
	}
}

func appendPath(path []string, segment string) []string {
	return append(append(make([]string, 0, len(path)+1), path...), segment)
}

func isZeroNumber(value interface{}) bool {
	switch number := value.(type) {
	case int64:
		return number == 0
	case float64:
		return number == 0
	default:
		return false
	}
}

func admitsZero(validation *schema.ValueValidation) bool {
	if validation == nil {
		return true
	}
	if minimum := validation.Minimum; minimum != nil && (*minimum > 0 || (*minimum == 0 && validation.ExclusiveMinimum)) {
		return false
	}
	if maximum := validation.Maximum; maximum != nil && (*maximum < 0 || (*maximum == 0 && validation.ExclusiveMaximum)) {
		return false
	}
	return true
}

// nestedValue builds the spec that holds path, with the last property set to
// leaf when provided and left out otherwise.
func nestedValue(path []string, leaf interface{}, provided bool) interface{} {
	if len(path) == 0 {
		return leaf
	}
	if path[0] == arrayItem {
		return []interface{}{nestedValue(path[1:], leaf, provided)}
	}
	object := map[string]interface{}{}
	if len(path) > 1 || provided {
		object[path[0]] = nestedValue(path[1:], leaf, provided)
	}
	return object
}

func lookupValue(value interface{}, path []string) (interface{}, bool) {
	for _, segment := range path {
		if segment == arrayItem {
			list, ok := value.([]interface{})
			if !ok || len(list) == 0 {
				return nil, false
			}
			value = list[0]
			continue
		}
		object, ok := value.(map[string]interface{})
		if !ok {
			return nil, false
		}
		if value, ok = object[segment]; !ok {
			return nil, false
		}
	}
	return value, true
}

// throughGoTypes returns the object a client sends after decoding spec into
// the API types and encoding it again, which is what a full Update does.
func throughGoTypes(t *testing.T, spec interface{}) map[string]interface{} {
	t.Helper()
	var typed SemanticRouter
	if err := json.Unmarshal([]byte(encodeJSON(t, map[string]interface{}{"spec": spec})), &typed); err != nil {
		t.Fatalf("decode into API types: %v", err)
	}
	var object map[string]interface{}
	if err := json.Unmarshal([]byte(encodeJSON(t, &typed)), &object); err != nil {
		t.Fatalf("decode typed object: %v", err)
	}
	return object
}

func encodeJSON(t *testing.T, value interface{}) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode JSON: %v", err)
	}
	return string(data)
}
