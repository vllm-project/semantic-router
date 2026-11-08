package openapischema

import (
	"encoding/json"
	"reflect"
	"slices"
	"testing"
	"time"
)

type embeddedFields struct {
	Shared string `json:"shared"`
}

type sampleBody struct {
	embeddedFields
	Name     string          `json:"name"`
	Note     string          `json:"note,omitempty"`
	Count    *int            `json:"count"`
	Tags     []string        `json:"tags"`
	Raw      json.RawMessage `json:"raw,omitempty"`
	Labels   map[string]int  `json:"labels"`
	Extra    map[string]any  `json:"extra"`
	At       time.Time       `json:"at"`
	Any      any             `json:"any"`
	Ignored  string          `json:"-"`
	internal string          //nolint:unused // exercises the unexported-field skip
}

func TestForFollowsJSONTags(t *testing.T) {
	schema := For[sampleBody](nil)
	if schema.Type != "object" {
		t.Fatalf("type = %q, want object", schema.Type)
	}
	for _, name := range []string{"Ignored", "-", "internal"} {
		if _, ok := schema.Properties[name]; ok {
			t.Errorf("unexpected property %q", name)
		}
	}
	if want := []string{"shared", "name", "count", "tags", "labels", "extra", "at", "any"}; !slices.Equal(schema.Required, want) {
		t.Errorf("required = %v, want %v", schema.Required, want)
	}
	for name, want := range map[string]Schema{
		"shared": {Type: "string"},
		"count":  {Type: "integer", Nullable: true},
		"raw":    {},
		"at":     {Type: "string", Format: "date-time"},
		"any":    {},
	} {
		if got := schema.Properties[name]; !reflect.DeepEqual(got, want) {
			t.Errorf("%s = %+v, want %+v", name, got, want)
		}
	}
	if items := schema.Properties["tags"].Items; items == nil || items.Type != "string" || !schema.Properties["tags"].Nullable {
		t.Errorf("tags = %+v, want nullable array of string", schema.Properties["tags"])
	}
	if additional, ok := schema.Properties["labels"].AdditionalProperties.(*Schema); !ok || additional.Type != "integer" {
		t.Errorf("labels additionalProperties = %#v, want integer schema", schema.Properties["labels"].AdditionalProperties)
	}
	if schema.Properties["extra"].AdditionalProperties != true {
		t.Errorf("extra additionalProperties = %#v, want true", schema.Properties["extra"].AdditionalProperties)
	}
}

type recursive struct {
	Children []recursive `json:"children"`
}

func TestForStopsAtRecursiveTypes(t *testing.T) {
	schema := For[recursive](nil)
	if items := schema.Properties["children"].Items; items == nil || items.Type != "object" || items.Properties != nil {
		t.Fatalf("children items = %+v, want a bare object", items)
	}
}

type tokenLimit struct{}

type withOverride struct {
	Limit    tokenLimit  `json:"limit"`
	Optional *tokenLimit `json:"optional,omitempty"`
}

func TestOverridesReplaceDerivedSchemas(t *testing.T) {
	union := Schema{OneOf: []Schema{{Type: "integer"}, {Type: "string", Enum: []string{"auto"}}}}
	schema := For[withOverride](Overrides{reflect.TypeFor[tokenLimit](): union})
	if got := schema.Properties["limit"]; !reflect.DeepEqual(got, union) {
		t.Errorf("limit = %+v, want the override", got)
	}
	optional := schema.Properties["optional"]
	if !optional.Nullable || len(optional.OneOf) != 2 {
		t.Errorf("optional = %+v, want the nullable override", optional)
	}
}

type wireScore struct {
	Score          float64 `json:"score"`
	ScoreAvailable bool    `json:"-"`
}

type wireScoreJSON struct {
	Score *float64 `json:"score"`
}

func (s wireScore) JSONWire() any {
	if !s.ScoreAvailable {
		return wireScoreJSON{}
	}
	return wireScoreJSON{Score: &s.Score}
}

func TestJSONWireRepresentationDescribesTheEncodedShape(t *testing.T) {
	score := For[wireScore](nil).Properties["score"]
	if score.Type != "number" || !score.Nullable {
		t.Fatalf("score = %+v, want nullable number from the wire type", score)
	}
}

func TestAccepts(t *testing.T) {
	open := Schema{Required: []string{"error", "message"}, Properties: map[string]Schema{
		"error": {Type: "string"}, "message": {Type: "string"},
	}}
	closed := open
	closed.AdditionalProperties = false

	for _, test := range []struct {
		name   string
		schema Schema
		keys   []string
		want   bool
	}{
		{"open schema ignores extra keys", open, []string{"error", "message", "stage"}, true},
		{"missing a required key", open, []string{"error"}, false},
		{"closed schema rejects an extra key", closed, []string{"error", "message", "stage"}, false},
		{"closed schema accepts its exact keys", closed, []string{"error", "message"}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := test.schema.Accepts(test.keys); got != test.want {
				t.Errorf("Accepts(%v) = %v, want %v", test.keys, got, test.want)
			}
		})
	}
}
