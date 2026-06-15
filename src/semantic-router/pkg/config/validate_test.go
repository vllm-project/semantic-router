package config

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCollectKnownFields_Simple(t *testing.T) {
	type Simple struct {
		Name string `yaml:"name"`
		Port int    `yaml:"port,omitempty"`
	}
	known := collectKnownFields(reflect.TypeOf(Simple{}))
	assert.Contains(t, known, "name")
	assert.Contains(t, known, "port")
	assert.NotContains(t, known, "Name")
}

func TestCollectKnownFields_Inline(t *testing.T) {
	type Inner struct {
		Foo string `yaml:"foo"`
	}
	type Outer struct {
		Inner `yaml:",inline"`
		Bar   string `yaml:"bar"`
	}
	known := collectKnownFields(reflect.TypeOf(Outer{}))
	assert.Contains(t, known, "foo", "inline field should be promoted")
	assert.Contains(t, known, "bar")
}

func TestCollectKnownFields_SkipDash(t *testing.T) {
	type WithDash struct {
		Public  string `yaml:"public"`
		Private string `yaml:"-"`
	}
	known := collectKnownFields(reflect.TypeOf(WithDash{}))
	assert.Contains(t, known, "public")
	assert.NotContains(t, known, "Private")
	assert.NotContains(t, known, "-")
}

func TestClosestField(t *testing.T) {
	known := map[string]fieldEntry{
		"topk":         {},
		"backend_type": {},
	}
	tests := []struct {
		input string
		want  string
	}{
		{"top_k", "topk"},
		{"backend_tpye", "backend_type"},
		{"completely_wrong_field_name_xyz", ""},
	}
	for _, tt := range tests {
		got := closestField(tt.input, known)
		assert.Equal(t, tt.want, got, "closestField(%q)", tt.input)
	}
}

func TestWarnUnknownFields_DetectsTypos(t *testing.T) {
	type Search struct {
		TopK int `yaml:"topk"`
	}
	type Cache struct {
		Enabled     bool   `yaml:"enabled"`
		BackendType string `yaml:"backend_type"`
		Search      Search `yaml:"search"`
	}

	raw := map[string]interface{}{
		"enabled":      true,
		"backend_type": "redis",
		"search": map[interface{}]interface{}{
			"top_k": 1, // typo: should be "topk"
		},
	}

	warnings := collectUnknownFields(raw, reflect.TypeOf(Cache{}))
	assert.Len(t, warnings, 1)
	assert.Contains(t, warnings[0], `"top_k"`)
	assert.Contains(t, warnings[0], `"topk"`)
}

func TestWarnUnknownFields_NoSuggestionForDistantTypo(t *testing.T) {
	type Config struct {
		Name string `yaml:"name"`
	}

	raw := map[string]interface{}{
		"xyzzy_blorp_foobar": "wat",
	}

	warnings := collectUnknownFields(raw, reflect.TypeOf(Config{}))
	assert.Len(t, warnings, 1)
	assert.Contains(t, warnings[0], `"xyzzy_blorp_foobar"`)
	assert.NotContains(t, warnings[0], "did you mean")
}

func TestWarnUnknownFields_ValidConfig(t *testing.T) {
	type Inner struct {
		Port int `yaml:"port"`
	}
	type Config struct {
		Name  string `yaml:"name"`
		Inner Inner  `yaml:"inner"`
	}

	raw := map[string]interface{}{
		"name": "test",
		"inner": map[interface{}]interface{}{
			"port": 8080,
		},
	}

	warnings := collectUnknownFields(raw, reflect.TypeOf(Config{}))
	assert.Empty(t, warnings, "valid config should produce no warnings")
}

func TestWarnUnknownFields_SliceOfStructs(t *testing.T) {
	type Item struct {
		Name string `yaml:"name"`
	}
	type Config struct {
		Items []Item `yaml:"items"`
	}

	raw := map[string]interface{}{
		"items": []interface{}{
			map[interface{}]interface{}{
				"name": "valid",
				"naem": "typo", // typo
			},
		},
	}

	warnings := collectUnknownFields(raw, reflect.TypeOf(Config{}))
	assert.Len(t, warnings, 1)
	assert.Contains(t, warnings[0], `"naem"`)
	assert.Contains(t, warnings[0], `"name"`)
}

func TestWarnUnknownFields_MapValues(t *testing.T) {
	type Params struct {
		Weight int `yaml:"weight"`
	}
	type Config struct {
		Models map[string]Params `yaml:"models"`
	}

	raw := map[string]interface{}{
		"models": map[interface{}]interface{}{
			"gpt-4": map[interface{}]interface{}{
				"weight": 1,
				"wieght": 2, // typo
			},
		},
	}

	warnings := collectUnknownFields(raw, reflect.TypeOf(Config{}))
	assert.Len(t, warnings, 1)
	assert.Contains(t, warnings[0], `"wieght"`)
	assert.Contains(t, warnings[0], `"weight"`)
}

func TestWarnUnknownFields_CanonicalConfig(t *testing.T) {
	raw := map[string]interface{}{
		"version": "0.3",
		"routing": map[interface{}]interface{}{},
		"global":  map[interface{}]interface{}{},
	}
	warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{}))
	assert.Empty(t, warnings, "minimal valid canonical config should produce no warnings")
}

func TestWarnUnknownFields_CanonicalConfigTypo(t *testing.T) {
	raw := map[string]interface{}{
		"version":  "0.3",
		"routingg": map[interface{}]interface{}{}, // typo
	}
	warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{}))
	assert.Len(t, warnings, 1)
	assert.Contains(t, warnings[0], `"routingg"`)
	assert.Contains(t, warnings[0], `"routing"`)
}

func TestWarnUnknownFields_ReferenceConfig(t *testing.T) {
	data := readReferenceConfigYAML(t)
	raw, err := parseRawConfigMap(data)
	if err != nil {
		t.Fatalf("failed to parse reference config: %v", err)
	}
	warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{}))
	assert.Empty(t, warnings, "reference config must produce zero warnings: %v", warnings)
}
