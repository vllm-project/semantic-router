package agentmanagement

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestToolSchemaSupportsLocalReferencesAndRejectsRemoteReferences(t *testing.T) {
	local := json.RawMessage(`{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "$defs":{"identifier":{"type":"string","pattern":"^[a-z][a-z0-9-]+$"}},
  "type":"object",
  "properties":{"id":{"$ref":"#/$defs/identifier"}},
  "required":["id"],
  "additionalProperties":false
}`)
	compiled, _, err := compileToolSchema(local)
	if err != nil {
		t.Fatalf("compileToolSchema(local) error = %v", err)
	}
	if err := compiled.validateRaw(json.RawMessage(`{"id":"recipe-one"}`), 1024); err != nil {
		t.Fatalf("validateRaw() error = %v", err)
	}
	if err := compiled.validateRaw(json.RawMessage(`{"id":"INVALID"}`), 1024); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid pattern error = %v, want ErrInvalid", err)
	}

	remote := json.RawMessage(`{"$ref":"https://schemas.example.invalid/tool.json"}`)
	if _, _, err := compileToolSchema(remote); !errors.Is(err, ErrInvalid) {
		t.Fatalf("remote ref error = %v, want ErrInvalid", err)
	}
}

func TestToolSchemaRejectsDuplicateKeysAndStructuralBombs(t *testing.T) {
	duplicate := json.RawMessage(`{"type":"object","type":"array"}`)
	if _, _, err := compileToolSchema(duplicate); !errors.Is(err, ErrInvalid) {
		t.Fatalf("duplicate key error = %v, want ErrInvalid", err)
	}

	bomb := strings.Repeat(`{"allOf":[`, maximumSchemaDepth+2) +
		`{"type":"object"}` + strings.Repeat(`]}`, maximumSchemaDepth+2)
	if _, _, err := compileToolSchema(json.RawMessage(bomb)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("schema bomb error = %v, want ErrInvalid", err)
	}
}

func TestToolSchemaRejectsCyclicLocalReferences(t *testing.T) {
	cyclic := json.RawMessage(`{
  "$defs":{"loop":{"$ref":"#/$defs/loop"}},
  "$ref":"#/$defs/loop"
}`)
	if _, _, err := compileToolSchema(cyclic); !errors.Is(err, ErrInvalid) {
		t.Fatalf("cyclic ref error = %v, want ErrInvalid", err)
	}
}
