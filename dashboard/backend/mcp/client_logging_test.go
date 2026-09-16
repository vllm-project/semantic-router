package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"reflect"
	"strings"
	"testing"
)

func TestCoerceArgumentTypes(t *testing.T) {
	args := map[string]interface{}{
		"emptyArray":    "",
		"singleArray":   "item",
		"nilArray":      nil,
		"emptyString":   []interface{}{},
		"singleString":  []interface{}{"item"},
		"multipleItems": []interface{}{"first", "second"},
		"floatString":   float64(1.5),
		"intString":     2,
		"boolString":    true,
		"number":        "1.25",
		"integer":       "2",
		"invalidNumber": "invalid",
		"trueValue":     "YES",
		"falseValue":    "",
		"unknownBool":   "perhaps",
		"nested": map[string]interface{}{
			"items":   "nested-item",
			"enabled": "1",
		},
		"unknown": "preserved",
	}
	schema := map[string]interface{}{
		"properties": map[string]interface{}{
			"emptyArray":    map[string]interface{}{"type": "array"},
			"singleArray":   map[string]interface{}{"type": "array"},
			"nilArray":      map[string]interface{}{"type": "array"},
			"emptyString":   map[string]interface{}{"type": "string"},
			"singleString":  map[string]interface{}{"type": "string"},
			"multipleItems": map[string]interface{}{"type": "string"},
			"floatString":   map[string]interface{}{"type": "string"},
			"intString":     map[string]interface{}{"type": "string"},
			"boolString":    map[string]interface{}{"type": "string"},
			"number":        map[string]interface{}{"type": "number"},
			"integer":       map[string]interface{}{"type": "integer"},
			"invalidNumber": map[string]interface{}{"type": "number"},
			"trueValue":     map[string]interface{}{"type": "boolean"},
			"falseValue":    map[string]interface{}{"type": "boolean"},
			"unknownBool":   map[string]interface{}{"type": "boolean"},
			"nested": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"items":   map[string]interface{}{"type": "array"},
					"enabled": map[string]interface{}{"type": "boolean"},
				},
			},
		},
	}
	expected := map[string]interface{}{
		"emptyArray":    []interface{}{},
		"singleArray":   []interface{}{"item"},
		"nilArray":      []interface{}{},
		"emptyString":   "",
		"singleString":  "item",
		"multipleItems": []interface{}{"first", "second"},
		"floatString":   "1.5",
		"intString":     "2",
		"boolString":    "true",
		"number":        float64(1.25),
		"integer":       float64(2),
		"invalidNumber": "invalid",
		"trueValue":     true,
		"falseValue":    false,
		"unknownBool":   "perhaps",
		"nested": map[string]interface{}{
			"items":   []interface{}{"nested-item"},
			"enabled": true,
		},
		"unknown": "preserved",
	}

	got := coerceArgumentTypes(args, schema)
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("coerceArgumentTypes() = %#v, want %#v", got, expected)
	}
	if got := coerceArgumentTypes(nil, schema); got != nil {
		t.Fatalf("coerceArgumentTypes(nil, schema) = %#v, want nil", got)
	}
}

func TestClientLogsArgumentShapeWithoutValues(t *testing.T) {
	const secretCanary = "mcp-argument-secret-canary" //nolint:gosec // Canary verifies MCP arguments are not logged.
	var output bytes.Buffer
	previousWriter := log.Writer()
	previousFlags := log.Flags()
	previousPrefix := log.Prefix()
	log.SetOutput(&output)
	log.SetFlags(0)
	log.SetPrefix("")
	t.Cleanup(func() {
		log.SetOutput(previousWriter)
		log.SetFlags(previousFlags)
		log.SetPrefix(previousPrefix)
	})

	client, err := NewClient(&ServerConfig{Name: "test-server"})
	if err != nil {
		t.Fatal(err)
	}
	_, _ = client.CallTool(
		context.Background(),
		"test-tool",
		json.RawMessage(`{"password":"`+secretCanary+`"}`),
	)
	coerceArgumentTypes(
		map[string]interface{}{"items": secretCanary},
		map[string]interface{}{
			"properties": map[string]interface{}{
				"items": map[string]interface{}{"type": "array"},
			},
		},
	)

	got := output.String()
	if strings.Contains(got, secretCanary) {
		t.Fatalf("MCP logs leaked argument values: %s", got)
	}
	if !strings.Contains(got, "argument_bytes=") {
		t.Fatalf("MCP logs omitted safe request shape: %s", got)
	}
}

func TestMCPCallFailureLogMessageOmitsEchoedErrorContent(t *testing.T) {
	const secretCanary = "mcp-error-echo-secret-canary" //nolint:gosec // Canary verifies remote errors are not logged.
	got := mcpCallFailureLogMessage(
		errors.New("remote tool echoed " + secretCanary),
	)

	if strings.Contains(got, secretCanary) {
		t.Fatalf("MCP failure log leaked remote error content: %s", got)
	}
	if !strings.Contains(got, "error_class=") {
		t.Fatalf("MCP failure log omitted safe error class: %s", got)
	}
}

func TestClientPublicStateAndStreamErrorsOmitRemoteDetails(t *testing.T) {
	const secretCanary = "mcp-public-error-secret-canary" //nolint:gosec // Canary verifies public state cannot expose remote errors.
	client, err := NewClient(&ServerConfig{Name: "test-server"})
	if err != nil {
		t.Fatal(err)
	}
	client.err = errors.New("remote server echoed " + secretCanary)

	state := client.GetState()
	if strings.Contains(state.Error, secretCanary) || state.Error != "Connection unavailable" {
		t.Fatalf("public state exposed remote error: %q", state.Error)
	}

	var chunk StreamChunk
	err = client.CallToolStreaming(
		context.Background(),
		"test-tool",
		nil,
		func(got StreamChunk) error {
			chunk = got
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(fmt.Sprint(chunk.Data), secretCanary) || chunk.Data != "Tool execution failed" {
		t.Fatalf("public stream exposed remote error: %#v", chunk)
	}
}

func TestManagerReturnsTransportFailuresInsteadOfPublicErrorResults(t *testing.T) {
	client, err := NewClient(&ServerConfig{Name: "test-server"})
	if err != nil {
		t.Fatal(err)
	}
	client.status = StatusConnected
	manager := &Manager{clients: map[string]*Client{"server-1": client}}

	result, err := manager.ExecuteTool(
		context.Background(),
		"server-1",
		"test-tool",
		nil,
	)
	if err == nil {
		t.Fatal("transport failure returned a public success result")
	}
	if result != nil {
		t.Fatalf("transport failure returned a public result: %#v", result)
	}
}
