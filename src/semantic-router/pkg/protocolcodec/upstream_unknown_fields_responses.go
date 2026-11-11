package protocolcodec

import (
	"encoding/json"
	"reflect"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Responses output items, their content parts and reasoning summaries are
// raw values decoded on their own, so the resource-level strip cannot see
// their members. These helpers walk them only to report what that decode
// drops; the decode itself validates them and reports its own errors.

func responsesOutputUnknownFields(raw json.RawMessage, policy llmprotocol.Policy) []string {
	if !dropUnknownUpstreamFields(policy) || !jsonFieldPresent(raw) {
		return nil
	}
	var bodies []json.RawMessage
	if err := json.Unmarshal(raw, &bodies); err != nil {
		return nil
	}
	dropped := make(map[string]struct{})
	for _, body := range bodies {
		addUnknownFieldPaths(dropped, "output[]", responsesItemUnknownFields(body, policy))
	}
	return sortedUnknownFieldPaths(dropped)
}

func responsesItemUnknownFields(body json.RawMessage, policy llmprotocol.Policy) []string {
	if !dropUnknownUpstreamFields(policy) {
		return nil
	}
	var item struct {
		Type    string          `json:"type"`
		Content json.RawMessage `json:"content"`
		Summary json.RawMessage `json:"summary"`
	}
	if err := json.Unmarshal(body, &item); err != nil {
		return nil
	}
	_, dropped := stripUnknownUpstreamFields(body, reflect.TypeOf(responsesItemWire{}))
	fields := make(map[string]struct{}, len(dropped))
	addUnknownFieldPaths(fields, "", dropped)
	switch item.Type {
	case "message":
		collectResponsesValueUnknownFields(fields, "content", item.Content, reflect.TypeOf([]responsesContentWire{}))
	case "reasoning":
		collectResponsesValueUnknownFields(fields, "summary", item.Summary, reflect.TypeOf([]responsesSummaryWire{}))
		collectResponsesValueUnknownFields(fields, "content", item.Content, reflect.TypeOf([]responsesContentWire{}))
	}
	return sortedUnknownFieldPaths(fields)
}

// responsesSummaryWire is the shape of one reasoning summary part.
type responsesSummaryWire struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func collectResponsesValueUnknownFields(destination map[string]struct{}, prefix string, raw json.RawMessage, target reflect.Type) {
	if !jsonFieldPresent(raw) {
		return
	}
	_, dropped := stripUnknownUpstreamFields(raw, target)
	addUnknownFieldPaths(destination, prefix, dropped)
}

func addUnknownFieldPaths(destination map[string]struct{}, prefix string, fields []string) {
	for _, field := range fields {
		path := field
		if prefix != "" {
			path = prefix + "." + field
			if strings.HasPrefix(field, "[]") {
				path = prefix + field
			}
		}
		destination[path] = struct{}{}
	}
}

func sortedUnknownFieldPaths(fields map[string]struct{}) []string {
	if len(fields) == 0 {
		return nil
	}
	paths := make([]string, 0, len(fields))
	for path := range fields {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths
}

// prefixedUnknownFieldPaths places the paths a nested decode reported under
// the member that holds the nested value.
func prefixedUnknownFieldPaths(prefix string, fields []string) []string {
	prefixed := make(map[string]struct{}, len(fields))
	addUnknownFieldPaths(prefixed, prefix, fields)
	return sortedUnknownFieldPaths(prefixed)
}
