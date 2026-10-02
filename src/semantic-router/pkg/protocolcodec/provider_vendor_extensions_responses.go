package protocolcodec

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func responsesOutputVendorExtensions(raw json.RawMessage, policy llmprotocol.Policy) ([]string, error) {
	if !providerVendorExtensionsAllowed(policy) || len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var bodies []json.RawMessage
	if err := decodeProviderValue(raw, &bodies, policy); err != nil {
		return nil, err
	}
	dropped := make(map[string]struct{})
	for _, body := range bodies {
		fields, err := responsesItemVendorExtensions(body, policy)
		addVendorExtensionPaths(dropped, "output[]", fields)
		if err != nil {
			return sortedVendorExtensionPaths(dropped), err
		}
	}
	return sortedVendorExtensionPaths(dropped), nil
}

func responsesItemVendorExtensions(body json.RawMessage, policy llmprotocol.Policy) ([]string, error) {
	if !providerVendorExtensionsAllowed(policy) {
		return nil, nil
	}
	var item responsesItemWire
	dropped, err := decodeProviderValueVendorAware(body, &item, policy)
	if err != nil {
		return dropped, err
	}
	fields := make(map[string]struct{}, len(dropped))
	addVendorExtensionPaths(fields, "", dropped)

	switch item.Type {
	case "message":
		if err := collectResponsesValueVendorExtensions(fields, "content", item.Content, &[]responsesContentWire{}, policy); err != nil {
			return sortedVendorExtensionPaths(fields), err
		}
	case "reasoning":
		var summaries []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := collectResponsesValueVendorExtensions(fields, "summary", item.Summary, &summaries, policy); err != nil {
			return sortedVendorExtensionPaths(fields), err
		}
		if err := collectResponsesValueVendorExtensions(fields, "content", item.Content, &[]responsesContentWire{}, policy); err != nil {
			return sortedVendorExtensionPaths(fields), err
		}
	}
	return sortedVendorExtensionPaths(fields), nil
}

func collectResponsesValueVendorExtensions(
	destination map[string]struct{},
	prefix string,
	raw json.RawMessage,
	target any,
	policy llmprotocol.Policy,
) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	dropped, err := decodeProviderValueVendorAware(raw, target, policy)
	addVendorExtensionPaths(destination, prefix, dropped)
	return err
}

func addVendorExtensionPaths(destination map[string]struct{}, prefix string, fields []string) {
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

func sortedVendorExtensionPaths(fields map[string]struct{}) []string {
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

func appendResponsesVendorExtensionDiagnostics(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	prefix string,
	fields []string,
) {
	prefixed := make(map[string]struct{}, len(fields))
	addVendorExtensionPaths(prefixed, prefix, fields)
	appendVendorExtensionDiagnostics(
		diagnostics,
		policy,
		llmprotocol.OpenAIResponsesV1,
		sortedVendorExtensionPaths(prefixed),
	)
}
