package protocolcodec

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

type officialNestedInventory struct {
	Protocol       string                          `json:"protocol"`
	SchemaRevision string                          `json:"schema_revision"`
	Objects        []officialNestedInventoryObject `json:"objects"`
}

type officialNestedInventoryObject struct {
	Wire       string   `json:"wire"`
	Official   []string `json:"official"`
	Extensions []string `json:"extensions,omitempty"`
	Fixtures   []string `json:"fixtures"`
}

// fixtureJSONObjects returns the direct field set of every JSON object visible
// to a reviewer in a golden input. Stream fixtures store provider events inside
// SSE data strings, so those JSON documents are decoded recursively as well.
// Keeping object boundaries prevents an unrelated top-level id, type, or usage
// field from masquerading as evidence for a nested wire object.
func fixtureJSONObjects(t *testing.T, path string) []map[string]struct{} {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document any
	if err := json.Unmarshal(body, &document); err != nil {
		t.Fatalf("invalid JSON fixture %s: %v", path, err)
	}
	var objects []map[string]struct{}
	collectFixtureJSONObjects(document, &objects)
	return objects
}

func collectFixtureJSONObjects(value any, objects *[]map[string]struct{}) {
	switch typed := value.(type) {
	case map[string]any:
		fields := make(map[string]struct{}, len(typed))
		for field, child := range typed {
			fields[field] = struct{}{}
			collectFixtureJSONObjects(child, objects)
		}
		*objects = append(*objects, fields)
	case []any:
		for _, child := range typed {
			collectFixtureJSONObjects(child, objects)
		}
	case string:
		collectEmbeddedFixtureJSONObjects(typed, objects)
	}
}

func collectEmbeddedFixtureJSONObjects(value string, objects *[]map[string]struct{}) {
	candidates := []string{strings.TrimSpace(value)}
	for _, line := range strings.Split(value, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data:") {
			candidates = append(candidates, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	for _, candidate := range candidates {
		if candidate == "" || candidate == "[DONE]" || (candidate[0] != '{' && candidate[0] != '[') {
			continue
		}
		var embedded any
		if json.Unmarshal([]byte(candidate), &embedded) == nil {
			collectFixtureJSONObjects(embedded, objects)
		}
	}
}

func fixtureObjectMatchesWire(fields, allowed map[string]struct{}) bool {
	if len(fields) == 0 {
		return false
	}
	for field := range fields {
		if _, ok := allowed[field]; !ok {
			return false
		}
	}
	return true
}

func fixtureJSONDiscriminators(t *testing.T, path string) map[string]struct{} {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document any
	if err := json.Unmarshal(body, &document); err != nil {
		t.Fatalf("invalid JSON fixture %s: %v", path, err)
	}
	discriminators := make(map[string]struct{})
	collectFixtureJSONDiscriminators(document, discriminators)
	return discriminators
}

func collectFixtureJSONDiscriminators(value any, discriminators map[string]struct{}) {
	switch typed := value.(type) {
	case map[string]any:
		if discriminator, ok := typed["type"].(string); ok {
			discriminators[discriminator] = struct{}{}
		}
		for _, child := range typed {
			collectFixtureJSONDiscriminators(child, discriminators)
		}
	case []any:
		for _, child := range typed {
			collectFixtureJSONDiscriminators(child, discriminators)
		}
	case string:
		collectEmbeddedFixtureDiscriminators(typed, discriminators)
	}
}

func collectEmbeddedFixtureDiscriminators(value string, discriminators map[string]struct{}) {
	candidates := []string{strings.TrimSpace(value)}
	for _, line := range strings.Split(value, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data:") {
			candidates = append(candidates, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	for _, candidate := range candidates {
		if candidate == "" || candidate == "[DONE]" || (candidate[0] != '{' && candidate[0] != '[') {
			continue
		}
		var embedded any
		if json.Unmarshal([]byte(candidate), &embedded) == nil {
			collectFixtureJSONDiscriminators(embedded, discriminators)
		}
	}
}

func TestOfficialResponsesStreamEventsHaveHumanReadableJSONEvidence(t *testing.T) {
	capabilityPath := filepath.Join(
		"testdata", "golden", "capability", "046-responses-unsupported-stream-events-in.json",
	)
	body, err := os.ReadFile(capabilityPath)
	if err != nil {
		t.Fatal(err)
	}
	var capability goldenCapabilityInput
	if err := json.Unmarshal(body, &capability); err != nil {
		t.Fatal(err)
	}
	unsupported := make([]string, 0, len(capability.Cases))
	for _, testCase := range capability.Cases {
		unsupported = append(unsupported, testCase.Name)
	}
	sort.Strings(unsupported)
	if !reflect.DeepEqual(unsupported, officialUnsupportedResponsesStreamEvents) {
		t.Fatalf("unsupported Responses stream event fixtures are incomplete\n got: %v\nwant: %v", unsupported, officialUnsupportedResponsesStreamEvents)
	}

	evidence := make(map[string]struct{})
	for _, directory := range []string{"stream", "capability", "rejection"} {
		paths, err := filepath.Glob(filepath.Join("testdata", "golden", directory, "*-in.json"))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range paths {
			for discriminator := range fixtureJSONDiscriminators(t, path) {
				evidence[discriminator] = struct{}{}
			}
		}
	}
	for _, eventType := range append(
		append([]string(nil), officialSupportedResponsesStreamEvents...),
		officialUnsupportedResponsesStreamEvents...,
	) {
		if _, found := evidence[eventType]; !found {
			t.Errorf("official Responses stream event %q has no human-readable JSON fixture", eventType)
		}
	}
}

func TestOfficialAnthropicStreamUnionsHaveHumanReadableJSONEvidence(t *testing.T) {
	capabilityPath := filepath.Join(
		"testdata", "golden", "capability", "047-anthropic-unsupported-stream-unions-in.json",
	)
	body, err := os.ReadFile(capabilityPath)
	if err != nil {
		t.Fatal(err)
	}
	var capability goldenCapabilityInput
	if err := json.Unmarshal(body, &capability); err != nil {
		t.Fatal(err)
	}
	wantUnsupported := make([]string, 0,
		len(officialUnsupportedAnthropicStreamContentBlocks)+len(officialUnsupportedAnthropicStreamDeltas),
	)
	for _, discriminator := range officialUnsupportedAnthropicStreamContentBlocks {
		wantUnsupported = append(wantUnsupported, "content_block:"+discriminator)
	}
	for _, discriminator := range officialUnsupportedAnthropicStreamDeltas {
		wantUnsupported = append(wantUnsupported, "content_delta:"+discriminator)
	}
	sort.Strings(wantUnsupported)

	unsupported := make([]string, 0, len(capability.Cases))
	for _, testCase := range capability.Cases {
		unsupported = append(unsupported, testCase.Name)
	}
	sort.Strings(unsupported)
	if !reflect.DeepEqual(unsupported, wantUnsupported) {
		t.Fatalf("unsupported Anthropic stream union fixtures are incomplete\n got: %v\nwant: %v", unsupported, wantUnsupported)
	}

	evidence := make(map[string]struct{})
	for _, directory := range []string{"stream", "capability", "rejection"} {
		paths, err := filepath.Glob(filepath.Join("testdata", "golden", directory, "*-in.json"))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range paths {
			for discriminator := range fixtureJSONDiscriminators(t, path) {
				evidence[discriminator] = struct{}{}
			}
		}
	}
	wantEvidence := append([]string(nil), officialAnthropicStreamEvents...)
	wantEvidence = append(wantEvidence, officialSupportedAnthropicStreamContentBlocks...)
	wantEvidence = append(wantEvidence, officialUnsupportedAnthropicStreamContentBlocks...)
	wantEvidence = append(wantEvidence, officialSupportedAnthropicStreamDeltas...)
	wantEvidence = append(wantEvidence, officialUnsupportedAnthropicStreamDeltas...)
	for _, discriminator := range wantEvidence {
		if _, found := evidence[discriminator]; !found {
			t.Errorf("official Anthropic stream discriminator %q has no human-readable JSON fixture", discriminator)
		}
	}
}

// TestOfficialNestedJSONInventoriesAreClosed keeps nested schema coverage in
// reviewable JSON. Every inventory entry is tied to one or more translation
// fixtures, while reflection prevents a wire field from being added without a
// deliberate public-contract decision.
func TestOfficialNestedJSONInventoriesAreClosed(t *testing.T) {
	expectedRevisions := map[string]string{
		"openai_chat_completions": "2929dd85eb799bd308460bfe4a439cabd0eb74c8",
		"openai_responses":        "2929dd85eb799bd308460bfe4a439cabd0eb74c8",
		"anthropic_messages":      "d19dea9ed85bbb5fdb2d6f20fb6f903920ed23fa",
	}
	wires := map[string]reflect.Type{
		"chat_message":                   reflect.TypeOf(chatMessageWire{}),
		"chat_content":                   reflect.TypeOf(chatContentWire{}),
		"chat_tool":                      reflect.TypeOf(chatToolWire{}),
		"chat_stream_chunk":              reflect.TypeOf(chatChunkWire{}),
		"chat_stream_options":            reflect.TypeOf(chatStreamOptionsWire{}),
		"chat_stream_choice":             reflect.TypeOf(chatChunkChoiceWire{}),
		"chat_stream_delta":              reflect.TypeOf(chatChunkDeltaWire{}),
		"chat_audio_output":              reflect.TypeOf(chatAudioOutputWire{}),
		"chat_legacy_function_call":      reflect.TypeOf(chatLegacyCallWire{}),
		"chat_annotation":                reflect.TypeOf(chatAnnotationWire{}),
		"chat_url_citation":              reflect.TypeOf(chatURLCitationAnnotationWire{}),
		"chat_file_input":                reflect.TypeOf(chatFileWire{}),
		"chat_image_url_input":           reflect.TypeOf(chatImageURLWire{}),
		"chat_audio_input":               reflect.TypeOf(chatInputAudioWire{}),
		"chat_function_definition":       reflect.TypeOf(chatFunctionDefinitionWire{}),
		"chat_function_call":             reflect.TypeOf(chatFunctionCallWire{}),
		"chat_tool_call":                 reflect.TypeOf(chatToolCallWire{}),
		"chat_stream_tool_call":          reflect.TypeOf(chatChunkToolCallWire{}),
		"chat_response_format":           reflect.TypeOf(chatOutputWire{}),
		"chat_response_choice":           reflect.TypeOf(chatChoiceWire{}),
		"chat_logprobs":                  reflect.TypeOf(chatLogprobsWire{}),
		"chat_token_logprob":             reflect.TypeOf(chatTokenLogprobWire{}),
		"chat_top_token_logprob":         reflect.TypeOf(chatTopTokenLogprobWire{}),
		"chat_error":                     reflect.TypeOf(chatErrorWire{}),
		"chat_prompt_usage_details":      reflect.TypeOf(chatPromptTokensDetailsWire{}),
		"chat_completion_usage_details":  reflect.TypeOf(chatCompletionTokensDetailsWire{}),
		"openai_transport_error":         reflect.TypeOf(openAITransportErrorWire{}),
		"openai_transport_error_detail":  reflect.TypeOf(openAITransportErrorDetailWire{}),
		"responses_reasoning":            reflect.TypeOf(responsesReasoningWire{}),
		"responses_stream_options":       reflect.TypeOf(responsesStreamOptionsWire{}),
		"responses_text":                 reflect.TypeOf(responsesTextWire{}),
		"responses_output_format":        reflect.TypeOf(responsesFormatWire{}),
		"responses_function_tool":        reflect.TypeOf(responsesToolWire{}),
		"responses_content":              reflect.TypeOf(responsesContentWire{}),
		"responses_url_citation":         reflect.TypeOf(responsesAnnotationWire{}),
		"responses_error":                reflect.TypeOf(responsesErrorWire{}),
		"responses_stream_error":         reflect.TypeOf(responsesTransportErrorEventWire{}),
		"responses_item":                 reflect.TypeOf(responsesItemWire{}),
		"responses_stream_event":         reflect.TypeOf(responsesEventWire{}),
		"responses_input_usage_details":  reflect.TypeOf(responsesInputUsageDetails{}),
		"responses_output_usage_details": reflect.TypeOf(responsesOutputUsageDetails{}),
		"anthropic_message":              reflect.TypeOf(anthropicMessageWire{}),
		"anthropic_metadata":             reflect.TypeOf(anthropicMetadataWire{}),
		"anthropic_content":              reflect.TypeOf(anthropicContentWire{}),
		"anthropic_stream_event":         reflect.TypeOf(anthropicEventWire{}),
		"anthropic_stream_delta":         reflect.TypeOf(anthropicDeltaWire{}),
		"anthropic_media_source":         reflect.TypeOf(anthropicMediaSourceWire{}),
		"anthropic_tool":                 reflect.TypeOf(anthropicToolWire{}),
		"anthropic_cache_control":        reflect.TypeOf(anthropicCacheControlWire{}),
		"anthropic_tool_choice":          reflect.TypeOf(anthropicToolChoiceWire{}),
		"anthropic_output_config":        reflect.TypeOf(anthropicOutputConfigWire{}),
		"anthropic_thinking":             reflect.TypeOf(anthropicThinkingWire{}),
		"anthropic_json_output_format":   reflect.TypeOf(anthropicJSONOutputFormatWire{}),
		"anthropic_cache_creation_usage": reflect.TypeOf(anthropicCacheCreationUsageWire{}),
		"anthropic_output_usage_details": reflect.TypeOf(anthropicOutputUsageDetailsWire{}),
		"anthropic_server_tool_usage":    reflect.TypeOf(anthropicServerToolUsageWire{}),
		"anthropic_error":                reflect.TypeOf(anthropicErrorWire{}),
		"anthropic_transport_error":      reflect.TypeOf(anthropicTransportErrorWire{}),
		"anthropic_message_delta_usage":  reflect.TypeOf(anthropicMessageDeltaUsageWire{}),
	}

	paths, err := filepath.Glob(filepath.Join("testdata", "contracts", "*-nested-fields.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 3 {
		t.Fatalf("nested inventory files = %d, want one for each public protocol", len(paths))
	}
	seen := make(map[string]string, len(wires))
	seenProtocols := make(map[string]string, len(expectedRevisions))
	for _, path := range paths {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var inventory officialNestedInventory
		if err := json.Unmarshal(body, &inventory); err != nil {
			t.Fatalf("invalid nested inventory %s: %v", path, err)
		}
		expectedRevision, knownProtocol := expectedRevisions[inventory.Protocol]
		if !knownProtocol || inventory.SchemaRevision != expectedRevision {
			t.Fatalf(
				"nested inventory %s has protocol/revision %q/%q, want a pinned published contract",
				path, inventory.Protocol, inventory.SchemaRevision,
			)
		}
		if previous, duplicate := seenProtocols[inventory.Protocol]; duplicate {
			t.Fatalf("protocol %q appears in both %s and %s", inventory.Protocol, previous, path)
		}
		seenProtocols[inventory.Protocol] = path
		for _, object := range inventory.Objects {
			wire, ok := wires[object.Wire]
			if !ok {
				t.Fatalf("nested inventory %s references unknown wire %q", path, object.Wire)
			}
			if previous, duplicate := seen[object.Wire]; duplicate {
				t.Fatalf("nested wire %q appears in both %s and %s", object.Wire, previous, path)
			}
			seen[object.Wire] = path
			want := append(append([]string(nil), object.Official...), object.Extensions...)
			sort.Strings(want)
			if got := jsonFieldNames(wire); !reflect.DeepEqual(got, want) {
				t.Fatalf("nested inventory %s/%s drifted\n got: %v\nwant: %v", inventory.Protocol, object.Wire, got, want)
			}
			if len(object.Fixtures) == 0 {
				t.Fatalf("nested inventory %s/%s has no reviewable translation fixture", inventory.Protocol, object.Wire)
			}
			allowedFields := make(map[string]struct{}, len(want))
			for _, field := range want {
				allowedFields[field] = struct{}{}
			}
			visibleFields := make(map[string]struct{})
			for _, fixture := range object.Fixtures {
				clean := filepath.Clean(fixture)
				if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) ||
					!strings.HasSuffix(clean, "-in.json") {
					t.Fatalf("nested inventory %s/%s has invalid fixture path %q", inventory.Protocol, object.Wire, fixture)
				}
				fixturePath := filepath.Join("testdata", "golden", clean)
				if _, err := os.Stat(fixturePath); err != nil {
					t.Fatalf("nested inventory %s/%s fixture %q: %v", inventory.Protocol, object.Wire, fixture, err)
				}
				for _, fields := range fixtureJSONObjects(t, fixturePath) {
					if !fixtureObjectMatchesWire(fields, allowedFields) {
						continue
					}
					for field := range fields {
						visibleFields[field] = struct{}{}
					}
				}
			}
			for _, field := range want {
				if _, ok := visibleFields[field]; !ok {
					t.Fatalf(
						"nested inventory %s/%s field %q has no human-readable JSON evidence in %v",
						inventory.Protocol, object.Wire, field, object.Fixtures,
					)
				}
			}
		}
	}
	if len(seenProtocols) != len(expectedRevisions) {
		t.Fatalf("nested JSON inventories cover %d protocols, want %d", len(seenProtocols), len(expectedRevisions))
	}
	if len(seen) != len(wires) {
		missing := make([]string, 0, len(wires)-len(seen))
		for wire := range wires {
			if _, ok := seen[wire]; !ok {
				missing = append(missing, wire)
			}
		}
		sort.Strings(missing)
		t.Fatalf("nested JSON inventories are incomplete: %v", missing)
	}
}
