package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const updateProtocolGoldensEnv = "UPDATE_PROTOCOL_GOLDENS"

var goldenFormats = []struct {
	name   string
	format llmprotocol.WireFormat
}{
	{name: "chat", format: llmprotocol.OpenAIChatV1},
	{name: "responses", format: llmprotocol.OpenAIResponsesV1},
	{name: "anthropic", format: llmprotocol.AnthropicMessagesV1},
}

func TestGoldenRequestTranslationMatrix(t *testing.T) {
	runGoldenTranslationMatrix(t, "request", func(
		engine *Engine,
		source,
		target llmprotocol.WireFormat,
		body []byte,
	) ([]byte, error) {
		result, err := engine.TranslateRequest(source, target, body, func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		})
		if err == nil {
			if _, _, _, decodeErr := engine.DecodeRequest(target, result.Body); decodeErr != nil {
				return nil, fmt.Errorf("target request does not satisfy its own codec contract: %w", decodeErr)
			}
		}
		return result.Body, err
	})
}

func TestGoldenResponseTranslationMatrix(t *testing.T) {
	runGoldenTranslationMatrix(t, "response", func(
		engine *Engine,
		source,
		target llmprotocol.WireFormat,
		body []byte,
	) ([]byte, error) {
		result, err := engine.TranslateResponse(source, target, body, func(response *llmprotocol.Response) error {
			response.Model = "routed-model"
			return nil
		})
		if err == nil {
			if _, _, _, decodeErr := engine.DecodeResponse(target, result.Body); decodeErr != nil {
				return nil, fmt.Errorf("target response does not satisfy its own codec contract: %w", decodeErr)
			}
		}
		return result.Body, err
	})
}

func TestGoldenTransportErrorTranslationMatrix(t *testing.T) {
	runGoldenTranslationMatrix(t, "error", func(
		engine *Engine,
		source,
		target llmprotocol.WireFormat,
		body []byte,
	) ([]byte, error) {
		result, err := engine.TranslateTransportError(source, target, body, nil)
		return result.Body, err
	})
}

type goldenStreamInput struct {
	PublicModel    string              `json:"public_model"`
	ProviderModel  string              `json:"provider_model"`
	ResponseID     string              `json:"response_id,omitempty"`
	FinalizeReason string              `json:"finalize_reason,omitempty"`
	Options        goldenStreamOptions `json:"stream_options,omitempty"`
	Chunks         []string            `json:"chunks"`
}

type goldenStreamOptions struct {
	IncludeUsage       *bool `json:"include_usage,omitempty"`
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

type goldenStreamTranscript struct {
	Frames []goldenStreamFrame `json:"frames"`
}

type goldenStreamFrame struct {
	Event string `json:"event"`
	Data  any    `json:"data"`
}

func TestGoldenStreamTranslationMatrix(t *testing.T) {
	directory := filepath.Join("testdata", "golden", "stream")
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) == 0 {
		t.Fatalf("no stream protocol golden inputs found in %s", directory)
	}
	engine := NewBuiltinEngine()
	for _, inputPath := range inputs {
		inputPath := inputPath
		prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
		source, err := goldenSourceFormat(prefix)
		if err != nil {
			t.Fatal(err)
		}
		body, err := os.ReadFile(inputPath)
		if err != nil {
			t.Fatal(err)
		}
		var input goldenStreamInput
		if err := json.Unmarshal(body, &input); err != nil {
			t.Fatalf("invalid stream golden input %s: %v", inputPath, err)
		}
		if input.PublicModel == "" || input.ProviderModel == "" || len(input.Chunks) == 0 {
			t.Fatalf("stream golden input %s requires public_model, provider_model, and chunks", inputPath)
		}
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				outputFrames := translateGoldenStream(t, engine, source, target.format, input, input.Chunks)
				verifyGoldenTargetStream(t, engine, target.format, input, outputFrames)
				actual := marshalGoldenStreamTranscript(t, outputFrames)
				expectedPath := filepath.Join(directory, prefix+"-"+target.name+"-out.json")
				if os.Getenv(updateProtocolGoldensEnv) == "1" {
					if err := os.WriteFile(expectedPath, actual, 0o644); err != nil {
						t.Fatal(err)
					}
				}
				expected, err := os.ReadFile(expectedPath)
				if err != nil {
					t.Fatalf("missing golden output %s: %v", expectedPath, err)
				}
				if !jsonEquivalent(expected, actual) {
					t.Fatalf("stream protocol translation drifted\nexpected (%s):\n%s\nactual:\n%s", expectedPath, prettyJSON(t, expected), actual)
				}
			})
		}
	}
}

func TestGoldenStreamTranslationIsChunkBoundaryIndependent(t *testing.T) {
	directory := filepath.Join("testdata", "golden", "stream")
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	engine := NewBuiltinEngine()
	for _, inputPath := range inputs {
		prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
		source, err := goldenSourceFormat(prefix)
		if err != nil {
			t.Fatal(err)
		}
		descriptor, err := os.ReadFile(inputPath)
		if err != nil {
			t.Fatal(err)
		}
		var input goldenStreamInput
		if err := json.Unmarshal(descriptor, &input); err != nil {
			t.Fatal(err)
		}
		byteChunks := splitGoldenChunksByByte(input.Chunks)
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				baseline := marshalGoldenStreamTranscript(
					t, translateGoldenStream(t, engine, source, target.format, input, input.Chunks),
				)
				byteSplit := marshalGoldenStreamTranscript(
					t, translateGoldenStream(t, engine, source, target.format, input, byteChunks),
				)
				if !jsonEquivalent(baseline, byteSplit) {
					t.Fatalf("translation changed with one-byte transport chunks\nbaseline:\n%s\nbyte split:\n%s", baseline, byteSplit)
				}
			})
		}
	}
}

func translateGoldenStream(
	t *testing.T,
	engine *Engine,
	source,
	target llmprotocol.WireFormat,
	input goldenStreamInput,
	chunks []string,
) [][]byte {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: input.PublicModel,
		ProviderModel: input.ProviderModel, ResponseID: input.ResponseID,
		Options: llmprotocol.StreamOptions{
			IncludeUsage: input.Options.IncludeUsage, IncludeObfuscation: input.Options.IncludeObfuscation,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var outputFrames [][]byte
	for index, chunk := range chunks {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		outputFrames = append(outputFrames, frames...)
		if pushErr != nil {
			t.Fatalf("stream chunk %d: %v", index, pushErr)
		}
	}
	frames, _, _, err := stream.Finalize(goldenFinalizeReason(t, input.FinalizeReason))
	outputFrames = append(outputFrames, frames...)
	if err != nil {
		t.Fatal(err)
	}
	return outputFrames
}

func verifyGoldenTargetStream(
	t *testing.T,
	engine *Engine,
	target llmprotocol.WireFormat,
	input goldenStreamInput,
	frames [][]byte,
) {
	t.Helper()
	verify, err := engine.NewStream(target, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: input.PublicModel,
		ProviderModel: input.PublicModel, ResponseID: input.ResponseID,
		Options: llmprotocol.StreamOptions{
			IncludeUsage: input.Options.IncludeUsage, IncludeObfuscation: input.Options.IncludeObfuscation,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := verify.Push(bytes.Join(frames, nil))
	if err != nil {
		t.Fatalf("target stream does not satisfy its own codec contract: %v\n%s", err, bytes.Join(frames, nil))
	}
	_, finalEvents, _, err := verify.Finalize(nil)
	if err != nil {
		t.Fatalf("target stream did not finalize cleanly: %v\n%s", err, bytes.Join(frames, nil))
	}
	events = append(events, finalEvents...)
	terminal := 0
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
			terminal++
		}
	}
	if terminal != 1 {
		t.Fatalf("target stream emitted %d semantic terminals, want 1\n%s", terminal, bytes.Join(frames, nil))
	}
}

func splitGoldenChunksByByte(chunks []string) []string {
	var result []string
	for _, chunk := range chunks {
		for _, value := range []byte(chunk) {
			result = append(result, string([]byte{value}))
		}
	}
	return result
}

func goldenFinalizeReason(t *testing.T, reason string) error {
	t.Helper()
	switch reason {
	case "":
		return nil
	case "canceled":
		return context.Canceled
	case "deadline_exceeded":
		return context.DeadlineExceeded
	default:
		t.Fatalf("golden stream fixture requested unknown finalize reason %q", reason)
		return nil
	}
}

type goldenRejectionInput struct {
	Operation     string                `json:"operation"`
	Body          json.RawMessage       `json:"body,omitempty"`
	BodyJSON      string                `json:"body_json,omitempty"`
	PublicModel   string                `json:"public_model,omitempty"`
	ProviderModel string                `json:"provider_model,omitempty"`
	Chunks        []string              `json:"chunks,omitempty"`
	Limits        goldenRejectionLimits `json:"limits,omitempty"`
}

type goldenRejectionLimits struct {
	BodyBytes     int `json:"body_bytes,omitempty"`
	JSONDepth     int `json:"json_depth,omitempty"`
	SSEFrameBytes int `json:"sse_frame_bytes,omitempty"`
	Events        int `json:"events,omitempty"`
}

type goldenProtocolError struct {
	Category llmprotocol.ErrorCategory `json:"category"`
	Code     string                    `json:"code"`
}

type goldenCapabilityInput struct {
	Operation string                 `json:"operation"`
	Body      json.RawMessage        `json:"body,omitempty"`
	Base      json.RawMessage        `json:"base,omitempty"`
	Cases     []goldenCapabilityCase `json:"cases,omitempty"`
	Stream    *goldenStreamInput     `json:"stream,omitempty"`
}

type goldenCapabilityCase struct {
	Name   string                     `json:"name"`
	Body   json.RawMessage            `json:"body,omitempty"`
	Patch  map[string]json.RawMessage `json:"patch,omitempty"`
	Stream *goldenStreamInput         `json:"stream,omitempty"`
}

type goldenCapabilityOutcome struct {
	Status      string               `json:"status"`
	Body        any                  `json:"body,omitempty"`
	Error       *goldenProtocolError `json:"error,omitempty"`
	Diagnostics []goldenDiagnostic   `json:"diagnostics,omitempty"`
}

type goldenDiagnostic struct {
	Field  string                       `json:"field"`
	Action llmprotocol.DiagnosticAction `json:"action"`
}

type goldenCapabilityCaseOutcome struct {
	Name    string                  `json:"name"`
	Outcome goldenCapabilityOutcome `json:"outcome"`
}

type goldenCapabilityTranscript struct {
	Cases []goldenCapabilityCaseOutcome `json:"cases"`
}

func TestGoldenOfficialRequestFieldCasesAreClosed(t *testing.T) {
	tests := []struct {
		file string
		wire any
	}{
		{file: "013-chat-official-request-fields-in.json", wire: chatRequestWire{}},
		{file: "014-responses-official-request-fields-in.json", wire: responsesRequestWire{}},
		{file: "015-anthropic-official-request-fields-in.json", wire: anthropicRequestWire{}},
	}
	for _, test := range tests {
		t.Run(test.file, func(t *testing.T) {
			body, err := os.ReadFile(filepath.Join("testdata", "golden", "capability", test.file))
			if err != nil {
				t.Fatal(err)
			}
			var input goldenCapabilityInput
			if err := json.Unmarshal(body, &input); err != nil {
				t.Fatal(err)
			}
			actual := make([]string, 0, len(input.Cases))
			for _, testCase := range input.Cases {
				actual = append(actual, testCase.Name)
			}
			sort.Strings(actual)
			expected := jsonFieldNames(reflect.TypeOf(test.wire))
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("official request field fixtures are incomplete\n got: %v\nwant: %v", actual, expected)
			}
		})
	}
}

func TestGoldenOfficialResponseFieldCasesAreClosed(t *testing.T) {
	tests := []struct {
		file string
		wire any
	}{
		{file: "016-chat-official-response-fields-in.json", wire: chatResponseWire{}},
		{file: "017-responses-official-response-fields-in.json", wire: responsesResponseWire{}},
		{file: "018-anthropic-official-response-fields-in.json", wire: anthropicResponseWire{}},
	}
	for _, test := range tests {
		t.Run(test.file, func(t *testing.T) {
			body, err := os.ReadFile(filepath.Join("testdata", "golden", "capability", test.file))
			if err != nil {
				t.Fatal(err)
			}
			var input goldenCapabilityInput
			if err := json.Unmarshal(body, &input); err != nil {
				t.Fatal(err)
			}
			actual := make([]string, 0, len(input.Cases))
			for _, testCase := range input.Cases {
				actual = append(actual, testCase.Name)
			}
			sort.Strings(actual)
			expected := jsonFieldNames(reflect.TypeOf(test.wire))
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("official response field fixtures are incomplete\n got: %v\nwant: %v", actual, expected)
			}
		})
	}
}

func TestGoldenOfficialToolDiscriminatorCasesAreClosed(t *testing.T) {
	tests := []struct {
		file string
		want []string
	}{
		{
			file: "035-responses-official-tool-discriminators-in.json",
			want: fields(
				"apply_patch", "code_interpreter", "computer", "computer_use_preview", "custom",
				"file_search", "function", "image_generation", "local_shell", "mcp", "namespace",
				"programmatic_tool_calling", "shell", "tool_search", "web_search", "web_search_preview",
			),
		},
		{
			file: "036-anthropic-official-tool-discriminators-in.json",
			want: fields(
				"bash_20250124", "browser_toolset_20260801", "code_execution_20250522",
				"code_execution_20250825", "code_execution_20260120", "code_execution_20260521",
				"computer_toolset_20260801", "custom", "memory_20250818", "text_editor_20250124",
				"text_editor_20250429", "text_editor_20250728", "tool_search_tool_bm25_20251119",
				"tool_search_tool_regex_20251119", "web_fetch_20250910", "web_fetch_20260209",
				"web_fetch_20260309", "web_fetch_20260318", "web_search_20250305",
				"web_search_20260209", "web_search_20260318",
			),
		},
	}
	for _, test := range tests {
		t.Run(test.file, func(t *testing.T) {
			body, err := os.ReadFile(filepath.Join("testdata", "golden", "capability", test.file))
			if err != nil {
				t.Fatal(err)
			}
			var input goldenCapabilityInput
			if err := json.Unmarshal(body, &input); err != nil {
				t.Fatal(err)
			}
			got := make([]string, 0, len(input.Cases))
			for _, testCase := range input.Cases {
				got = append(got, testCase.Name)
			}
			sort.Strings(got)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("official tool discriminator fixtures are incomplete\n got: %v\nwant: %v", got, test.want)
			}
		})
	}
}

// TestGoldenCapabilityMatrix records both successful translations and typed
// capability failures. Provider-specific features must never disappear merely
// because one target in the 3x3 matrix cannot represent them.
func TestGoldenCapabilityMatrix(t *testing.T) {
	directory := filepath.Join("testdata", "golden", "capability")
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) == 0 {
		t.Fatalf("no capability protocol golden inputs found in %s", directory)
	}
	engine := NewBuiltinEngine()
	for _, inputPath := range inputs {
		inputPath := inputPath
		prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
		source, err := goldenSourceFormat(prefix)
		if err != nil {
			t.Fatal(err)
		}
		descriptor, err := os.ReadFile(inputPath)
		if err != nil {
			t.Fatal(err)
		}
		var input goldenCapabilityInput
		if err := json.Unmarshal(descriptor, &input); err != nil {
			t.Fatalf("invalid capability golden input %s: %v", inputPath, err)
		}
		validateGoldenCapabilityInput(t, inputPath, input)
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				actualValue := runCapabilityCases(t, engine, source, target.format, input)
				if fragmented, hasStream := fragmentGoldenCapabilityInput(input); hasStream {
					fragmentedValue := runCapabilityCases(t, engine, source, target.format, fragmented)
					if !reflect.DeepEqual(actualValue, fragmentedValue) {
						t.Fatalf("stream capability outcome changed with one-byte transport chunks\nbaseline: %#v\nfragmented: %#v", actualValue, fragmentedValue)
					}
				}
				actual, err := json.MarshalIndent(actualValue, "", "  ")
				if err != nil {
					t.Fatal(err)
				}
				actual = append(actual, '\n')
				expectedPath := filepath.Join(directory, prefix+"-"+target.name+"-out.json")
				if os.Getenv(updateProtocolGoldensEnv) == "1" {
					if err := os.WriteFile(expectedPath, actual, 0o644); err != nil {
						t.Fatal(err)
					}
				}
				expected, err := os.ReadFile(expectedPath)
				if err != nil {
					t.Fatalf("missing golden output %s: %v", expectedPath, err)
				}
				if !jsonEquivalent(expected, actual) {
					t.Fatalf("capability outcome drifted\nexpected (%s):\n%s\nactual:\n%s", expectedPath, expected, actual)
				}
			})
		}
	}
}

func validateGoldenCapabilityInput(t *testing.T, path string, input goldenCapabilityInput) {
	t.Helper()
	if input.Operation != "request" && input.Operation != "response" && input.Operation != "stream" {
		t.Fatalf("capability fixture %s has unsupported operation %q", path, input.Operation)
	}
	if len(input.Cases) == 0 {
		if input.Operation == "stream" && input.Stream == nil {
			t.Fatalf("stream capability fixture %s has no stream", path)
		}
		if input.Operation != "stream" && len(input.Body) == 0 {
			t.Fatalf("%s capability fixture %s has no body", input.Operation, path)
		}
		return
	}
	if len(input.Body) != 0 || input.Stream != nil {
		t.Fatalf("capability fixture %s cannot combine top-level body/stream with named cases", path)
	}
	for _, testCase := range input.Cases {
		hasBody := len(testCase.Body) != 0
		hasPatch := len(testCase.Patch) != 0
		hasStream := testCase.Stream != nil
		if hasBody && hasPatch {
			t.Fatalf("capability fixture %s case %q cannot combine body and patch", path, testCase.Name)
		}
		if input.Operation == "stream" {
			if !hasStream || hasBody || hasPatch {
				t.Fatalf("stream capability fixture %s case %q must contain exactly one stream", path, testCase.Name)
			}
			continue
		}
		if hasStream || (!hasBody && !hasPatch) {
			t.Fatalf("%s capability fixture %s case %q must contain exactly one body or patch", input.Operation, path, testCase.Name)
		}
		if hasPatch && len(input.Base) == 0 {
			t.Fatalf("capability fixture %s case %q uses a patch without a base", path, testCase.Name)
		}
	}
}

func runCapabilityCases(
	t *testing.T,
	engine *Engine,
	source,
	target llmprotocol.WireFormat,
	input goldenCapabilityInput,
) any {
	t.Helper()
	if len(input.Cases) == 0 {
		return runCapabilityOutcome(t, engine, source, target, input)
	}
	transcript := goldenCapabilityTranscript{Cases: make([]goldenCapabilityCaseOutcome, 0, len(input.Cases))}
	seen := make(map[string]struct{}, len(input.Cases))
	for _, testCase := range input.Cases {
		if strings.TrimSpace(testCase.Name) == "" {
			t.Fatal("capability case name is required")
		}
		if _, duplicate := seen[testCase.Name]; duplicate {
			t.Fatalf("duplicate capability case name %q", testCase.Name)
		}
		seen[testCase.Name] = struct{}{}
		body := testCase.Body
		if testCase.Stream == nil && len(body) == 0 {
			body = applyGoldenJSONPatch(t, input.Base, testCase.Patch)
		}
		caseInput := input
		caseInput.Body = body
		caseInput.Cases = nil
		if testCase.Stream != nil {
			caseInput.Stream = testCase.Stream
		}
		transcript.Cases = append(transcript.Cases, goldenCapabilityCaseOutcome{
			Name:    testCase.Name,
			Outcome: runCapabilityOutcome(t, engine, source, target, caseInput),
		})
	}
	return transcript
}

func fragmentGoldenCapabilityInput(input goldenCapabilityInput) (goldenCapabilityInput, bool) {
	fragmented := input
	hasStream := false
	if input.Stream != nil {
		stream := *input.Stream
		stream.Chunks = splitGoldenChunksByByte(input.Stream.Chunks)
		fragmented.Stream = &stream
		hasStream = true
	}
	if len(input.Cases) != 0 {
		fragmented.Cases = append([]goldenCapabilityCase(nil), input.Cases...)
		for index := range fragmented.Cases {
			if input.Cases[index].Stream == nil {
				continue
			}
			stream := *input.Cases[index].Stream
			stream.Chunks = splitGoldenChunksByByte(input.Cases[index].Stream.Chunks)
			fragmented.Cases[index].Stream = &stream
			hasStream = true
		}
	}
	return fragmented, hasStream
}

func runCapabilityOutcome(
	t *testing.T,
	engine *Engine,
	source,
	target llmprotocol.WireFormat,
	input goldenCapabilityInput,
) goldenCapabilityOutcome {
	t.Helper()
	translated, diagnostics, translateErr := runCapabilityTranslation(t, engine, source, target, input)
	outcome := goldenCapabilityOutcome{Status: "ok"}
	for _, diagnostic := range diagnostics {
		outcome.Diagnostics = append(outcome.Diagnostics, goldenDiagnostic{Field: diagnostic.Field, Action: diagnostic.Action})
	}
	if translateErr != nil {
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(translateErr, &protocolError) {
			t.Fatalf("capability case returned %T, want *llmprotocol.ProtocolError: %v", translateErr, translateErr)
		}
		outcome.Status = "error"
		outcome.Error = &goldenProtocolError{Category: protocolError.Category, Code: protocolError.Code}
		return outcome
	}
	decoder := json.NewDecoder(bytes.NewReader(translated))
	decoder.UseNumber()
	if err := decoder.Decode(&outcome.Body); err != nil {
		t.Fatalf("capability output is invalid JSON: %v\n%s", err, translated)
	}
	return outcome
}

func applyGoldenJSONPatch(
	t *testing.T,
	base json.RawMessage,
	patch map[string]json.RawMessage,
) json.RawMessage {
	t.Helper()
	if len(base) == 0 {
		t.Fatal("capability case patch requires a base object")
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(base, &object); err != nil {
		t.Fatalf("capability base must be a JSON object: %v", err)
	}
	for field, value := range patch {
		object[field] = append(json.RawMessage(nil), value...)
	}
	body, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func runCapabilityTranslation(
	t *testing.T,
	engine *Engine,
	source,
	target llmprotocol.WireFormat,
	input goldenCapabilityInput,
) ([]byte, llmprotocol.Diagnostics, error) {
	switch input.Operation {
	case "request":
		result, err := engine.TranslateRequest(source, target, input.Body, func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		})
		if err == nil {
			if _, _, _, decodeErr := engine.DecodeRequest(target, result.Body); decodeErr != nil {
				return nil, result.Diagnostics, fmt.Errorf("target request does not satisfy its own codec contract: %w", decodeErr)
			}
		}
		return result.Body, result.Diagnostics, err
	case "response":
		result, err := engine.TranslateResponse(source, target, input.Body, func(response *llmprotocol.Response) error {
			response.Model = "routed-model"
			return nil
		})
		if err == nil {
			if _, _, _, decodeErr := engine.DecodeResponse(target, result.Body); decodeErr != nil {
				return nil, result.Diagnostics, fmt.Errorf("target response does not satisfy its own codec contract: %w", decodeErr)
			}
		}
		return result.Body, result.Diagnostics, err
	case "stream":
		if input.Stream == nil || input.Stream.PublicModel == "" || input.Stream.ProviderModel == "" || len(input.Stream.Chunks) == 0 {
			return nil, nil, fmt.Errorf("stream capability case requires public_model, provider_model, and chunks")
		}
		frames, diagnostics, err := translateGoldenCapabilityStream(engine, source, target, *input.Stream)
		if err != nil {
			return nil, diagnostics, err
		}
		verifyGoldenTargetStream(t, engine, target, *input.Stream, frames)
		return marshalGoldenStreamTranscript(t, frames), diagnostics, nil
	default:
		return nil, nil, fmt.Errorf("unsupported capability operation %q", input.Operation)
	}
}

func translateGoldenCapabilityStream(
	engine *Engine,
	source,
	target llmprotocol.WireFormat,
	input goldenStreamInput,
) ([][]byte, llmprotocol.Diagnostics, error) {
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: input.PublicModel,
		ProviderModel: input.ProviderModel, ResponseID: input.ResponseID,
		Options: llmprotocol.StreamOptions{
			IncludeUsage: input.Options.IncludeUsage, IncludeObfuscation: input.Options.IncludeObfuscation,
		},
	})
	if err != nil {
		return nil, nil, err
	}
	var frames [][]byte
	var diagnostics llmprotocol.Diagnostics
	for _, chunk := range input.Chunks {
		output, _, chunkDiagnostics, pushErr := stream.Push([]byte(chunk))
		frames = append(frames, output...)
		diagnostics = appendDiagnostics(diagnostics, chunkDiagnostics, llmprotocol.DefaultPolicy().Limits.Diagnostics)
		if pushErr != nil {
			return frames, diagnostics, pushErr
		}
	}
	output, _, finalDiagnostics, finalErr := stream.Finalize(nil)
	frames = append(frames, output...)
	diagnostics = appendDiagnostics(diagnostics, finalDiagnostics, llmprotocol.DefaultPolicy().Limits.Diagnostics)
	return frames, diagnostics, finalErr
}

func TestGoldenRejectionMatrix(t *testing.T) {
	directory := filepath.Join("testdata", "golden", "rejection")
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) == 0 {
		t.Fatalf("no rejection protocol golden inputs found in %s", directory)
	}
	engine := NewBuiltinEngine()
	for _, inputPath := range inputs {
		inputPath := inputPath
		prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
		source, err := goldenSourceFormat(prefix)
		if err != nil {
			t.Fatal(err)
		}
		descriptor, err := os.ReadFile(inputPath)
		if err != nil {
			t.Fatal(err)
		}
		var input goldenRejectionInput
		if err := json.Unmarshal(descriptor, &input); err != nil {
			t.Fatalf("invalid rejection golden input %s: %v", inputPath, err)
		}
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				translateErr := runRejectedTranslation(engine, source, target.format, input)
				if translateErr == nil {
					t.Fatalf("rejection case %s unexpectedly succeeded", inputPath)
				}
				var protocolError *llmprotocol.ProtocolError
				if !errors.As(translateErr, &protocolError) {
					t.Fatalf("rejection case returned %T, want *llmprotocol.ProtocolError: %v", translateErr, translateErr)
				}
				actual, err := json.MarshalIndent(goldenProtocolError{Category: protocolError.Category, Code: protocolError.Code}, "", "  ")
				if err != nil {
					t.Fatal(err)
				}
				actual = append(actual, '\n')
				expectedPath := filepath.Join(directory, prefix+"-"+target.name+"-out.json")
				if os.Getenv(updateProtocolGoldensEnv) == "1" {
					if err := os.WriteFile(expectedPath, actual, 0o644); err != nil {
						t.Fatal(err)
					}
				}
				expected, err := os.ReadFile(expectedPath)
				if err != nil {
					t.Fatalf("missing golden output %s: %v", expectedPath, err)
				}
				if !jsonEquivalent(expected, actual) {
					t.Fatalf("rejection contract drifted\nexpected (%s):\n%s\nactual:\n%s", expectedPath, expected, actual)
				}
			})
		}
	}
}

func TestGoldenStreamRejectionsAreChunkBoundaryIndependent(t *testing.T) {
	directory := filepath.Join("testdata", "golden", "rejection")
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	engine := NewBuiltinEngine()
	for _, inputPath := range inputs {
		prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
		source, err := goldenSourceFormat(prefix)
		if err != nil {
			t.Fatal(err)
		}
		descriptor, err := os.ReadFile(inputPath)
		if err != nil {
			t.Fatal(err)
		}
		var input goldenRejectionInput
		if err := json.Unmarshal(descriptor, &input); err != nil {
			t.Fatal(err)
		}
		if input.Operation != "stream" {
			continue
		}
		byteSplit := input
		byteSplit.Chunks = splitGoldenChunksByByte(input.Chunks)
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				baseline := goldenProtocolErrorFrom(t, runRejectedTranslation(engine, source, target.format, input))
				fragmented := goldenProtocolErrorFrom(t, runRejectedTranslation(engine, source, target.format, byteSplit))
				if baseline != fragmented {
					t.Fatalf("rejection changed with one-byte transport chunks: baseline=%+v fragmented=%+v", baseline, fragmented)
				}
			})
		}
	}
}

func TestGoldenStreamRejectionsPoisonAndFailClosedAcrossProtocolMatrix(t *testing.T) {
	directory := filepath.Join("testdata", "golden", "rejection")
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	engine := NewBuiltinEngine()
	for _, inputPath := range inputs {
		prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
		source, err := goldenSourceFormat(prefix)
		if err != nil {
			t.Fatal(err)
		}
		descriptor, err := os.ReadFile(inputPath)
		if err != nil {
			t.Fatal(err)
		}
		var input goldenRejectionInput
		if err := json.Unmarshal(descriptor, &input); err != nil {
			t.Fatal(err)
		}
		if input.Operation != "stream" {
			continue
		}
		inputEngine, err := goldenRejectionEngine(engine, input)
		if err != nil {
			t.Fatal(err)
		}
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				stream, err := inputEngine.NewStream(source, target.format, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: input.PublicModel,
					ProviderModel: input.ProviderModel,
				})
				if err != nil {
					t.Fatal(err)
				}
				var wire []byte
				var firstErr error
				for _, chunk := range input.Chunks {
					frames, _, _, pushErr := stream.Push([]byte(chunk))
					wire = appendGoldenWireFrames(wire, frames)
					if pushErr != nil {
						firstErr = pushErr
						break
					}
				}
				if firstErr != nil {
					_, _, _, repeatedErr := stream.Push([]byte("data: {}\n\n"))
					if goldenProtocolErrorFrom(t, firstErr) != goldenProtocolErrorFrom(t, repeatedErr) {
						t.Fatalf("poisoned stream changed its first failure: first=%v repeated=%v", firstErr, repeatedErr)
					}
				}
				finalFrames, _, _, finalErr := stream.Finalize(firstErr)
				wire = appendGoldenWireFrames(wire, finalFrames)
				if firstErr == nil {
					firstErr = finalErr
				} else if finalErr != nil && goldenProtocolErrorFrom(t, firstErr) != goldenProtocolErrorFrom(t, finalErr) {
					t.Fatalf("stream finalization changed its first failure: first=%v final=%v", firstErr, finalErr)
				}
				if firstErr == nil {
					t.Fatal("rejection stream unexpectedly finalized without an error")
				}
				goldenProtocolErrorFrom(t, firstErr)
				assertNoSuccessfulStreamTerminal(t, target.format, wire)
				if !bytes.Contains(wire, []byte("error")) {
					t.Fatalf("rejected stream has no public failure terminal: %s", wire)
				}
			})
		}
	}
}

func appendGoldenWireFrames(destination []byte, frames [][]byte) []byte {
	for _, frame := range frames {
		destination = append(destination, frame...)
	}
	return destination
}

func goldenProtocolErrorFrom(t *testing.T, err error) goldenProtocolError {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("returned %T, want *llmprotocol.ProtocolError: %v", err, err)
	}
	return goldenProtocolError{Category: protocolError.Category, Code: protocolError.Code}
}

func runRejectedTranslation(
	engine *Engine,
	source,
	target llmprotocol.WireFormat,
	input goldenRejectionInput,
) error {
	var err error
	engine, err = goldenRejectionEngine(engine, input)
	if err != nil {
		return err
	}
	body := input.Body
	if input.BodyJSON != "" {
		body = []byte(input.BodyJSON)
	}
	switch input.Operation {
	case "request":
		_, err := engine.TranslateRequest(source, target, body, func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		})
		return err
	case "response":
		_, err := engine.TranslateResponse(source, target, body, func(response *llmprotocol.Response) error {
			response.Model = "routed-model"
			return nil
		})
		return err
	case "error":
		_, err := engine.TranslateTransportError(source, target, body, nil)
		return err
	case "stream":
		stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: input.PublicModel, ProviderModel: input.ProviderModel,
		})
		if err != nil {
			return err
		}
		for _, chunk := range input.Chunks {
			if _, _, _, err := stream.Push([]byte(chunk)); err != nil {
				return err
			}
		}
		_, _, _, err = stream.Finalize(nil)
		return err
	default:
		return fmt.Errorf("unsupported rejection operation %q", input.Operation)
	}
}

func goldenRejectionEngine(base *Engine, input goldenRejectionInput) (*Engine, error) {
	limits := input.Limits
	if limits.BodyBytes == 0 && limits.JSONDepth == 0 && limits.SSEFrameBytes == 0 && limits.Events == 0 {
		return base, nil
	}
	policy := llmprotocol.DefaultPolicy()
	if limits.BodyBytes != 0 {
		policy.Limits.BodyBytes = limits.BodyBytes
	}
	if limits.JSONDepth != 0 {
		policy.Limits.JSONDepth = limits.JSONDepth
	}
	if limits.SSEFrameBytes != 0 {
		policy.Limits.SSEFrameBytes = limits.SSEFrameBytes
	}
	if limits.Events != 0 {
		policy.Limits.Events = limits.Events
	}
	return NewEngine(NewBuiltinRegistry(), policy)
}

func marshalGoldenStreamTranscript(t *testing.T, wireFrames [][]byte) []byte {
	t.Helper()
	framer := newSSEFramer(8 << 20)
	transcript := goldenStreamTranscript{Frames: []goldenStreamFrame{}}
	for _, wireFrame := range wireFrames {
		frames, err := framer.Push(wireFrame)
		if err != nil {
			t.Fatal(err)
		}
		for _, frame := range frames {
			parsed, err := parseSSEFrame(frame, 8<<20)
			if err != nil {
				t.Fatal(err)
			}
			if !parsed.HasData {
				continue
			}
			var data any
			if bytes.Equal(parsed.Data, []byte("[DONE]")) {
				data = "[DONE]"
			} else {
				decoder := json.NewDecoder(bytes.NewReader(parsed.Data))
				decoder.UseNumber()
				if err := decoder.Decode(&data); err != nil {
					t.Fatalf("stream output contains invalid JSON: %v\n%s", err, parsed.Data)
				}
			}
			transcript.Frames = append(transcript.Frames, goldenStreamFrame{Event: parsed.Event, Data: normalizeGoldenVolatileFields(data)})
		}
	}
	leftover, err := framer.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if len(leftover) != 0 {
		t.Fatalf("stream encoder left %d incomplete SSE frame(s)", len(leftover))
	}
	body, err := json.MarshalIndent(transcript, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(body, '\n')
}

func normalizeGoldenVolatileFields(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if key == "obfuscation" {
				typed[key] = "<random>"
				continue
			}
			typed[key] = normalizeGoldenVolatileFields(child)
		}
	case []any:
		for index := range typed {
			typed[index] = normalizeGoldenVolatileFields(typed[index])
		}
	}
	return value
}

type goldenTranslator func(
	engine *Engine,
	source,
	target llmprotocol.WireFormat,
	body []byte,
) ([]byte, error)

func runGoldenTranslationMatrix(t *testing.T, kind string, translate goldenTranslator) {
	t.Helper()
	directory := filepath.Join("testdata", "golden", kind)
	inputs, err := filepath.Glob(filepath.Join(directory, "*-in.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) == 0 {
		t.Fatalf("no %s protocol golden inputs found in %s", kind, directory)
	}
	engine := NewBuiltinEngine()
	for _, inputPath := range inputs {
		inputPath := inputPath
		prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
		source, err := goldenSourceFormat(prefix)
		if err != nil {
			t.Fatal(err)
		}
		body, err := os.ReadFile(inputPath)
		if err != nil {
			t.Fatal(err)
		}
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				actual, err := translate(engine, source, target.format, body)
				if err != nil {
					t.Fatal(err)
				}
				expectedPath := filepath.Join(directory, prefix+"-"+target.name+"-out.json")
				if os.Getenv(updateProtocolGoldensEnv) == "1" {
					if err := os.WriteFile(expectedPath, prettyJSON(t, actual), 0o644); err != nil {
						t.Fatal(err)
					}
				}
				expected, err := os.ReadFile(expectedPath)
				if err != nil {
					t.Fatalf("missing golden output %s: %v", expectedPath, err)
				}
				if !jsonEquivalent(expected, actual) {
					t.Fatalf("protocol translation drifted\nexpected (%s):\n%s\nactual:\n%s", expectedPath, prettyJSON(t, expected), prettyJSON(t, actual))
				}
			})
		}
	}
}

func goldenSourceFormat(prefix string) (llmprotocol.WireFormat, error) {
	parts := strings.Split(prefix, "-")
	if len(parts) < 3 {
		return "", fmt.Errorf("golden input %q must be named NNN-{chat|responses|anthropic}-description-in.json", prefix)
	}
	for _, format := range goldenFormats {
		if parts[1] == format.name {
			return format.format, nil
		}
	}
	return "", fmt.Errorf("golden input %q has unknown source protocol %q", prefix, parts[1])
}

func jsonEquivalent(expected, actual []byte) bool {
	var expectedValue, actualValue any
	expectedDecoder := json.NewDecoder(bytes.NewReader(expected))
	expectedDecoder.UseNumber()
	actualDecoder := json.NewDecoder(bytes.NewReader(actual))
	actualDecoder.UseNumber()
	return expectedDecoder.Decode(&expectedValue) == nil && actualDecoder.Decode(&actualValue) == nil &&
		reflect.DeepEqual(expectedValue, actualValue)
}

func prettyJSON(t *testing.T, body []byte) []byte {
	t.Helper()
	var value any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		t.Fatalf("invalid JSON fixture: %v\n%s", err, body)
	}
	formatted, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(formatted, '\n')
}
