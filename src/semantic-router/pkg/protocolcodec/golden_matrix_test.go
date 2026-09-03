package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
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

type goldenFormat struct {
	name   string
	format llmprotocol.WireFormat
}

var goldenFormats = []goldenFormat{
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
				return nil, fmt.Errorf(
					"target request does not satisfy its own codec contract: %w\nencoded body: %s",
					decodeErr,
					result.Body,
				)
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
			decodeErr := validateGoldenEncodedResponse(engine, target, result)
			if decodeErr != nil {
				return nil, fmt.Errorf(
					"target response does not satisfy its own codec contract: %w\nencoded body: %s",
					decodeErr,
					result.Body,
				)
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
		runGoldenStreamInput(t, engine, directory, inputPath)
	}
}

func runGoldenStreamInput(t *testing.T, engine *Engine, directory, inputPath string) {
	t.Helper()
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
			assertGoldenStreamTarget(t, engine, directory, prefix, source, target, input)
		})
	}
}

func assertGoldenStreamTarget(
	t *testing.T,
	engine *Engine,
	directory, prefix string,
	source llmprotocol.WireFormat,
	target goldenFormat,
	input goldenStreamInput,
) {
	t.Helper()
	outputFrames := translateGoldenStream(t, engine, source, target.format, input, input.Chunks)
	verifyGoldenTargetStream(t, engine, target.format, input, outputFrames)
	actual := marshalGoldenStreamTranscript(t, outputFrames)
	assertGoldenOutput(t, filepath.Join(directory, prefix+"-"+target.name+"-out.json"), actual, "stream protocol translation")
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
