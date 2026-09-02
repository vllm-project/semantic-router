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
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
		runGoldenCapabilityInput(t, engine, directory, inputPath)
	}
}

func runGoldenCapabilityInput(t *testing.T, engine *Engine, directory, inputPath string) {
	t.Helper()
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
			assertGoldenCapabilityTarget(t, engine, directory, prefix, source, target, input)
		})
	}
}

func assertGoldenCapabilityTarget(
	t *testing.T,
	engine *Engine,
	directory, prefix string,
	source llmprotocol.WireFormat,
	target goldenFormat,
	input goldenCapabilityInput,
) {
	t.Helper()
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
	assertGoldenOutput(t, filepath.Join(directory, prefix+"-"+target.name+"-out.json"), actual, "capability outcome")
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
		validateGoldenCapabilityCase(t, path, input.Operation, input.Base, testCase)
	}
}

func validateGoldenCapabilityCase(
	t *testing.T,
	path, operation string,
	base json.RawMessage,
	testCase goldenCapabilityCase,
) {
	t.Helper()
	hasBody := len(testCase.Body) != 0
	hasPatch := len(testCase.Patch) != 0
	hasStream := testCase.Stream != nil
	if hasBody && hasPatch {
		t.Fatalf("capability fixture %s case %q cannot combine body and patch", path, testCase.Name)
	}
	if operation == "stream" {
		if !hasStream || hasBody || hasPatch {
			t.Fatalf("stream capability fixture %s case %q must contain exactly one stream", path, testCase.Name)
		}
		return
	}
	if hasStream || (!hasBody && !hasPatch) {
		t.Fatalf("%s capability fixture %s case %q must contain exactly one body or patch", operation, path, testCase.Name)
	}
	if hasPatch && len(base) == 0 {
		t.Fatalf("capability fixture %s case %q uses a patch without a base", path, testCase.Name)
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
		return runCapabilityRequestTranslation(engine, source, target, input.Body)
	case "response":
		return runCapabilityResponseTranslation(engine, source, target, input.Body)
	case "stream":
		return runCapabilityStreamTranslation(t, engine, source, target, input.Stream)
	default:
		return nil, nil, fmt.Errorf("unsupported capability operation %q", input.Operation)
	}
}

func runCapabilityRequestTranslation(
	engine *Engine,
	source, target llmprotocol.WireFormat,
	body []byte,
) ([]byte, llmprotocol.Diagnostics, error) {
	result, err := engine.TranslateRequest(source, target, body, func(request *llmprotocol.Request) error {
		request.Model = "routed-model"
		return nil
	})
	if err == nil {
		if _, _, _, decodeErr := engine.DecodeRequest(target, result.Body); decodeErr != nil {
			return nil, result.Diagnostics, fmt.Errorf("target request does not satisfy its own codec contract: %w\nencoded body: %s", decodeErr, result.Body)
		}
	}
	return result.Body, result.Diagnostics, err
}

func runCapabilityResponseTranslation(
	engine *Engine,
	source, target llmprotocol.WireFormat,
	body []byte,
) ([]byte, llmprotocol.Diagnostics, error) {
	result, err := engine.TranslateResponse(source, target, body, func(response *llmprotocol.Response) error {
		response.Model = "routed-model"
		return nil
	})
	if err == nil {
		if decodeErr := validateGoldenEncodedResponse(engine, target, result); decodeErr != nil {
			return nil, result.Diagnostics, fmt.Errorf("target response does not satisfy its own codec contract: %w\nencoded body: %s", decodeErr, result.Body)
		}
	}
	return result.Body, result.Diagnostics, err
}

func runCapabilityStreamTranslation(
	t *testing.T,
	engine *Engine,
	source, target llmprotocol.WireFormat,
	input *goldenStreamInput,
) ([]byte, llmprotocol.Diagnostics, error) {
	if input == nil || input.PublicModel == "" || input.ProviderModel == "" || len(input.Chunks) == 0 {
		return nil, nil, fmt.Errorf("stream capability case requires public_model, provider_model, and chunks")
	}
	frames, diagnostics, err := translateGoldenCapabilityStream(engine, source, target, *input)
	if err != nil {
		return nil, diagnostics, err
	}
	verifyGoldenTargetStream(t, engine, target, *input, frames)
	return marshalGoldenStreamTranscript(t, frames), diagnostics, nil
}

func validateGoldenEncodedResponse(
	engine *Engine,
	target llmprotocol.WireFormat,
	result ResponseResult,
) error {
	// Responses can represent a failed model response as a response resource.
	// Chat Completions and Messages expose the same failure as an HTTP error
	// envelope, so validate those bodies against the transport-error contract.
	if result.Response.Error != nil && target != llmprotocol.OpenAIResponsesV1 {
		_, _, err := engine.DecodeTransportError(target, result.Body)
		return err
	}
	_, _, _, err := engine.DecodeResponse(target, result.Body)
	return err
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
