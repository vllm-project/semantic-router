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
		runGoldenRejectionInput(t, engine, directory, inputPath)
	}
}

func runGoldenRejectionInput(t *testing.T, engine *Engine, directory, inputPath string) {
	t.Helper()
	prefix := strings.TrimSuffix(filepath.Base(inputPath), "-in.json")
	source, input := loadGoldenRejectionInput(t, inputPath, prefix)
	for _, target := range goldenFormats {
		t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
			translateErr := runRejectedTranslation(engine, source, target.format, input)
			if translateErr == nil {
				t.Fatalf("rejection case %s unexpectedly succeeded", inputPath)
			}
			protocolError := goldenProtocolErrorFrom(t, translateErr)
			actual, err := json.MarshalIndent(protocolError, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			actual = append(actual, '\n')
			assertGoldenOutput(t, filepath.Join(directory, prefix+"-"+target.name+"-out.json"), actual, "rejection contract")
		})
	}
}

func loadGoldenRejectionInput(t *testing.T, inputPath, prefix string) (llmprotocol.WireFormat, goldenRejectionInput) {
	t.Helper()
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
	return source, input
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
		source, input := loadGoldenRejectionInput(t, inputPath, prefix)
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
		source, input := loadGoldenRejectionInput(t, inputPath, prefix)
		if input.Operation != "stream" {
			continue
		}
		inputEngine, err := goldenRejectionEngine(engine, input)
		if err != nil {
			t.Fatal(err)
		}
		for _, target := range goldenFormats {
			t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
				assertGoldenRejectedStreamFailsClosed(t, inputEngine, source, target.format, input)
			})
		}
	}
}

func assertGoldenRejectedStreamFailsClosed(
	t *testing.T,
	engine *Engine,
	source, target llmprotocol.WireFormat,
	input goldenRejectionInput,
) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: input.PublicModel, ProviderModel: input.ProviderModel,
	})
	if err != nil {
		t.Fatal(err)
	}
	wire, firstErr := pushGoldenRejectedChunks(stream, input.Chunks)
	if firstErr != nil {
		assertGoldenPoisonedError(t, stream, firstErr)
	}
	finalFrames, _, _, finalErr := stream.Finalize(firstErr)
	wire = appendGoldenWireFrames(wire, finalFrames)
	firstErr = assertGoldenFinalErrorStable(t, firstErr, finalErr)
	goldenProtocolErrorFrom(t, firstErr)
	assertNoSuccessfulStreamTerminal(t, target, wire)
	if !bytes.Contains(wire, []byte("error")) {
		t.Fatalf("rejected stream has no public failure terminal: %s", wire)
	}
}

func pushGoldenRejectedChunks(stream *StreamEngine, chunks []string) ([]byte, error) {
	var wire []byte
	for _, chunk := range chunks {
		frames, _, _, err := stream.Push([]byte(chunk))
		wire = appendGoldenWireFrames(wire, frames)
		if err != nil {
			return wire, err
		}
	}
	return wire, nil
}

func assertGoldenPoisonedError(t *testing.T, stream *StreamEngine, firstErr error) {
	t.Helper()
	_, _, _, repeatedErr := stream.Push([]byte("data: {}\n\n"))
	if goldenProtocolErrorFrom(t, firstErr) != goldenProtocolErrorFrom(t, repeatedErr) {
		t.Fatalf("poisoned stream changed its first failure: first=%v repeated=%v", firstErr, repeatedErr)
	}
}

func assertGoldenFinalErrorStable(t *testing.T, firstErr, finalErr error) error {
	t.Helper()
	if firstErr == nil {
		firstErr = finalErr
	} else if finalErr != nil && goldenProtocolErrorFrom(t, firstErr) != goldenProtocolErrorFrom(t, finalErr) {
		t.Fatalf("stream finalization changed its first failure: first=%v final=%v", firstErr, finalErr)
	}
	if firstErr == nil {
		t.Fatal("rejection stream unexpectedly finalized without an error")
	}
	return firstErr
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
			appendGoldenTranscriptFrame(t, &transcript, frame)
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

func appendGoldenTranscriptFrame(t *testing.T, transcript *goldenStreamTranscript, frame []byte) {
	t.Helper()
	parsed, err := parseSSEFrame(frame, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	if !parsed.HasData {
		return
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
		runGoldenTranslationInput(t, engine, directory, inputPath, translate)
	}
}

func runGoldenTranslationInput(
	t *testing.T,
	engine *Engine,
	directory, inputPath string,
	translate goldenTranslator,
) {
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
	for _, target := range goldenFormats {
		t.Run(prefix+"_to_"+target.name, func(t *testing.T) {
			actual, err := translate(engine, source, target.format, body)
			if err != nil {
				t.Fatal(err)
			}
			actual = prettyJSON(t, actual)
			assertGoldenOutput(t, filepath.Join(directory, prefix+"-"+target.name+"-out.json"), actual, "protocol translation")
		})
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

func assertGoldenOutput(t *testing.T, expectedPath string, actual []byte, label string) {
	t.Helper()
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
		t.Fatalf("%s drifted\nexpected (%s):\n%s\nactual:\n%s", label, expectedPath, prettyJSON(t, expected), prettyJSON(t, actual))
	}
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
