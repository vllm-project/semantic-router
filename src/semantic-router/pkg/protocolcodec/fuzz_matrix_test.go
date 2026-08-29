package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func FuzzTranslateBuiltinRequestMatrixNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(requestFixture(format))
	}
	addGoldenBodiesAsFuzzSeeds(f, "request")
	addGoldenCapabilityBodiesAsFuzzSeeds(f, "request")
	addGoldenRejectionBodiesAsFuzzSeeds(f, "request")
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte) {
		for _, source := range builtinFormats {
			for _, target := range builtinFormats {
				translated, err := engine.TranslateRequest(source, target, body, nil)
				if err != nil {
					continue
				}
				if _, _, _, err := engine.DecodeRequest(target, translated.Body); err != nil {
					t.Fatalf("accepted %s to %s request produced invalid target wire: %v\n%s", source, target, err, translated.Body)
				}
			}
		}
	})
}

func FuzzTranslateBuiltinResponseMatrixNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(responseFixture(format))
	}
	addGoldenBodiesAsFuzzSeeds(f, "response")
	addGoldenCapabilityBodiesAsFuzzSeeds(f, "response")
	addGoldenRejectionBodiesAsFuzzSeeds(f, "response")
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte) {
		for _, source := range builtinFormats {
			for _, target := range builtinFormats {
				translated, err := engine.TranslateResponse(source, target, body, nil)
				if err != nil {
					continue
				}
				if err := validateGoldenEncodedResponse(engine, target, translated); err != nil {
					t.Fatalf("accepted %s to %s response produced invalid target wire: %v\n%s", source, target, err, translated.Body)
				}
			}
		}
	})
}

func FuzzTranslateBuiltinTransportErrorMatrixNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(transportErrorFixture(format))
	}
	addGoldenBodiesAsFuzzSeeds(f, "error")
	addGoldenRejectionBodiesAsFuzzSeeds(f, "error")
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte) {
		for _, source := range builtinFormats {
			for _, target := range builtinFormats {
				translated, err := engine.TranslateTransportError(source, target, body, nil)
				if err != nil {
					continue
				}
				if _, err := engine.TranslateTransportError(target, target, translated.Body, nil); err != nil {
					t.Fatalf("accepted %s to %s transport error produced invalid target wire: %v\n%s", source, target, err, translated.Body)
				}
			}
		}
	})
}

func FuzzBuiltinStreamMatrixNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(streamFixture(format))
	}
	addGoldenStreamsAsFuzzSeeds(f)
	addGoldenCapabilityBodiesAsFuzzSeeds(f, "stream")
	for _, body := range goldenRejectionSeeds(f, "stream") {
		f.Add(body)
	}
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte) {
		for _, source := range builtinFormats {
			for _, target := range builtinFormats {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model",
				})
				if err != nil {
					continue
				}
				frames, _, _, pushErr := stream.Push(body)
				finalFrames, _, _, finalErr := stream.Finalize(nil)
				frames = append(frames, finalFrames...)
				if pushErr == nil && finalErr == nil {
					assertFuzzTargetStreamDecodes(t, engine, target, frames)
				}
			}
		}
	})
}

func FuzzBuiltinStreamMatrixArbitraryChunkingNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(streamFixture(format), uint8(1))
		f.Add(streamFixture(format), uint8(31))
	}
	for _, body := range goldenRejectionSeeds(f, "stream") {
		f.Add(body, uint8(1))
		f.Add(body, uint8(31))
	}
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte, strideSeed uint8) {
		stride := int(strideSeed) + 1
		for _, source := range builtinFormats {
			for _, target := range builtinFormats {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model",
				})
				if err != nil {
					continue
				}
				var frames [][]byte
				pushFailed := false
				for offset := 0; offset < len(body); {
					end := offset + stride
					if end > len(body) {
						end = len(body)
					}
					output, _, _, err := stream.Push(body[offset:end])
					frames = append(frames, output...)
					if err != nil {
						pushFailed = true
						break
					}
					offset = end
				}
				finalFrames, _, _, finalErr := stream.Finalize(nil)
				frames = append(frames, finalFrames...)
				if !pushFailed && finalErr == nil {
					assertFuzzTargetStreamDecodes(t, engine, target, frames)
				}
			}
		}
	})
}

func assertFuzzTargetStreamDecodes(
	t *testing.T,
	engine *Engine,
	target llmprotocol.WireFormat,
	frames [][]byte,
) {
	t.Helper()
	verify, err := engine.NewStream(target, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := verify.Push(bytes.Join(frames, nil)); err != nil {
		t.Fatalf("accepted stream produced invalid %s wire: %v\n%s", target, err, bytes.Join(frames, nil))
	}
	if _, _, _, err := verify.Finalize(nil); err != nil {
		t.Fatalf("accepted stream produced incomplete %s wire: %v\n%s", target, err, bytes.Join(frames, nil))
	}
}

func addGoldenBodiesAsFuzzSeeds(f *testing.F, kind string) {
	f.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "golden", kind, "*-in.json"))
	if err != nil {
		f.Fatal(err)
	}
	for _, path := range paths {
		body, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}
		f.Add(body)
	}
}

func addGoldenStreamsAsFuzzSeeds(f *testing.F) {
	f.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "golden", "stream", "*-in.json"))
	if err != nil {
		f.Fatal(err)
	}
	for _, path := range paths {
		body, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}
		var input goldenStreamInput
		if err := json.Unmarshal(body, &input); err != nil {
			f.Fatalf("decode stream fuzz seed %s: %v", path, err)
		}
		f.Add([]byte(strings.Join(input.Chunks, "")))
	}
}

func addGoldenRejectionBodiesAsFuzzSeeds(f *testing.F, operation string) {
	f.Helper()
	for _, body := range goldenRejectionSeeds(f, operation) {
		f.Add(body)
	}
}

func addGoldenCapabilityBodiesAsFuzzSeeds(f *testing.F, operation string) {
	f.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "golden", "capability", "*-in.json"))
	if err != nil {
		f.Fatal(err)
	}
	for _, path := range paths {
		descriptor, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}
		var input goldenCapabilityInput
		if err := json.Unmarshal(descriptor, &input); err != nil {
			f.Fatalf("decode capability fuzz seed %s: %v", path, err)
		}
		if input.Operation != operation {
			continue
		}
		if operation == "stream" && input.Stream != nil {
			f.Add([]byte(strings.Join(input.Stream.Chunks, "")))
			continue
		}
		if len(input.Cases) == 0 {
			if len(input.Body) > 0 {
				f.Add([]byte(input.Body))
			}
			continue
		}
		for _, testCase := range input.Cases {
			if operation == "stream" && testCase.Stream != nil {
				f.Add([]byte(strings.Join(testCase.Stream.Chunks, "")))
				continue
			}
			body := testCase.Body
			if len(body) == 0 {
				body = capabilityFuzzSeedPatch(f, path, input.Base, testCase.Patch)
			}
			f.Add([]byte(body))
		}
	}
}

func capabilityFuzzSeedPatch(
	f *testing.F,
	path string,
	base json.RawMessage,
	patch map[string]json.RawMessage,
) json.RawMessage {
	f.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(base, &object); err != nil {
		f.Fatalf("decode capability base %s: %v", path, err)
	}
	for field, value := range patch {
		object[field] = append(json.RawMessage(nil), value...)
	}
	body, err := json.Marshal(object)
	if err != nil {
		f.Fatalf("encode capability fuzz seed %s: %v", path, err)
	}
	return body
}

func goldenRejectionSeeds(f *testing.F, operation string) [][]byte {
	f.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "golden", "rejection", "*-in.json"))
	if err != nil {
		f.Fatal(err)
	}
	var seeds [][]byte
	for _, path := range paths {
		descriptor, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}
		var input goldenRejectionInput
		if err := json.Unmarshal(descriptor, &input); err != nil {
			f.Fatalf("decode rejection fuzz seed %s: %v", path, err)
		}
		if input.Operation != operation {
			continue
		}
		body := input.Body
		if input.BodyJSON != "" {
			body = []byte(input.BodyJSON)
		}
		if operation == "stream" {
			body = []byte(strings.Join(input.Chunks, ""))
		}
		if len(body) > 0 {
			seeds = append(seeds, append([]byte(nil), body...))
		}
	}
	return seeds
}

func transportErrorFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1:
		return []byte(`{"error":{"message":"rate limited","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`)
	default:
		return nil
	}
}
