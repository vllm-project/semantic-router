package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func FuzzTranslateBuiltinRequestMatrixNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(requestFixture(format))
	}
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte) {
		for _, source := range builtinFormats {
			for _, target := range builtinFormats {
				_, _ = engine.TranslateRequest(source, target, body, nil)
			}
		}
	})
}

func FuzzTranslateBuiltinResponseMatrixNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(responseFixture(format))
	}
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte) {
		for _, source := range builtinFormats {
			for _, target := range builtinFormats {
				_, _ = engine.TranslateResponse(source, target, body, nil)
			}
		}
	})
}

func FuzzBuiltinStreamMatrixNeverPanics(f *testing.F) {
	for _, format := range builtinFormats {
		f.Add(streamFixture(format))
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
				_, _, _, _ = stream.Push(body)
				_, _, _, _ = stream.Finalize(nil)
			}
		}
	})
}
