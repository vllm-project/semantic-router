package protocolcodec

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestEngineConcurrentBufferedMatrixIsRequestScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	var wait sync.WaitGroup
	errors := make(chan error, len(builtinFormats)*len(builtinFormats)*8)
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for iteration := 0; iteration < 8; iteration++ {
				source, target, iteration := source, target, iteration
				wait.Add(1)
				go func() {
					defer wait.Done()
					model := fmt.Sprintf("routed-%s-%s-%d", source, target, iteration)
					translated, err := engine.TranslateRequest(source, target, requestFixture(source), func(request *llmprotocol.Request) error {
						request.Model = model
						return nil
					})
					if err != nil {
						errors <- fmt.Errorf("%s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					decoded, _, _, err := engine.DecodeRequest(target, translated.Body)
					if err != nil {
						errors <- fmt.Errorf("decode %s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					if decoded.Model != model {
						errors <- fmt.Errorf("%s to %s iteration %d model = %q, want %q", source, target, iteration, decoded.Model, model)
					}
				}()
			}
		}
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

func TestEngineConcurrentResponseMatrixIsRequestScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	var wait sync.WaitGroup
	errors := make(chan error, len(builtinFormats)*len(builtinFormats)*8)
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for iteration := 0; iteration < 8; iteration++ {
				source, target, iteration := source, target, iteration
				wait.Add(1)
				go func() {
					defer wait.Done()
					model := fmt.Sprintf("routed-%s-%s-%d", source, target, iteration)
					translated, err := engine.TranslateResponse(source, target, responseFixture(source), func(response *llmprotocol.Response) error {
						response.Model = model
						return nil
					})
					if err != nil {
						errors <- fmt.Errorf("%s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					decoded, _, _, err := engine.DecodeResponse(target, translated.Body)
					if err != nil {
						errors <- fmt.Errorf("decode %s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					if decoded.Model != model {
						errors <- fmt.Errorf("%s to %s iteration %d model = %q, want %q", source, target, iteration, decoded.Model, model)
					}
				}()
			}
		}
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

func TestEngineConcurrentTransportErrorMatrixIsRequestScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	var wait sync.WaitGroup
	errors := make(chan error, len(builtinFormats)*len(builtinFormats)*8)
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for iteration := 0; iteration < 8; iteration++ {
				source, target, iteration := source, target, iteration
				wait.Add(1)
				go func() {
					defer wait.Done()
					message := fmt.Sprintf("failure-%s-%s-%d", source, target, iteration)
					translated, err := engine.TranslateTransportError(source, target, transportErrorFixture(source), func(transportError *llmprotocol.TransportError) error {
						transportError.Error.Message = message
						return nil
					})
					if err != nil {
						errors <- fmt.Errorf("%s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					decoded, err := engine.TranslateTransportError(target, target, translated.Body, nil)
					if err != nil {
						errors <- fmt.Errorf("decode %s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					if decoded.TransportError.Error == nil || decoded.TransportError.Error.Message != message {
						errors <- fmt.Errorf("%s to %s iteration %d message = %+v, want %q", source, target, iteration, decoded.TransportError.Error, message)
					}
				}()
			}
		}
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

func TestEngineConcurrentStreamMatrixIsRequestScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	var wait sync.WaitGroup
	errors := make(chan error, len(builtinFormats)*len(builtinFormats)*4)
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for iteration := 0; iteration < 4; iteration++ {
				source, target, iteration := source, target, iteration
				wait.Add(1)
				go func() {
					defer wait.Done()
					model := fmt.Sprintf("public-%s-%s-%d", source, target, iteration)
					stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
						Context: context.Background(), PublicModel: model, ProviderModel: "provider-model",
					})
					if err != nil {
						errors <- fmt.Errorf("new stream %s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					frames, _, _, err := stream.Push(streamFixture(source))
					if err != nil {
						errors <- fmt.Errorf("push stream %s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					finalFrames, _, _, err := stream.Finalize(nil)
					if err != nil {
						errors <- fmt.Errorf("finalize stream %s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					frames = append(frames, finalFrames...)
					decoded, _, err := engine.DecodeResponseStream(target, bytes.Join(frames, nil), llmprotocol.StreamContext{
						Context: context.Background(), PublicModel: model, ProviderModel: model,
					})
					if err != nil {
						errors <- fmt.Errorf("decode stream %s to %s iteration %d: %w", source, target, iteration, err)
						return
					}
					if decoded.Model != model {
						errors <- fmt.Errorf("stream %s to %s iteration %d model = %q, want %q", source, target, iteration, decoded.Model, model)
					}
				}()
			}
		}
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}
