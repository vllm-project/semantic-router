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
	runConcurrentProtocolMatrix(t, 8, func(source, target llmprotocol.WireFormat, iteration int) error {
		return assertConcurrentRequest(engine, source, target, iteration)
	})
}

func TestEngineConcurrentResponseMatrixIsRequestScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	runConcurrentProtocolMatrix(t, 8, func(source, target llmprotocol.WireFormat, iteration int) error {
		return assertConcurrentResponse(engine, source, target, iteration)
	})
}

func TestEngineConcurrentTransportErrorMatrixIsRequestScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	runConcurrentProtocolMatrix(t, 8, func(source, target llmprotocol.WireFormat, iteration int) error {
		return assertConcurrentTransportError(engine, source, target, iteration)
	})
}

func TestEngineConcurrentStreamMatrixIsRequestScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	runConcurrentProtocolMatrix(t, 4, func(source, target llmprotocol.WireFormat, iteration int) error {
		return assertConcurrentStream(engine, source, target, iteration)
	})
}

func runConcurrentProtocolMatrix(
	t *testing.T,
	iterations int,
	run func(llmprotocol.WireFormat, llmprotocol.WireFormat, int) error,
) {
	t.Helper()
	var wait sync.WaitGroup
	errors := make(chan error, len(builtinFormats)*len(builtinFormats)*iterations)
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for iteration := 0; iteration < iterations; iteration++ {
				source, target, iteration := source, target, iteration
				wait.Add(1)
				go func() {
					defer wait.Done()
					if err := run(source, target, iteration); err != nil {
						errors <- err
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

func assertConcurrentRequest(engine *Engine, source, target llmprotocol.WireFormat, iteration int) error {
	model := fmt.Sprintf("routed-%s-%s-%d", source, target, iteration)
	translated, err := engine.TranslateRequest(source, target, requestFixture(source), func(request *llmprotocol.Request) error {
		request.Model = model
		return nil
	})
	if err != nil {
		return fmt.Errorf("%s to %s iteration %d: %w", source, target, iteration, err)
	}
	decoded, _, _, err := engine.DecodeRequest(target, translated.Body)
	if err != nil {
		return fmt.Errorf("decode %s to %s iteration %d: %w", source, target, iteration, err)
	}
	if decoded.Model != model {
		return fmt.Errorf("%s to %s iteration %d model = %q, want %q", source, target, iteration, decoded.Model, model)
	}
	return nil
}

func assertConcurrentResponse(engine *Engine, source, target llmprotocol.WireFormat, iteration int) error {
	model := fmt.Sprintf("routed-%s-%s-%d", source, target, iteration)
	translated, err := engine.TranslateResponse(source, target, responseFixture(source), func(response *llmprotocol.Response) error {
		response.Model = model
		return nil
	})
	if err != nil {
		return fmt.Errorf("%s to %s iteration %d: %w", source, target, iteration, err)
	}
	decoded, _, _, err := engine.DecodeResponse(target, translated.Body)
	if err != nil {
		return fmt.Errorf("decode %s to %s iteration %d: %w", source, target, iteration, err)
	}
	if decoded.Model != model {
		return fmt.Errorf("%s to %s iteration %d model = %q, want %q", source, target, iteration, decoded.Model, model)
	}
	return nil
}

func assertConcurrentTransportError(engine *Engine, source, target llmprotocol.WireFormat, iteration int) error {
	message := fmt.Sprintf("failure-%s-%s-%d", source, target, iteration)
	translated, err := engine.TranslateTransportError(source, target, transportErrorFixture(source), func(transportError *llmprotocol.TransportError) error {
		transportError.Error.Message = message
		return nil
	})
	if err != nil {
		return fmt.Errorf("%s to %s iteration %d: %w", source, target, iteration, err)
	}
	decoded, err := engine.TranslateTransportError(target, target, translated.Body, nil)
	if err != nil {
		return fmt.Errorf("decode %s to %s iteration %d: %w", source, target, iteration, err)
	}
	if decoded.TransportError.Error == nil || decoded.TransportError.Error.Message != message {
		return fmt.Errorf("%s to %s iteration %d message = %+v, want %q", source, target, iteration, decoded.TransportError.Error, message)
	}
	return nil
}

func assertConcurrentStream(engine *Engine, source, target llmprotocol.WireFormat, iteration int) error {
	model := fmt.Sprintf("public-%s-%s-%d", source, target, iteration)
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: model, ProviderModel: "provider-model"})
	if err != nil {
		return fmt.Errorf("new stream %s to %s iteration %d: %w", source, target, iteration, err)
	}
	frames, _, _, err := stream.Push(streamFixture(source))
	if err != nil {
		return fmt.Errorf("push stream %s to %s iteration %d: %w", source, target, iteration, err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		return fmt.Errorf("finalize stream %s to %s iteration %d: %w", source, target, iteration, err)
	}
	frames = append(frames, finalFrames...)
	decoded, _, err := engine.DecodeResponseStream(target, bytes.Join(frames, nil), llmprotocol.StreamContext{Context: context.Background(), PublicModel: model, ProviderModel: model})
	if err != nil {
		return fmt.Errorf("decode stream %s to %s iteration %d: %w", source, target, iteration, err)
	}
	if decoded.Model != model {
		return fmt.Errorf("stream %s to %s iteration %d model = %q, want %q", source, target, iteration, decoded.Model, model)
	}
	return nil
}
