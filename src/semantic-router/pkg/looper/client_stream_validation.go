package looper

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func decodeModelStream(body []byte, modelName string) ([]llmprotocol.Event, error) {
	decoder := (protocolcodec.OpenAIChatCodec{}).NewDecoder(llmprotocol.StreamContext{
		Source: llmprotocol.OpenAIChatV1, Target: llmprotocol.OpenAIChatV1,
		PublicModel: modelName,
	}, llmprotocol.DefaultPolicy())
	events, _, err := decoder.Push(body)
	if err != nil {
		return nil, fmt.Errorf("invalid model stream: %w", err)
	}
	terminal, _, err := decoder.Finalize(nil)
	if err != nil {
		return nil, fmt.Errorf("incomplete model stream: %w", err)
	}
	events = append(events, terminal...)
	completed := false
	for _, event := range events {
		switch event.Type {
		case llmprotocol.EventResponseFailed:
			if event.Error != nil {
				return nil, fmt.Errorf("model stream failed: %w", event.Error)
			}
			return nil, fmt.Errorf("model stream failed")
		case llmprotocol.EventResponseCompleted:
			completed = true
		}
	}
	if !completed {
		return nil, fmt.Errorf("model stream ended without completion")
	}
	return events, nil
}
