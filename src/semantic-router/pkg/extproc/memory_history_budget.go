package extproc

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// These limits apply to persistence preparation, independently of ingress body
// limits. Oversized snapshots are rejected intact, preserving session stride.
const (
	maxMemorySnapshotMessages = 256
	maxMemorySnapshotBytes    = 1 << 20
	maxMemorySnapshotNodes    = 4096
	maxMemorySnapshotDepth    = 32
)

var errMemoryHistoryTooLarge = errors.New("memory history exceeds snapshot budget")

type memoryHistoryBudget struct {
	bytes, nodes, messages int
}

func takeMemoryBudget(remaining *int, count int) bool {
	if count > *remaining {
		return false
	}
	*remaining -= count
	return true
}

func (b *memoryHistoryBudget) strings(values ...string) bool {
	for _, value := range values {
		if !takeMemoryBudget(&b.bytes, len(value)) {
			return false
		}
	}
	return true
}

// validateMemorySnapshotBudget walks only bounded structure and string lengths
// before admission, cloning, JSON decoding, text joining, or think-tag stripping.
func validateMemorySnapshotBudget(jobCtx context.Context, messages []llmprotocol.Message, retained []*responseapi.StoredResponse, response *llmprotocol.Response) error {
	b := memoryHistoryBudget{maxMemorySnapshotBytes, maxMemorySnapshotNodes, maxMemorySnapshotMessages}
	if !b.assistantResponse(response) {
		return errMemoryHistoryTooLarge
	}
	if !takeMemoryBudget(&b.messages, len(messages)) {
		return errMemoryHistoryTooLarge
	}
	for _, message := range messages {
		if err := jobCtx.Err(); err != nil {
			return err
		}
		if !b.strings(message.ID, string(message.Role)) || !b.contents(message.Content, 0) {
			return errMemoryHistoryTooLarge
		}
	}
	if len(retained) > 0 {
		if !takeMemoryBudget(&b.nodes, len(retained)) {
			return errMemoryHistoryTooLarge
		}
		for _, stored := range retained {
			if err := jobCtx.Err(); err != nil {
				return err
			}
			if stored == nil {
				continue
			}
			if !takeMemoryBudget(&b.messages, len(stored.Input)) ||
				!takeMemoryBudget(&b.messages, len(stored.Output)) || !b.strings(stored.OutputText) {
				return errMemoryHistoryTooLarge
			}
			// OutputText produces a message even when Output is absent.
			if stored.OutputText != "" && len(stored.Output) == 0 && !takeMemoryBudget(&b.messages, 1) {
				return errMemoryHistoryTooLarge
			}
			for _, input := range stored.Input {
				if !takeMemoryBudget(&b.bytes, len(input.Content)) || !b.strings(input.Role) {
					return errMemoryHistoryTooLarge
				}
			}
			for _, output := range stored.Output {
				if !takeMemoryBudget(&b.nodes, len(output.Content)) || !b.strings(output.Role) {
					return errMemoryHistoryTooLarge
				}
				for _, part := range output.Content {
					if !b.strings(part.Text) {
						return errMemoryHistoryTooLarge
					}
				}
			}
		}
	}
	return jobCtx.Err()
}

// Count the same primary assistant text consumed by semanticAssistantContent.
// Charge raw bytes so stripping think tags cannot hide unbounded preparation.
func (b *memoryHistoryBudget) assistantResponse(response *llmprotocol.Response) bool {
	if response == nil {
		return true
	}
	if !takeMemoryBudget(&b.nodes, len(response.Output)) {
		return false
	}
	for _, item := range response.Output {
		if item.Role != llmprotocol.RoleAssistant {
			continue
		}
		if !takeMemoryBudget(&b.nodes, len(item.Content)) {
			return false
		}
		for _, content := range item.Content {
			if (content.Kind == llmprotocol.ContentText || content.Kind == llmprotocol.ContentRefusal) && !b.strings(content.Text) {
				return false
			}
		}
	}
	return true
}

func (b *memoryHistoryBudget) contents(contents []llmprotocol.Content, depth int) bool {
	if depth > maxMemorySnapshotDepth || !takeMemoryBudget(&b.nodes, len(contents)) {
		return false
	}
	for _, c := range contents {
		if !b.strings(string(c.Kind), c.Text, c.MediaType, c.URL, c.Data, c.FileID,
			c.Filename, c.Detail, c.Signature, string(c.Reasoning)) || !takeMemoryBudget(&b.nodes, len(c.Citations)) {
			return false
		}
		for _, citation := range c.Citations {
			if !b.strings(citation.URL, citation.Title) {
				return false
			}
		}
		if c.Cache != nil && !b.strings(c.Cache.Type, c.Cache.TTL) {
			return false
		}
		if c.ToolCall != nil && !b.strings(c.ToolCall.ID, c.ToolCall.Name, c.ToolCall.Arguments) {
			return false
		}
		if c.ToolResult != nil && (!b.strings(c.ToolResult.CallID) || !b.contents(c.ToolResult.Content, depth+1)) {
			return false
		}
		if c.GeneratedImage != nil {
			g := c.GeneratedImage
			if !b.strings(string(g.Status), g.PartialImage, g.Size, g.Quality, g.Background, g.OutputFormat) ||
				(g.Result != nil && !b.strings(*g.Result)) {
				return false
			}
		}
	}
	return true
}
