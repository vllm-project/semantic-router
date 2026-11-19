package protocolcodec

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Messages clients expect a reply's identifiers to carry these prefixes.
const (
	anthropicMessageIDPrefix = "msg_"
	anthropicToolUseIDPrefix = "toolu_"
)

// anthropicReplyID is the identifier a Messages client receives for one a
// backend assigned. An identifier from a backend of another wire format, such
// as Chat's chatcmpl-... ids, gets the prefix the client expects, derived from
// it so that a buffered and a streamed reply agree and the tool results a
// client sends in later turns pair with the calls it was given. An identifier
// from a Messages backend, or one that already has the prefix, is kept.
func anthropicReplyID(source llmprotocol.WireFormat, prefix, id string) string {
	if source == "" || source == llmprotocol.AnthropicMessagesV1 || id == "" || strings.HasPrefix(id, prefix) {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	return prefix + hex.EncodeToString(sum[:12])
}

// AnthropicReplyIdentifiers gives a response from a backend of the source wire
// format the identifiers a Messages client receives when the response is
// translated, for a reply rendered again from the neutral response. It never
// edits the caller's slices.
func AnthropicReplyIdentifiers(source llmprotocol.WireFormat, response llmprotocol.Response) llmprotocol.Response {
	response.ID = anthropicReplyID(source, anthropicMessageIDPrefix, response.ID)
	response.Output = slices.Clone(response.Output)
	for index := range response.Output {
		contents := slices.Clone(response.Output[index].Content)
		for position, content := range contents {
			if content.Kind != llmprotocol.ContentToolCall || content.ToolCall == nil {
				continue
			}
			call := *content.ToolCall
			call.ID = anthropicReplyID(source, anthropicToolUseIDPrefix, call.ID)
			contents[position].ToolCall = &call
		}
		response.Output[index].Content = contents
	}
	return response
}
