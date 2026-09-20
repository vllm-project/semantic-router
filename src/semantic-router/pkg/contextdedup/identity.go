package contextdedup

import (
	"hash/fnv"
	"reflect"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Resolver returns the neutral request message behind one view message ID.
// The adapter that owns the request supplies it; the policy only reads the
// returned value. A false result means the message cannot be proven equal.
type Resolver func(messageID int) (llmprotocol.Message, bool)

// Normalize applies the configured text comparison. Exact mode returns the
// text unchanged; whitespace mode collapses runs of white space to one space
// and trims the ends. No mode changes case, punctuation, or Unicode form.
func Normalize(text string, mode Normalization) string {
	if mode == NormalizationWhitespace {
		return strings.Join(strings.Fields(text), " ")
	}
	return text
}

// turnIdentity is the stage-one identity of one turn. The key is an internal
// comparison value only; it is never placed on a receipt or a log.
type turnIdentity struct {
	hash uint64
	key  string
}

func identityOf(messages []contextcompression.MessageView, mode Normalization) turnIdentity {
	var builder strings.Builder
	for _, message := range messages {
		writeField(&builder, message.Role)
		writeField(&builder, string(message.Source))
		builder.WriteString(strconv.Itoa(len(message.Blocks)))
		builder.WriteByte(0)
		for _, block := range message.Blocks {
			writeField(&builder, string(block.Source))
			writeField(&builder, Normalize(block.Text, mode))
		}
		builder.WriteByte(1)
	}
	key := builder.String()
	digest := fnv.New64a()
	_, _ = digest.Write([]byte(key))
	return turnIdentity{hash: digest.Sum64(), key: key}
}

// writeField length-prefixes a value so adjacent fields cannot run together.
func writeField(builder *strings.Builder, value string) {
	builder.WriteString(strconv.Itoa(len(value)))
	builder.WriteByte(':')
	builder.WriteString(value)
	builder.WriteByte(0)
}

func (identity turnIdentity) equals(other turnIdentity) bool {
	return identity.hash == other.hash && identity.key == other.key
}

// EquivalentMessages proves that a later message is an exact repeat of an
// earlier one. It reports the bounded retention reason when it is not. The
// later copy may omit an ID the earlier copy carries, because a client that
// re-sends history it received from a provider often drops item IDs; a later
// copy carrying a different ID is a different item.
func EquivalentMessages(earlier, later llmprotocol.Message, mode Normalization) (bool, string) {
	if earlier.Role != later.Role || len(earlier.Content) != len(later.Content) {
		return false, RetainedIdentityMismatch
	}
	if later.ID != "" && later.ID != earlier.ID {
		return false, RetainedIdentityMismatch
	}
	for index := range earlier.Content {
		if ok, reason := equivalentContent(earlier.Content[index], later.Content[index], mode); !ok {
			return false, reason
		}
	}
	return true, ""
}

func equivalentContent(earlier, later llmprotocol.Content, mode Normalization) (bool, string) {
	if earlier.Kind != later.Kind {
		return false, RetainedIdentityMismatch
	}
	switch earlier.Kind {
	case llmprotocol.ContentRefusal:
		return false, RetainedRefusal
	case llmprotocol.ContentText, llmprotocol.ContentReasoning:
	default:
		return false, RetainedIdentityMismatch
	}
	if carriesNonText(earlier) || carriesNonText(later) {
		return false, RetainedIdentityMismatch
	}
	if Normalize(earlier.Text, mode) != Normalize(later.Text, mode) {
		return false, RetainedIdentityMismatch
	}
	if earlier.Signature != later.Signature || earlier.Reasoning != later.Reasoning {
		return false, RetainedIdentityMismatch
	}
	if !sameCitations(earlier.Citations, later.Citations) || !reflect.DeepEqual(earlier.Cache, later.Cache) {
		return false, RetainedIdentityMismatch
	}
	return true, ""
}

// sameCitations treats a missing list and an empty one alike, because codecs
// differ in which they produce for a block without citations.
func sameCitations(earlier, later []llmprotocol.Citation) bool {
	if len(earlier) == 0 && len(later) == 0 {
		return true
	}
	return reflect.DeepEqual(earlier, later)
}

// carriesNonText reports fields a text block must not carry. A block with any
// of them set is not plain text even when its Kind says so.
func carriesNonText(content llmprotocol.Content) bool {
	return content.ToolCall != nil || content.ToolResult != nil || content.GeneratedImage != nil ||
		content.MediaType != "" || content.URL != "" || content.Data != "" ||
		content.FileID != "" || content.Filename != "" || content.Detail != ""
}

// RequestResolver resolves view message IDs against the neutral request the
// IR was parsed from. Both keep one entry per message in the same order, and
// the shared executor removes from both together, so the position of a view
// ID in the IR is the position of its message in the request.
func RequestResolver(ir *contextcompression.RequestIR) Resolver {
	return func(id int) (llmprotocol.Message, bool) {
		if ir == nil || ir.Semantic == nil {
			return llmprotocol.Message{}, false
		}
		for position, message := range ir.Messages {
			if message.Index != id {
				continue
			}
			if position >= len(ir.Semantic.Messages) {
				return llmprotocol.Message{}, false
			}
			return ir.Semantic.Messages[position], true
		}
		return llmprotocol.Message{}, false
	}
}
