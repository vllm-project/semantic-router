package config

import (
	"fmt"
	"regexp"
	"strings"
)

// Bounds for global.router.routing_input. Every user message of every request
// is matched against the patterns and scanned for the markers.
const (
	MaxRoutingInputSkipPatterns = 32
	MaxRoutingInputPatternBytes = 1024
	MaxRoutingInputStripBlocks  = 32
	MaxRoutingInputMarkerBytes  = 256
)

// RoutingInputConfig chooses the user text that routing reads. Routing signals
// and their conversation facts, the semantic-cache query, tool selection, the
// RAG query, memory retrieval and storage, the model-selection query, the
// hallucination check, Replay's prompt and the routing preview use the
// selected text; the request sent to the model is unchanged. Jailbreak, PII and
// content-safety signals keep scanning the text it leaves out. Empty lists
// route on the last user message that has text.
type RoutingInputConfig struct {
	// SkipUserMessagesMatching lists RE2 patterns. A user message whose text
	// matches one is never routing text and does not count as a user turn.
	// A pattern matches anywhere in the text unless anchored with ^ and $. A
	// message that also carries images, audio or tool results is never skipped.
	SkipUserMessagesMatching []string `yaml:"skip_user_messages_matching,omitempty" jsonschema:"maxItems=32"`
	// StripBlocks cuts spans from the routing text, markers included. Markers
	// are literal and case-sensitive. Blocks do not nest: a block ends at the
	// first end marker after its start marker, and a start marker with no end
	// marker after it is left in place.
	StripBlocks []RoutingInputStripBlock `yaml:"strip_blocks,omitempty" jsonschema:"maxItems=32"`
}

// RoutingInputStripBlock delimits the spans of one block type.
type RoutingInputStripBlock struct {
	// Start is the literal marker that opens a span.
	Start string `yaml:"start" jsonschema:"required,minLength=1,maxLength=256"`
	// End is the literal marker that closes a span.
	End string `yaml:"end" jsonschema:"required,minLength=1,maxLength=256"`
}

// Enabled reports whether the configuration can change the routing text.
func (c RoutingInputConfig) Enabled() bool {
	return len(c.SkipUserMessagesMatching) > 0 || len(c.StripBlocks) > 0
}

func validateRoutingInputContracts(cfg *RouterConfig) error {
	input := cfg.RoutingInput
	const prefix = "global.router.routing_input"
	if len(input.SkipUserMessagesMatching) > MaxRoutingInputSkipPatterns {
		return fmt.Errorf("%s.skip_user_messages_matching: at most %d patterns are allowed, got %d",
			prefix, MaxRoutingInputSkipPatterns, len(input.SkipUserMessagesMatching))
	}
	for index, pattern := range input.SkipUserMessagesMatching {
		field := fmt.Sprintf("%s.skip_user_messages_matching[%d]", prefix, index)
		if strings.TrimSpace(pattern) == "" {
			return fmt.Errorf("%s: pattern must not be empty", field)
		}
		if len(pattern) > MaxRoutingInputPatternBytes {
			return fmt.Errorf("%s: pattern must be at most %d bytes", field, MaxRoutingInputPatternBytes)
		}
		compiled, err := regexp.Compile(pattern)
		if err != nil {
			return fmt.Errorf("%s: invalid RE2 pattern: %w", field, err)
		}
		if compiled.MatchString("") {
			return fmt.Errorf("%s: pattern matches empty text, so it would skip every user message", field)
		}
	}
	if len(input.StripBlocks) > MaxRoutingInputStripBlocks {
		return fmt.Errorf("%s.strip_blocks: at most %d blocks are allowed, got %d",
			prefix, MaxRoutingInputStripBlocks, len(input.StripBlocks))
	}
	for index, block := range input.StripBlocks {
		for _, marker := range []struct{ name, value string }{{"start", block.Start}, {"end", block.End}} {
			field := fmt.Sprintf("%s.strip_blocks[%d].%s", prefix, index, marker.name)
			if strings.TrimSpace(marker.value) == "" {
				return fmt.Errorf("%s: marker must not be empty", field)
			}
			if len(marker.value) > MaxRoutingInputMarkerBytes {
				return fmt.Errorf("%s: marker must be at most %d bytes", field, MaxRoutingInputMarkerBytes)
			}
		}
	}
	return nil
}
