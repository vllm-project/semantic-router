package masking

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// maxToolResultDepth bounds ToolResult.Content recursion, matching the
// decode-time ToolResultDepth limit (llmprotocol/policy.go:89, default 8). A
// neutral request cannot legally nest deeper than that, so this is a
// consistency check, not a new restriction.
const maxToolResultDepth = 8

// ScanFunc produces PII spans for one text. It is injected so this package
// never imports pkg/classification, which links the Rust bindings (D7).
type ScanFunc func(text string) ([]Span, error)

// Result carries only bounded, non-identifying facts. Raw values and the
// value-to-placeholder mapping never leave this package (#3566).
type Result struct {
	Changed          bool
	MaskedCount      int
	EntityTypes      []string // sorted, de-duplicated
	CitationsDropped int
}

// Apply walks a neutral request and masks every text-bearing position:
// system/developer instructions, message content, tool call arguments, and
// content nested inside tool results (requirements 1-4). Everything else is
// left alone by construction — see applyToContent's skip list (requirement
// 5). Any scan error aborts the walk immediately (requirement 7): the block
// being scanned at the time of the error is never mutated, and the caller is
// expected to discard the whole request rather than dispatch it (D4).
func Apply(request *llmprotocol.Request, a *Allocator, scan ScanFunc) (Result, error) {
	var result Result
	// Instructions before Messages, so placeholder indices are stable
	// regardless of where a value first appears in the request.
	for i := range request.Instructions {
		if err := applyToContent(request.Instructions[i].Content, a, scan, &result, 0); err != nil {
			return Result{}, err
		}
	}
	for i := range request.Messages {
		if err := applyToContent(request.Messages[i].Content, a, scan, &result, 0); err != nil {
			return Result{}, err
		}
	}
	result.EntityTypes = dedupSortedEntityTypes(result.EntityTypes)
	return result, nil
}

// applyToContent dispatches by content kind. This switch is the security
// contract for what masking touches, so it is exhaustive and every branch is
// commented (requirement 5).
func applyToContent(blocks []llmprotocol.Content, a *Allocator, scan ScanFunc, out *Result, depth int) error {
	if depth > maxToolResultDepth {
		return fmt.Errorf("masking: tool result nesting exceeds depth %d", maxToolResultDepth)
	}
	for i := range blocks {
		block := &blocks[i] // a range copy would silently discard mutations
		switch block.Kind {
		case llmprotocol.ContentText:
			if err := maskTextBlock(block, a, scan, out); err != nil {
				return err
			}
		case llmprotocol.ContentToolCall:
			if err := maskToolCallArguments(block.ToolCall, a, scan, out); err != nil {
				return err
			}
		case llmprotocol.ContentToolResult:
			if block.ToolResult != nil {
				if err := applyToContent(block.ToolResult.Content, a, scan, out, depth+1); err != nil {
					return err
				}
			}
		case llmprotocol.ContentReasoning:
			// Anthropic thinking blocks carry a provider Signature over their
			// text. Rewriting the text invalidates it and the provider
			// rejects the request, so reasoning is never masked (#3566).
		default:
			// Media (image/audio/video/file), generated images, and refusal
			// blocks are explicit non-goals: the issue scopes masking to text
			// and structured tool payloads, not binary payloads, URLs, or
			// opaque identifiers.
		}
	}
	return nil
}

// maskTextBlock replaces PII in one text block's Text and adjusts its
// Citations, in place. Empty text short-circuits without calling scan
// (requirement 8).
func maskTextBlock(block *llmprotocol.Content, a *Allocator, scan ScanFunc, out *Result) error {
	if block.Text == "" {
		return nil
	}
	spans, err := scan(block.Text)
	if err != nil {
		return fmt.Errorf("masking: scan failed: %w", err)
	}
	// Filtering and merging here (in addition to inside MaskText) only
	// counts what will actually be masked; MaskText re-derives the same set
	// to perform the splice, so no placeholder is allocated twice.
	merged := MergeSpans(a.filter(spans))
	if len(merged) == 0 {
		return nil
	}
	maskedText, survivingCitations, err := MaskText(block.Text, spans, block.Citations, a)
	if err != nil {
		return err
	}
	out.Changed = true
	out.MaskedCount += len(merged)
	for _, span := range merged {
		out.EntityTypes = append(out.EntityTypes, span.EntityType)
	}
	out.CitationsDropped += len(block.Citations) - len(survivingCitations)
	block.Text = maskedText
	block.Citations = survivingCitations
	return nil
}

// maskToolCallArguments masks string leaves of the arguments JSON. Keys,
// types and structure are preserved, so tool-call correlation and schema
// validity survive. ToolCall.ID is never touched.
func maskToolCallArguments(call *llmprotocol.ToolCall, a *Allocator, scan ScanFunc, out *Result) error {
	if call == nil || strings.TrimSpace(call.Arguments) == "" {
		return nil
	}
	// UseNumber preserves numeric literals exactly on re-marshal; decoding
	// into float64 can lose precision for a large int64 (risk 1).
	decoder := json.NewDecoder(strings.NewReader(call.Arguments))
	decoder.UseNumber()
	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		// Arguments the router cannot parse cannot be masked, and dispatching
		// them unmasked would be a silent bypass (D4).
		return fmt.Errorf("masking: tool call %q arguments are not valid JSON: %w", call.Name, err)
	}
	masked, changed, err := maskJSONValue(decoded, a, scan, out)
	if err != nil {
		return err
	}
	if !changed {
		return nil
	}
	serialized, err := json.Marshal(masked)
	if err != nil {
		return fmt.Errorf("masking: failed to re-serialize masked arguments for tool call %q: %w", call.Name, err)
	}
	out.Changed = true
	call.Arguments = string(serialized)
	return nil
}

// maskJSONValue recurses into a decoded JSON value, masking string leaves and
// leaving object keys, numbers, booleans and null untouched.
func maskJSONValue(value any, a *Allocator, scan ScanFunc, out *Result) (any, bool, error) {
	switch typed := value.(type) {
	case string:
		if typed == "" {
			return typed, false, nil
		}
		spans, err := scan(typed)
		if err != nil {
			return nil, false, fmt.Errorf("masking: scan failed: %w", err)
		}
		merged := MergeSpans(a.filter(spans))
		if len(merged) == 0 {
			return typed, false, nil
		}
		masked, _, err := MaskText(typed, spans, nil, a)
		if err != nil {
			return nil, false, err
		}
		out.MaskedCount += len(merged)
		for _, span := range merged {
			out.EntityTypes = append(out.EntityTypes, span.EntityType)
		}
		return masked, true, nil
	case map[string]any:
		changed := false
		for key, child := range typed {
			maskedChild, childChanged, err := maskJSONValue(child, a, scan, out)
			if err != nil {
				return nil, false, err
			}
			if childChanged {
				typed[key] = maskedChild
				changed = true
			}
		}
		return typed, changed, nil
	case []any:
		changed := false
		for i, child := range typed {
			maskedChild, childChanged, err := maskJSONValue(child, a, scan, out)
			if err != nil {
				return nil, false, err
			}
			if childChanged {
				typed[i] = maskedChild
				changed = true
			}
		}
		return typed, changed, nil
	default:
		// json.Number, bool, and nil are never masked.
		return value, false, nil
	}
}

func dedupSortedEntityTypes(entityTypes []string) []string {
	if len(entityTypes) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(entityTypes))
	unique := make([]string, 0, len(entityTypes))
	for _, entityType := range entityTypes {
		if !seen[entityType] {
			seen[entityType] = true
			unique = append(unique, entityType)
		}
	}
	sort.Strings(unique)
	return unique
}
