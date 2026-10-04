package masking

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	EntityTypes      []string       // sorted, de-duplicated
	EntityCounts     map[string]int // masked span count per entity type; a metric label, never a value (Phase 5)
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
		// A range copy would silently discard mutations.
		if err := applyToBlock(&blocks[i], a, scan, out, depth); err != nil {
			return err
		}
	}
	return nil
}

// applyToBlock dispatches one content block by kind.
func applyToBlock(
	block *llmprotocol.Content, a *Allocator, scan ScanFunc, out *Result, depth int,
) error {
	switch block.Kind {
	case llmprotocol.ContentText:
		return maskTextBlock(block, a, scan, out)
	case llmprotocol.ContentToolCall:
		return maskToolCallArguments(block.ToolCall, a, scan, out)
	case llmprotocol.ContentToolResult:
		if block.ToolResult != nil {
			return maskToolResult(block.ToolResult, a, scan, out, depth)
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
		incrementEntityCount(out, span.EntityType)
	}
	out.CitationsDropped += len(block.Citations) - len(survivingCitations)
	block.Text = maskedText
	block.Citations = survivingCitations
	return nil
}

// maskToolResult masks a tool result's content. A structured (JSON object or
// array) text payload is masked as decoded JSON leaves, so object keys stay
// intact, escapes are seen through, and the result stays valid JSON. Custom
// tools and plain-text payloads keep ordinary text masking.
func maskToolResult(
	result *llmprotocol.ToolResult, a *Allocator, scan ScanFunc, out *Result, depth int,
) error {
	if depth+1 > maxToolResultDepth {
		return fmt.Errorf("masking: tool result nesting exceeds depth %d", maxToolResultDepth)
	}
	for i := range result.Content {
		block := &result.Content[i]
		structured := block.Kind == llmprotocol.ContentText &&
			result.Kind != llmprotocol.ToolKindCustom &&
			isCompleteJSONDocument(block.Text)
		if !structured {
			if err := applyToBlock(block, a, scan, out, depth+1); err != nil {
				return err
			}
			continue
		}
		masked, changed, err := maskJSONDocument(block.Text, a, scan, out)
		if err != nil {
			return fmt.Errorf("masking: tool result %q content: %w", result.CallID, err)
		}
		if !changed {
			continue
		}
		out.Changed = true
		block.Text = masked
		// Offsets into a rewritten JSON document no longer denote anything,
		// and a citation on tool output is not attribution worth keeping (D6).
		out.CitationsDropped += len(block.Citations)
		block.Citations = nil
	}
	return nil
}

// isCompleteJSONDocument reports whether text is exactly one JSON object or
// array with nothing after it. A tool result is ordinary text, so a payload
// like `{"a":1} alice@example.com` is mixed text and belongs on the text path:
// decoding it would scan only the leading value and leave the rest unscanned.
// Bare scalars stay on the text path too, so a plain result is never
// reserialised.
func isCompleteJSONDocument(text string) bool {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
		return false
	}
	decoder := json.NewDecoder(strings.NewReader(trimmed))
	var probe json.RawMessage
	if err := decoder.Decode(&probe); err != nil {
		return false
	}
	return isAtEndOfInput(decoder)
}

// isAtEndOfInput reports whether a decoder that has read one value consumed
// the whole input. A second value, or trailing bytes that are not JSON at all,
// both mean the payload holds more than the value already decoded.
func isAtEndOfInput(decoder *json.Decoder) bool {
	var trailing json.RawMessage
	return errors.Is(decoder.Decode(&trailing), io.EOF)
}

// maskToolCallArguments masks string leaves of the arguments JSON. Keys,
// types and structure are preserved, so tool-call correlation and schema
// validity survive. ToolCall.ID is never touched.
func maskToolCallArguments(call *llmprotocol.ToolCall, a *Allocator, scan ScanFunc, out *Result) error {
	if call == nil || strings.TrimSpace(call.Arguments) == "" {
		return nil
	}
	// A custom tool carries free-form text in Arguments, not a JSON object
	// (llmprotocol.ToolKindCustom). Decoding it as JSON would refuse a valid
	// call, so it is masked as text.
	if call.Kind == llmprotocol.ToolKindCustom {
		masked, changed, err := maskStringValue(call.Arguments, a, scan, out)
		if err != nil {
			return err
		}
		if changed {
			out.Changed = true
			call.Arguments = masked
		}
		return nil
	}
	masked, changed, err := maskJSONDocument(call.Arguments, a, scan, out)
	if err != nil {
		return fmt.Errorf("masking: tool call %q arguments: %w", call.Name, err)
	}
	if !changed {
		return nil
	}
	out.Changed = true
	call.Arguments = masked
	return nil
}

// maskJSONDocument masks the string leaves of a JSON document and returns it
// re-serialised. Decoding first is what keeps object keys intact, keeps the
// result valid JSON, and sees through escapes such as a.
func maskJSONDocument(document string, a *Allocator, scan ScanFunc, out *Result) (string, bool, error) {
	// UseNumber preserves numeric literals exactly on re-marshal; decoding
	// into float64 can lose precision for a large int64 (risk 1).
	decoder := json.NewDecoder(strings.NewReader(document))
	decoder.UseNumber()
	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		// Content the router cannot parse cannot be masked, and dispatching
		// it unmasked would be a silent bypass (D4).
		return "", false, fmt.Errorf("not valid JSON: %w", err)
	}
	// Re-serialising would drop whatever follows the first value, and only
	// that value is ever scanned, so a trailing remainder is a bypass (D4).
	if !isAtEndOfInput(decoder) {
		return "", false, fmt.Errorf("trailing content after the first JSON value")
	}
	masked, changed, err := maskJSONValue(decoded, a, scan, out)
	if err != nil {
		return "", false, err
	}
	if !changed {
		return document, false, nil
	}
	serialized, err := json.Marshal(masked)
	if err != nil {
		return "", false, fmt.Errorf("failed to re-serialize masked JSON: %w", err)
	}
	return string(serialized), true, nil
}

// maskStringValue masks one plain string, reporting whether it changed.
func maskStringValue(text string, a *Allocator, scan ScanFunc, out *Result) (string, bool, error) {
	if text == "" {
		return text, false, nil
	}
	spans, err := scan(text)
	if err != nil {
		return "", false, fmt.Errorf("masking: scan failed: %w", err)
	}
	merged := MergeSpans(a.filter(spans))
	if len(merged) == 0 {
		return text, false, nil
	}
	masked, _, err := MaskText(text, spans, nil, a)
	if err != nil {
		return "", false, err
	}
	out.MaskedCount += len(merged)
	for _, span := range merged {
		out.EntityTypes = append(out.EntityTypes, span.EntityType)
		incrementEntityCount(out, span.EntityType)
	}
	return masked, true, nil
}

// maskJSONValue recurses into a decoded JSON value, masking string leaves and
// leaving object keys, numbers, booleans and null untouched.
func maskJSONValue(value any, a *Allocator, scan ScanFunc, out *Result) (any, bool, error) {
	switch typed := value.(type) {
	case string:
		masked, changed, err := maskStringValue(typed, a, scan, out)
		if err != nil {
			return nil, false, err
		}
		return masked, changed, nil
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

func incrementEntityCount(out *Result, entityType string) {
	if out.EntityCounts == nil {
		out.EntityCounts = make(map[string]int)
	}
	out.EntityCounts[entityType]++
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
