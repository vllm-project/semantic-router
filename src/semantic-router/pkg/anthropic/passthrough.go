// Package anthropic - passthrough carrier for Anthropic-only request fields.
//
// The default request-body translation in this package takes an OpenAI-shape
// ChatCompletionNewParams and produces an Anthropic-shape MessageNewParams.
// Several Anthropic-only fields have no OpenAI representation (cache_control
// markers, top_k, metadata.user_id, multi-block system prompts with per-block
// markers, image content blocks, tool_result.is_error, tool_result with array
// content, anthropic-version, anthropic-beta). When those fields are present on
// the inbound request, today's translation silently drops them.
//
// AnthropicPassthrough is a per-request sidecar that the caller populates from
// the raw inbound body (or supplies directly when the upstream IR knows the
// shape) and ToAnthropicRequestBodyWithPassthrough consumes during rebuild.
// Every consumer treats a nil *AnthropicPassthrough as "no passthrough, use
// today's defaults", so existing callers remain byte-identical.
package anthropic

import (
	"fmt"

	"github.com/tidwall/gjson"
)

// AnthropicPassthrough carries Anthropic-only request fields that have no
// OpenAI representation, so the outbound write path can replay them on the
// request body and headers. Lifetime is one request: built by the caller from
// the raw inbound body (or supplied directly by upstream IR), consumed by
// ToAnthropicRequestBodyWithPassthrough and BuildRequestHeadersWithPassthrough,
// then discarded.
type AnthropicPassthrough struct {
	// AnthropicVersion is the inbound anthropic-version header value. Empty
	// means "use the package default" (AnthropicAPIVersion).
	AnthropicVersion string

	// AnthropicBeta is the inbound anthropic-beta header value, comma-separated
	// per Anthropic convention. Empty means "do not emit".
	AnthropicBeta string

	// TopK preserves the Anthropic top_k sampling parameter. nil means "not set
	// on inbound; do not emit".
	TopK *int64

	// MetadataUserID populates metadata.user_id on the outbound body. Empty
	// means "do not emit".
	MetadataUserID string

	// SystemBlocks preserves array-form system prompts. When non-empty,
	// overrides the string-form system derived from OpenAI messages.
	SystemBlocks []SystemBlock

	// CacheControl maps stable block IDs to cache_control markers. Block IDs
	// match the JSON-path convention: "system[i]",
	// "messages[i].content[j]", "tools[i]". The emitter applies markers
	// best-effort: when the block exists on the outbound side, attach; when an
	// upstream mutation removed the block, drop silently.
	CacheControl map[string]CacheControlSpec

	// ToolResultErrors maps tool_call_id to the is_error flag, so the emitter
	// can restore is_error: true on the matching outbound tool_result.
	ToolResultErrors map[string]bool

	// ToolResultArrayContent maps tool_call_id to the preserved array content
	// (text + image blocks) for tool_result blocks whose original content was
	// an array rather than a plain string.
	ToolResultArrayContent map[string][]ToolResultContentBlock

	// UserMessageImageBlocks maps user-message index to image blocks dropped by
	// the OpenAI-side flatten. The emitter re-attaches them to the
	// corresponding user message on the outbound side. Indices are counted
	// against user messages only (i.e. index 0 is the first user message,
	// regardless of any preceding system or assistant messages).
	UserMessageImageBlocks map[int][]ImageBlock
}

// SystemBlock represents one element of an array-form Anthropic system prompt.
type SystemBlock struct {
	Text         string
	CacheControl *CacheControlSpec
}

// CacheControlSpec captures the Anthropic cache_control marker. Type is always
// "ephemeral" today; TTL is one of "", "5m", "1h".
type CacheControlSpec struct {
	Type string
	TTL  string
}

// ToolResultContentBlock represents one block of a tool_result content array.
// Type is "text" or "image"; only the matching fields are populated.
type ToolResultContentBlock struct {
	Type   string
	Text   string
	Source *ImageSource
}

// ImageBlock represents a user-message image content block.
type ImageBlock struct {
	Source ImageSource
}

// ImageSource represents the source of an image content block. Type is
// "base64" or "url"; only the matching fields are populated.
type ImageSource struct {
	Type      string
	MediaType string
	Data      string
	URL       string
}

// SetHeadersFromIncoming populates AnthropicVersion and AnthropicBeta from a
// case-insensitive lookup of the incoming request header map. Empty header
// values are ignored so the package default applies.
func (p *AnthropicPassthrough) SetHeadersFromIncoming(headers map[string]string) {
	if p == nil || len(headers) == 0 {
		return
	}
	for k, v := range headers {
		if v == "" {
			continue
		}
		switch lowerASCII(k) {
		case "anthropic-version":
			p.AnthropicVersion = v
		case "anthropic-beta":
			p.AnthropicBeta = v
		}
	}
}

// lowerASCII lower-cases ASCII letters in s. HTTP header names are ASCII per
// RFC 7230, so strings.ToLower (which is Unicode-aware) is unnecessary
// overhead.
func lowerASCII(s string) string {
	b := []byte(s)
	for i := 0; i < len(b); i++ {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
}

// BuildPassthroughFromAnthropicBody parses the raw Anthropic-shape JSON body
// and returns a populated passthrough carrier. Lenient: malformed JSON or
// unrecognised shapes are silently skipped so the outbound path falls back to
// today's defaults rather than failing the request. Returns (nil, nil) when
// the input is empty.
func BuildPassthroughFromAnthropicBody(raw []byte) (*AnthropicPassthrough, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	pt := &AnthropicPassthrough{}
	captureSamplingAndMetadata(raw, pt)
	captureSystemBlocks(raw, pt)
	captureMessages(raw, pt)
	captureTools(raw, pt)
	return pt, nil
}

func captureSamplingAndMetadata(raw []byte, pt *AnthropicPassthrough) {
	if v := gjson.GetBytes(raw, "top_k"); v.Exists() && v.Type == gjson.Number {
		k := v.Int()
		pt.TopK = &k
	}
	if v := gjson.GetBytes(raw, "metadata.user_id"); v.Exists() && v.Type == gjson.String {
		pt.MetadataUserID = v.String()
	}
}

func captureSystemBlocks(raw []byte, pt *AnthropicPassthrough) {
	sys := gjson.GetBytes(raw, "system")
	if !sys.IsArray() {
		return
	}
	sys.ForEach(func(i, block gjson.Result) bool {
		sb := SystemBlock{Text: block.Get("text").String()}
		if cc := block.Get("cache_control"); cc.Exists() {
			spec := parsePassthroughCacheControl(cc)
			sb.CacheControl = &spec
			pt.setCacheControl(fmt.Sprintf("system[%d]", i.Int()), spec)
		}
		pt.SystemBlocks = append(pt.SystemBlocks, sb)
		return true
	})
}

func captureMessages(raw []byte, pt *AnthropicPassthrough) {
	userMsgIndex := -1
	gjson.GetBytes(raw, "messages").ForEach(func(mi, msg gjson.Result) bool {
		isUser := msg.Get("role").String() == "user"
		if isUser {
			userMsgIndex++
		}
		msg.Get("content").ForEach(func(ci, content gjson.Result) bool {
			captureContentBlock(pt, mi.Int(), ci.Int(), isUser, userMsgIndex, content)
			return true
		})
		return true
	})
}

func captureContentBlock(pt *AnthropicPassthrough, mi, ci int64, isUser bool, userMsgIndex int, content gjson.Result) {
	if cc := content.Get("cache_control"); cc.Exists() {
		pt.setCacheControl(fmt.Sprintf("messages[%d].content[%d]", mi, ci), parsePassthroughCacheControl(cc))
	}
	switch content.Get("type").String() {
	case "image":
		if isUser {
			if img, ok := parseImageBlock(content); ok {
				pt.appendImageBlock(userMsgIndex, img)
			}
		}
	case "tool_result":
		captureToolResult(pt, content)
	}
}

func captureToolResult(pt *AnthropicPassthrough, content gjson.Result) {
	id := content.Get("tool_use_id").String()
	if id == "" {
		return
	}
	if content.Get("is_error").Bool() {
		pt.setToolResultError(id, true)
	}
	if c := content.Get("content"); c.IsArray() {
		if blocks := parseToolResultBlocks(c); len(blocks) > 0 {
			pt.setToolResultArrayContent(id, blocks)
		}
	}
}

func captureTools(raw []byte, pt *AnthropicPassthrough) {
	gjson.GetBytes(raw, "tools").ForEach(func(i, tool gjson.Result) bool {
		if cc := tool.Get("cache_control"); cc.Exists() {
			pt.setCacheControl(fmt.Sprintf("tools[%d]", i.Int()), parsePassthroughCacheControl(cc))
		}
		return true
	})
}

func parsePassthroughCacheControl(cc gjson.Result) CacheControlSpec {
	spec := CacheControlSpec{Type: cc.Get("type").String()}
	if spec.Type == "" {
		spec.Type = "ephemeral"
	}
	if ttl := cc.Get("ttl").String(); ttl != "" {
		spec.TTL = ttl
	}
	return spec
}

func parseImageBlock(content gjson.Result) (ImageBlock, bool) {
	source := content.Get("source")
	if !source.Exists() {
		return ImageBlock{}, false
	}
	src := parseImageSource(source)
	if src.Type == "" {
		return ImageBlock{}, false
	}
	return ImageBlock{Source: src}, true
}

func parseImageSource(source gjson.Result) ImageSource {
	srcType := source.Get("type").String()
	switch srcType {
	case "base64":
		return ImageSource{
			Type:      "base64",
			MediaType: source.Get("media_type").String(),
			Data:      source.Get("data").String(),
		}
	case "url":
		return ImageSource{
			Type: "url",
			URL:  source.Get("url").String(),
		}
	}
	return ImageSource{}
}

func parseToolResultBlocks(content gjson.Result) []ToolResultContentBlock {
	var blocks []ToolResultContentBlock
	content.ForEach(func(_, block gjson.Result) bool {
		switch block.Get("type").String() {
		case "text":
			blocks = append(blocks, ToolResultContentBlock{
				Type: "text",
				Text: block.Get("text").String(),
			})
		case "image":
			src := parseImageSource(block.Get("source"))
			if src.Type != "" {
				blocks = append(blocks, ToolResultContentBlock{
					Type:   "image",
					Source: &src,
				})
			}
		}
		return true
	})
	return blocks
}

func (p *AnthropicPassthrough) setCacheControl(id string, spec CacheControlSpec) {
	if p.CacheControl == nil {
		p.CacheControl = map[string]CacheControlSpec{}
	}
	p.CacheControl[id] = spec
}

func (p *AnthropicPassthrough) setToolResultError(id string, isError bool) {
	if p.ToolResultErrors == nil {
		p.ToolResultErrors = map[string]bool{}
	}
	p.ToolResultErrors[id] = isError
}

func (p *AnthropicPassthrough) setToolResultArrayContent(id string, blocks []ToolResultContentBlock) {
	if p.ToolResultArrayContent == nil {
		p.ToolResultArrayContent = map[string][]ToolResultContentBlock{}
	}
	p.ToolResultArrayContent[id] = blocks
}

func (p *AnthropicPassthrough) appendImageBlock(userMsgIndex int, img ImageBlock) {
	if p.UserMessageImageBlocks == nil {
		p.UserMessageImageBlocks = map[int][]ImageBlock{}
	}
	p.UserMessageImageBlocks[userMsgIndex] = append(p.UserMessageImageBlocks[userMsgIndex], img)
}
