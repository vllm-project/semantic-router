package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/tidwall/gjson"
)

// OpenAI response_format discriminator values from the Chat Completions API.
const (
	responseFormatTypeText       = "text"
	responseFormatTypeJSONObject = "json_object"
	responseFormatTypeJSONSchema = "json_schema"
)

// restoreResponseFormat rehydrates req.ResponseFormat from the raw request
// bytes after the SDK unmarshal.
//
// openai-go's ChatCompletionNewParamsResponseFormatUnion unmarshals by picking
// the first structurally matching variant, and the type-only text variant
// matches every response_format object. The union therefore keeps the "type"
// string but silently drops the json_schema payload (name, strict, schema).
// Every path that re-serializes the SDK struct then forwards
// {"type":"json_schema"} with no schema, and the backend degrades to plain
// JSON mode while still returning HTTP 200 (issue #3024).
//
// Re-decoding the documented variants directly from the client bytes keeps
// full response_format semantics on every struct-serialized body. Unknown
// type values are left exactly as the SDK parsed them so vendor-specific
// formats keep their current behavior.
func restoreResponseFormat(raw []byte, req *openai.ChatCompletionNewParams) error {
	format := gjson.GetBytes(raw, "response_format")
	if !format.Exists() || !format.IsObject() {
		return nil
	}

	formatRaw := []byte(format.Raw)
	switch format.Get("type").String() {
	case responseFormatTypeJSONSchema:
		// A non-object json_schema payload cannot be represented by the SDK
		// variant and would be dropped silently on re-serialization; reject it
		// instead of downgrading the schema contract.
		if payload := format.Get("json_schema"); payload.Exists() && !payload.IsObject() {
			return fmt.Errorf("response_format.json_schema must be an object")
		}
		var variant shared.ResponseFormatJSONSchemaParam
		if err := json.Unmarshal(formatRaw, &variant); err != nil {
			return fmt.Errorf("invalid response_format.json_schema: %w", err)
		}
		req.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &variant}
	case responseFormatTypeJSONObject:
		var variant shared.ResponseFormatJSONObjectParam
		if err := json.Unmarshal(formatRaw, &variant); err != nil {
			return fmt.Errorf("invalid response_format.json_object: %w", err)
		}
		req.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &variant}
	case responseFormatTypeText:
		var variant shared.ResponseFormatTextParam
		if err := json.Unmarshal(formatRaw, &variant); err != nil {
			return fmt.Errorf("invalid response_format.text: %w", err)
		}
		req.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfText: &variant}
	}
	return nil
}
