package tools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/openai/openai-go"
)

// decodeTool preserves schema numbers before any provider SDK decoding can
// collapse integers into float64. Rebuilding the known function fields also
// removes SDK raw-JSON overrides that could alias or shadow the owned schema.
func decodeTool(data []byte) (openai.ChatCompletionToolParam, error) {
	var wire struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description json.RawMessage `json:"description"`
			Strict      json.RawMessage `json:"strict"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	var tool openai.ChatCompletionToolParam
	if err := json.Unmarshal(data, &wire); err != nil {
		return tool, err
	}
	name := strings.TrimSpace(wire.Function.Name)
	if name == "" || len(name) > 256 || wire.Type != "" && wire.Type != "function" {
		return tool, fmt.Errorf("tools: invalid function identity")
	}
	tool.Function.Name = name
	tool.Type = "function"
	if len(wire.Function.Description) > 0 {
		if err := json.Unmarshal(wire.Function.Description, &tool.Function.Description); err != nil {
			return tool, err
		}
	}
	if len(wire.Function.Strict) > 0 {
		if err := json.Unmarshal(wire.Function.Strict, &tool.Function.Strict); err != nil {
			return tool, err
		}
	}
	if len(wire.Function.Parameters) > 0 {
		decoder := json.NewDecoder(bytes.NewReader(wire.Function.Parameters))
		decoder.UseNumber()
		if err := decoder.Decode(&tool.Function.Parameters); err != nil {
			return tool, fmt.Errorf("tools: invalid parameters: %w", err)
		}
	}
	return tool, nil
}

func prepareToolEntry(entry ToolEntry) (ToolEntry, error) {
	// Validate the provider boundary without serializing the SDK envelope:
	// its shim encoder treats encoding/json.Number as an ordinary string.
	if _, err := SemanticTool(entry.Tool); err != nil {
		return ToolEntry{}, err
	}
	name := strings.TrimSpace(entry.Tool.Function.Name)
	if name == "" || len(name) > 256 {
		return ToolEntry{}, fmt.Errorf("tools: invalid function identity")
	}
	data, err := json.Marshal(entry.Tool.Function.Parameters)
	if err != nil {
		return ToolEntry{}, fmt.Errorf("tools: invalid definition: %w", err)
	}
	tool := openai.ChatCompletionToolParam{Type: "function", Function: openai.FunctionDefinitionParam{
		Name: name, Description: entry.Tool.Function.Description, Strict: entry.Tool.Function.Strict,
	}}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&tool.Function.Parameters); err != nil {
		return ToolEntry{}, fmt.Errorf("tools: invalid parameters: %w", err)
	}
	entry.Tool = tool
	entry.Tags = slices.Clone(entry.Tags)
	return entry, nil
}

func finiteScore(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func validateEmbedding(vector []float32) error {
	if len(vector) == 0 {
		return fmt.Errorf("tools: empty embedding")
	}
	for _, value := range vector {
		if !finiteScore(value) {
			return fmt.Errorf("tools: embedding values must be finite")
		}
	}
	return nil
}

// appendEntries publishes a batch atomically. Recheck names while locked so a
// concurrent file load or AddTool cannot introduce a duplicate after validation.
func (db *ToolsDatabase) appendEntries(entries []ToolEntry) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	names := make(map[string]bool, len(db.entries)+len(entries))
	for _, entry := range db.entries {
		names[entry.Tool.Function.Name] = true
	}
	for _, entry := range entries {
		name := entry.Tool.Function.Name
		if names[name] {
			return fmt.Errorf("tools: duplicate normalized function name")
		}
		names[name] = true
	}
	for _, entry := range entries {
		db.entries = append(db.entries, cloneToolEntry(entry))
	}
	sort.Slice(db.entries, func(i, j int) bool {
		return db.entries[i].Tool.Function.Name < db.entries[j].Tool.Function.Name
	})
	// Only a published batch counts as availability evidence; a rejected one
	// returns above and leaves loadedAt unchanged.
	db.loadedAt = time.Now()
	return nil
}

func cloneToolEntry(entry ToolEntry) ToolEntry {
	entry.Embedding = slices.Clone(entry.Embedding)
	entry.Tags = slices.Clone(entry.Tags)
	entry.Tool = cloneToolDefinition(entry.Tool)
	return entry
}

func cloneToolDefinition(tool openai.ChatCompletionToolParam) openai.ChatCompletionToolParam {
	if tool.Function.Parameters != nil {
		tool.Function.Parameters = cloneSchemaValue(map[string]any(tool.Function.Parameters)).(map[string]any)
	}
	return tool
}

// Admission decodes with UseNumber. Convert numeric leaves into raw JSON so
// both encoding/json and the SDK's shim encoder preserve their numeric type.
// Raw bytes, maps and arrays are copied at every ownership boundary.
func cloneSchemaValue(value any) any {
	switch typed := value.(type) {
	case json.Number:
		return json.RawMessage(typed.String())
	case json.RawMessage:
		return slices.Clone(typed)
	case map[string]any:
		result := make(map[string]any, len(typed))
		for key, child := range typed {
			result[key] = cloneSchemaValue(child)
		}
		return result
	case []any:
		result := make([]any, len(typed))
		for i, child := range typed {
			result[i] = cloneSchemaValue(child)
		}
		return result
	default:
		return value
	}
}
