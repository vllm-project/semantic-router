package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// maxToolNameLength bounds a catalog tool's name.
const maxToolNameLength = 256

// CatalogTool is one tool of a tools_db catalog: its definition as a request
// carries it, and the description the decision model reads to choose it,
// the entry's own or else the tool's.
type CatalogTool struct {
	Tool        llmprotocol.Tool
	Description string
}

// DuplicateToolError is a catalog that lists one tool name twice.
type DuplicateToolError struct {
	Name string
}

func (e *DuplicateToolError) Error() string {
	return fmt.Sprintf("tool %q is listed twice", e.Name)
}

// AllowedTools are the catalog's tools the policy allows, in catalog order.
func (p *ToolsPolicyConfig) AllowedTools(catalog []CatalogTool) []CatalogTool {
	allowed := make([]CatalogTool, 0, len(catalog))
	for _, tool := range catalog {
		if p.Allows(tool.Tool.Name) {
			allowed = append(allowed, tool)
		}
	}
	return allowed
}

// LoadToolCatalog reads a tools_db catalog, a JSON list of
// {"tool": {"type": "function", "function": {...}}, "description": "..."}
// entries, in file order.
func LoadToolCatalog(path string) ([]CatalogTool, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	var entries []struct {
		Tool struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Strict      *bool           `json:"strict"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tool"`
		Description string `json:"description"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	catalog := make([]CatalogTool, 0, len(entries))
	seen := make(map[string]bool, len(entries))
	for index, entry := range entries {
		function := entry.Tool.Function
		name := strings.TrimSpace(function.Name)
		switch {
		case entry.Tool.Type != "" && entry.Tool.Type != "function":
			return nil, fmt.Errorf("%s: entry %d: tool type must be function", path, index)
		case name == "" || len(name) > maxToolNameLength:
			return nil, fmt.Errorf("%s: entry %d: function name must have 1 to %d characters", path, index, maxToolNameLength)
		case seen[name]:
			return nil, &DuplicateToolError{Name: name}
		}
		seen[name] = true
		schema := json.RawMessage(`{"type":"object"}`)
		if len(function.Parameters) > 0 && string(function.Parameters) != "null" {
			var object map[string]json.RawMessage
			if json.Unmarshal(function.Parameters, &object) != nil {
				return nil, fmt.Errorf("%s: tool %q: parameters must be a JSON Schema object", path, name)
			}
			schema = function.Parameters
		}
		description := strings.TrimSpace(entry.Description)
		if description == "" {
			description = strings.TrimSpace(function.Description)
		}
		catalog = append(catalog, CatalogTool{
			Tool:        llmprotocol.Tool{Name: name, Description: function.Description, Strict: function.Strict, InputSchema: schema},
			Description: description,
		})
	}
	return catalog, nil
}
