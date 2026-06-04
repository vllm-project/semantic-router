package mcpconfig

import "github.com/mark3labs/mcp-go/mcp"

func getConfigToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        ToolGetConfig,
		Description: "Return the current router config document as JSON.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"actor": actorProperty(),
			},
		},
	}
}

func exportConfigJSONToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        ToolExportConfigJSON,
		Description: "Export the current router config document as JSON.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"actor": actorProperty(),
			},
		},
	}
}

func exportConfigYAMLToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        ToolExportConfigYAML,
		Description: "Export the current router config document as canonical YAML.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"actor": actorProperty(),
			},
		},
	}
}

func validateConfigToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        ToolValidateConfig,
		Description: "Validate a router config document using the same parser as /config/router.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"actor":    actorProperty(),
				"document": documentProperty(),
				"merge":    mergeProperty(),
			},
			Required: []string{"document"},
		},
	}
}

func diffConfigToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        ToolDiffConfig,
		Description: "Preview how a patch merges into the current router config.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"actor": actorProperty(),
				"patch": patchProperty(),
				"merge": mergeProperty(),
			},
			Required: []string{"patch"},
		},
	}
}

func applyPatchToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        ToolApplyPatch,
		Description: "Apply a router config patch with merge semantics by default, then validate, backup, and hot-reload.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"actor": actorProperty(),
				"patch": patchProperty(),
				"merge": mergeProperty(),
				"dsl": map[string]any{
					"type":        "string",
					"description": "Optional DSL source archived with the config version.",
				},
			},
			Required: []string{"patch"},
		},
	}
}

func actorProperty() map[string]any {
	return map[string]any{
		"type":        "string",
		"description": "Optional actor identifier recorded in the audit log.",
	}
}

func documentProperty() map[string]any {
	return map[string]any{
		"type":        "object",
		"description": "Router config document object.",
	}
}

func patchProperty() map[string]any {
	return map[string]any{
		"type":        "object",
		"description": "Router config patch object.",
	}
}

func mergeProperty() map[string]any {
	return map[string]any{
		"type":        "boolean",
		"description": "When true (default), merge into the current config. When false, replace the document.",
	}
}
