package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// loadToolsPlugin loads a configuration whose one decision, agent, has the
// tools plugin configuration written as a YAML flow mapping.
func loadToolsPlugin(configuration string) (*RouterConfig, error) {
	return ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults: {model: local}
  models:
    - name: local
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
      description: Local model
recipes:
  - name: default
    decisions:
      - name: agent
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: local}]
        plugins:
          - type: tools
            configuration: ` + configuration + "\n"))
}

// writeToolCatalog writes a catalog of the named tools and returns its path.
func writeToolCatalog(t *testing.T, names ...string) string {
	t.Helper()
	entries := make([]map[string]any, len(names))
	for index, name := range names {
		entries[index] = map[string]any{
			"tool":        map[string]any{"type": "function", "function": map[string]any{"name": name, "description": "Tool " + name}},
			"description": "Use " + name,
		}
	}
	data, err := json.Marshal(entries)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tools_db.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func numberedTools(count int) []string {
	names := make([]string, count)
	for index := range names {
		names[index] = fmt.Sprintf("tool_%03d", index)
	}
	return names
}

func TestToolsPluginLoads(t *testing.T) {
	catalog := writeToolCatalog(t, "docs.search", "tickets.lookup", "admin.delete")
	for name, configuration := range map[string]string{
		"policy only":       `{enabled: true, policy: {mode: none, strip_tool_history: true}}`,
		"filter selection":  `{enabled: true, policy: {mode: filtered, allow_tools: [a, b]}, select: {mode: filter, preserve_count: 1, need_threshold: 0.3, on_error: strip, input: {history: true}, timeout_ms: 500}}`,
		"add selection":     `{enabled: true, policy: {mode: filtered, block_tools: [admin.delete]}, select: {mode: add, tools_db_path: ` + catalog + `, match: {judge: {preset: tool_rank, top_k: 2, min_score: 0.2}}}}`,
		"bm25 narrowing":    `{enabled: true, select: {mode: filter, match: {recall: {method: bm25, top_k: 40}, judge: {preset: tool_rank, max_candidates: 20}}}}`,
		"a disabled plugin": `{enabled: false}`,
	} {
		if _, err := loadToolsPlugin(configuration); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// Each error is one line that names the field by its document path.
func TestToolsPluginLoadErrors(t *testing.T) {
	const at = "recipes[default].decisions[agent].plugins[0].configuration"
	twice := writeToolCatalog(t, "docs.search", "docs.search")
	cases := map[string]struct{ configuration, want string }{
		"select without a mode":       {`{enabled: true, select: {}}`, at + ".select.mode must be filter or add"},
		"selection after policy none": {`{enabled: true, policy: {mode: none}, select: {mode: filter}}`, "policy mode none removes every tool"},
		"add without a catalog":       {`{enabled: true, select: {mode: add}}`, at + ".select.tools_db_path is required"},
		"filter with a catalog":       {`{enabled: true, select: {mode: filter, tools_db_path: tools.json}}`, "tools_db_path is read in add mode"},
		"add with preserve_count":     {`{enabled: true, select: {mode: add, tools_db_path: tools.json, preserve_count: 1}}`, "preserve_count keeps request tools in filter mode"},
		"on_overflow":                 {`{enabled: true, select: {mode: filter, on_overflow: reject}}`, `unknown field "on_overflow"`},
		"add with on_error keep":      {`{enabled: true, select: {mode: add, tools_db_path: tools.json, on_error: keep}}`, "on_error must be skip or fail in add mode"},
		"filter with on_error skip":   {`{enabled: true, select: {mode: filter, on_error: skip}}`, "on_error must be keep, strip or fail in filter mode"},
		"preserve_count too large":    {`{enabled: true, select: {mode: filter, preserve_count: 256}}`, at + ".select.preserve_count must be within [0, 255]"},
		"need_threshold out of range": {`{enabled: true, select: {mode: filter, need_threshold: 1.5}}`, at + ".select.need_threshold must be within [0, 1]"},
		"top_k too large":             {`{enabled: true, select: {mode: filter, match: {judge: {preset: tool_rank, top_k: 256}}}}`, at + ".select.match.judge.top_k must be within [1, 255]"},
		"min_ratio out of range":      {`{enabled: true, select: {mode: filter, match: {judge: {preset: tool_rank, min_ratio: 0}}}}`, at + ".select.match.judge.min_ratio must be within (0, 1]"},
		"judge on_error":              {`{enabled: true, select: {mode: filter, match: {judge: {preset: tool_rank, on_error: fail}}}}`, "judge.on_error: the selection's on_error decides"},
		"judge timeout":               {`{enabled: true, select: {mode: filter, match: {judge: {preset: tool_rank, timeout_ms: 300}}}}`, "judge.timeout_ms: the selection's timeout_ms bounds its call"},
		"embedding recall":            {`{enabled: true, select: {mode: filter, match: {recall: {method: embedding, top_k: 5}}}}`, at + ".select.match.recall.method: tool selection recalls with bm25 only"},
		"choice reading 256":          {`{enabled: true, select: {mode: filter, match: {judge: {preset: tool_rank, max_candidates: 256}}}}`, at + ".select.match.judge.max_candidates must be within [1, 255]"},
		"empty match block":           {`{enabled: true, select: {mode: filter, match: {}}}`, at + ".select.match: match_empty:"},
		"sticky enabled":              {`{enabled: true, select: {mode: add, tools_db_path: tools.json, sticky: {enabled: true}}}`, "sticky.enabled is not supported in this release"},
		"filtered without lists":      {`{enabled: true, policy: {mode: filtered}}`, at + ".policy: mode filtered requires allow_tools or block_tools"},
		"strip history without none":  {`{enabled: true, policy: {strip_tool_history: true}}`, "strip_tool_history requires mode none"},
		"an unknown select field":     {`{enabled: true, select: {mode: filter, relevance_threshold: 0.3}}`, `unknown field "relevance_threshold"`},
		"an untrusted trust source":   {`{enabled: true, policy: {trusted_facts: {enabled: true, trust_sources: [prompt], stage_roles: [final]}}}`, at + `.policy.trusted_facts: trust_sources "prompt" is not authoritative`},
		"a tool listed twice":         {`{enabled: true, select: {mode: add, tools_db_path: ` + twice + `}}`, at + ".select.tools_db_path: duplicate_tool_name: " + twice + ` lists tool "docs.search" twice`},
	}
	for name, test := range cases {
		_, err := loadToolsPlugin(test.configuration)
		if err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("%s: err = %v, want it to contain %q", name, err, test.want)
		}
		if err != nil && strings.Contains(err.Error(), "\n") {
			t.Errorf("%s: a load error is one line: %q", name, err)
		}
	}
	var located *LocatedError
	if _, err := loadToolsPlugin(`{enabled: true, select: {mode: add, tools_db_path: ` + twice + `}}`); !errors.As(err, &located) || located.Code != CodeDuplicateToolName {
		t.Fatalf("a tool listed twice is a located error: %v", err)
	}
}

// A catalog of any size loads, since the judge truncates what it reads; a
// catalog of which the policy allows no tool selects nothing.
func TestToolCatalogOfAnySizeLoads(t *testing.T) {
	names := numberedTools(300)
	catalog := writeToolCatalog(t, names...)
	if _, err := loadToolsPlugin(`{enabled: true, select: {mode: add, tools_db_path: ` + catalog + `}}`); err != nil {
		t.Fatalf("a catalog of 300 tools loads: %v", err)
	}
	if _, err := loadToolsPlugin(`{enabled: true, policy: {mode: filtered, allow_tools: [` + strings.Join(names[:10], ", ") + `]}, select: {mode: add, tools_db_path: ` + catalog + `}}`); err != nil {
		t.Fatalf("the policy allows 10 of the catalog's tools: %v", err)
	}
	if _, err := loadToolsPlugin(`{enabled: true, policy: {mode: filtered, allow_tools: [missing]}, select: {mode: add, tools_db_path: ` + catalog + `}}`); err == nil || !strings.Contains(err.Error(), "has no tool the policy allows") {
		t.Fatalf("err = %v", err)
	}
	if _, err := loadToolsPlugin(`{enabled: true, select: {mode: add, tools_db_path: ` + filepath.Join(t.TempDir(), "absent.json") + `}}`); err != nil {
		t.Fatalf("a catalog not there to read is checked when the Router reads it: %v", err)
	}
}

func TestLoadToolCatalog(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tools_db.json")
	write := func(content string) {
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(`[{"tool": {"type": "function", "function": {"name": " get_weather ", "description": "Weather now", "strict": true,
		"parameters": {"type": "object", "properties": {"days": {"type": "integer", "maximum": 9007199254740993}}}}}, "description": "Current weather for a city"},
		{"tool": {"function": {"name": "search"}}}]`)
	catalog, err := LoadToolCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(catalog) != 2 || catalog[0].Tool.Name != "get_weather" || catalog[0].Description != "Current weather for a city" || catalog[0].Tool.Strict == nil || !*catalog[0].Tool.Strict {
		t.Fatalf("catalog %+v", catalog)
	}
	if !strings.Contains(string(catalog[0].Tool.InputSchema), "9007199254740993") {
		t.Fatalf("the schema keeps its numbers as written: %s", catalog[0].Tool.InputSchema)
	}
	if catalog[1].Description != "" || string(catalog[1].Tool.InputSchema) != `{"type":"object"}` {
		t.Fatalf("a tool without parameters takes an object schema: %+v", catalog[1])
	}
	for content, want := range map[string]string{
		`[{"tool": {"type": "custom", "function": {"name": "a"}}}]`:                      "tool type must be function",
		`[{"tool": {"function": {"name": " "}}}]`:                                        "function name must have 1 to 256 characters",
		`[{"tool": {"function": {"name": "a", "parameters": [1]}}}]`:                     "parameters must be a JSON Schema object",
		`[{"tool": {"function": {"name": "a"}}}, {"tool": {"function": {"name": "a"}}}]`: `tool "a" is listed twice`,
	} {
		write(content)
		if _, err := LoadToolCatalog(path); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err = %v, want %q", content, err, want)
		}
	}
}
