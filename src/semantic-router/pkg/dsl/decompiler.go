package dsl

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Decompile converts runtime config into the DSL contract.

func Decompile(cfg *config.RouterConfig) (string, error) {
	return DecompileConfig(cfg)
}

func DecompileToAST(cfg *config.RouterConfig) *Program {
	if err := config.ValidateDecisionRuleLimits(cfg); err != nil {
		return nil
	}
	prog := DecompileRoutingToAST(topLevelView(cfg))
	appendConfigScopesToAST(prog, cfg)
	return prog
}

type decompiler struct {
	cfg             *config.RouterConfig
	sb              strings.Builder
	pluginTemplates map[string]*pluginTemplate // auto-extracted templates
}

type pluginTemplate struct {
	name       string
	pluginType string
	usageCount int
}

func (d *decompiler) extractPluginTemplates() {
	// Count plugin usage across decisions to find repeated plugins
	type pluginKey struct {
		pluginType string
	}
	seen := make(map[pluginKey]*pluginTemplate)

	for _, dec := range d.cfg.Decisions {
		for _, p := range dec.DeclaredPlugins() {
			key := pluginKey{pluginType: p.Type}
			// Use a simple fingerprint: type
			if _, exists := seen[key]; !exists {
				name := sanitizeName(p.Type)
				seen[key] = &pluginTemplate{
					name:       name,
					pluginType: p.Type,
					usageCount: 1,
				}
			} else {
				seen[key].usageCount++
			}
		}
	}

	// Only extract templates that are used 2+ times
	for _, tmpl := range seen {
		if tmpl.usageCount >= 2 {
			d.pluginTemplates[tmpl.pluginType] = tmpl
		}
	}
}

func (d *decompiler) decompilePluginTemplates() {
	// Sort by plugin type for deterministic output
	keys := sortedKeys(d.pluginTemplates)
	for _, key := range keys {
		tmpl := d.pluginTemplates[key]
		d.write("PLUGIN %s %s {}\n\n", tmpl.name, sanitizeName(tmpl.pluginType))
	}
}

func (d *decompiler) write(format string, args ...interface{}) {
	fmt.Fprintf(&d.sb, format, args...)
}

func (d *decompiler) writeSection(name string) {
	d.write("# =============================================================================\n")
	d.write("# %s\n", name)
	d.write("# =============================================================================\n\n")
}

func formatStringArray(items []string) string {
	quoted := make([]string, len(items))
	for i, item := range items {
		quoted[i] = fmt.Sprintf("%q", item)
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

func stringsToArray(items []string) ArrayValue {
	vals := make([]Value, len(items))
	for i, s := range items {
		vals[i] = StringValue{V: s}
	}
	return ArrayValue{Items: vals}
}

func Format(input string) (string, error) {
	return FormatWithLimits(input, config.DecisionRuleLimits{})
}

// FormatWithLimits keeps the enclosing budget during compile/decompile formatting.
func FormatWithLimits(input string, limits config.DecisionRuleLimits) (string, error) {
	prog, errs := Parse(input)
	if len(errs) > 0 {
		return "", fmt.Errorf("parse errors: %v", errs)
	}

	cfg, compileErrs := CompileASTWithLimits(prog, limits)
	if len(compileErrs) > 0 {
		return "", fmt.Errorf("compile errors: %v", compileErrs)
	}

	formatted, err := Decompile(cfg)
	if err != nil {
		return "", err
	}
	if len(prog.TestBlocks) == 0 {
		return formatted, nil
	}
	return appendFormattedTestBlocks(formatted, prog.TestBlocks), nil
}

func sanitizeName(name string) string {
	return strings.ReplaceAll(name, "-", "_")
}

func quoteName(name string) string {
	for _, ch := range name {
		if !isIdentPart(ch) {
			return fmt.Sprintf("%q", name)
		}
	}
	return name
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
