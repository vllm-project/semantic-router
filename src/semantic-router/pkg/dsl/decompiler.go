package dsl

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Decompile converts runtime config into the DSL contract.

func Decompile(cfg *config.RouterConfig) (string, error) {
	return DecompileRouting(cfg)
}

func DecompileToAST(cfg *config.RouterConfig) *Program {
	return DecompileRoutingToAST(cfg)
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
		for _, p := range dec.Plugins {
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
	prog, errs := Parse(input)
	if len(errs) > 0 {
		return "", fmt.Errorf("parse errors: %v", errs)
	}

	cfg, compileErrs := CompileAST(prog)
	if len(compileErrs) > 0 {
		return "", fmt.Errorf("compile errors: %v", compileErrs)
	}

	formatted, err := DecompileRouting(cfg)
	if err != nil {
		return "", err
	}
	if len(prog.TestBlocks) == 0 {
		return formatted, nil
	}
	return appendFormattedTestBlocks(formatted, prog.TestBlocks), nil
}

func (d *decompiler) decompileEmit(e config.EmitDirective) {
	if e.Kind == "" {
		return
	}
	switch e.Kind {
	case "retention":
		if e.Retention == nil {
			d.write("  EMIT retention {}\n")
			return
		}
		r := e.Retention
		lines := make([]string, 0, 4)
		if r.Drop != nil {
			lines = append(lines, fmt.Sprintf("    drop: %t", *r.Drop))
		}
		if r.TTLTurns != nil {
			lines = append(lines, fmt.Sprintf("    ttl_turns: %d", *r.TTLTurns))
		}
		if r.KeepCurrentModel != nil {
			lines = append(lines, fmt.Sprintf("    keep_current_model: %t", *r.KeepCurrentModel))
		}
		if r.PreferPrefixRetention != nil {
			lines = append(lines, fmt.Sprintf("    prefer_prefix_retention: %t", *r.PreferPrefixRetention))
		}
		if len(lines) == 0 {
			d.write("  EMIT retention {}\n")
			return
		}
		d.write("  EMIT retention {\n%s\n  }\n", strings.Join(lines, "\n"))
	default:
		// Unknown kind: preserve as an empty block so validator surfaces the error.
		d.write("  EMIT %s {}\n", sanitizeName(e.Kind))
	}
}

func emitDirectiveToDecl(e config.EmitDirective) *EmitDecl {
	decl := &EmitDecl{Kind: e.Kind}
	if e.Kind == "retention" && e.Retention != nil {
		decl.Retention = &RetentionDirective{
			Drop:                  clonePtrBoolDSL(e.Retention.Drop),
			TTLTurns:              clonePtrIntDSL(e.Retention.TTLTurns),
			KeepCurrentModel:      clonePtrBoolDSL(e.Retention.KeepCurrentModel),
			PreferPrefixRetention: clonePtrBoolDSL(e.Retention.PreferPrefixRetention),
		}
	}
	return decl
}

func clonePtrBoolDSL(p *bool) *bool {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func clonePtrIntDSL(p *int) *int {
	if p == nil {
		return nil
	}
	v := *p
	return &v
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
