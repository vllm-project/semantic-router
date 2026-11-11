package config

import (
	"bytes"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Codes of the one-line load errors and warnings. The removed_* codes,
// invalid_value and the two renamed_* codes name forms the Router no longer
// reads as written, which `vllm-sr config migrate` rewrites; the others are
// checks of a canonical configuration.
const (
	CodeRemovedSignalType        = "removed_signal_type"
	CodeRemovedAlgorithmType     = "removed_algorithm_type"
	CodeRemovedPluginType        = "removed_plugin_type"
	CodeRemovedField             = "removed_field"
	CodeRemovedSection           = "removed_section"
	CodeInvalidValue             = "invalid_value"
	CodeRenamedSignalType        = "renamed_signal_type"
	CodeRenamedAlgorithmType     = "renamed_algorithm_type"
	CodeReservedHeader           = "reserved_header"
	CodeSpanRequired             = "span_required"
	CodeInputModalityUnsupported = "input_modality_unsupported"
	CodeMatchEmpty               = "match_empty"
	CodeOutputCapabilityMissing  = "output_capability_missing"
	CodeOutputNotExecutable      = "output_not_executable"
	CodeDuplicateToolName        = "duplicate_tool_name"
	// CodeUncalibratedPreset warns of a preset asked with the neutral
	// threshold, because the answering model has no calibrated one.
	CodeUncalibratedPreset = "uncalibrated_preset"
)

// LocatedError is a one-line load error that names its own place in the
// document, so no caller adds another prefix to it.
type LocatedError struct {
	Path, Code, Message string
}

func (e *LocatedError) Error() string {
	return e.Path + ": " + e.Code + ": " + e.Message
}

// FormError is the load error of a form the Router no longer reads:
// `<path>: <code>: <what>; run `vllm-sr config migrate` (<becomes>)`.
func FormError(path, code, what, becomes string) error {
	return &LocatedError{Path: path, Code: code, Message: migrateHint(what, becomes)}
}

// FormWarning is the load warning of a renamed form, which loads as its
// canonical form.
func FormWarning(path, code, what, becomes string) ConfigWarning {
	return ConfigWarning{Code: code, Field: path, Message: migrateHint(what, becomes)}
}

// CheckError is the load error of a canonical configuration that a check
// refuses: `<path>: <code>: <message>`.
func CheckError(path, code, message string) error {
	return &LocatedError{Path: path, Code: code, Message: message}
}

func migrateHint(what, becomes string) string {
	return what + "; run `vllm-sr config migrate` (" + becomes + ")"
}

// ConfigForm is a form of the configuration the Router no longer reads as
// written. A removed form fails the load; a renamed one (only decision ->
// systemone) loads as its canonical form, with a warning. `vllm-sr config
// migrate` rewrites both with Migrate.
type ConfigForm struct {
	Code string
	// Paths are patterns over the raw document: dotted mapping keys, `name[]`
	// for every item of the sequence under name, `*` for any key and `**`
	// for any depth. RoutingScopes builds the patterns of every recipe.
	Paths []string
	// Match, when set, keeps only the matches it accepts; MatchAt does so for
	// a form whose matches depend on the document around them, such as the
	// models a decision references.
	Match   func(node *yaml.Node, path string) bool
	MatchAt func(at FormMatch) bool
	// What was removed or renamed, and what it becomes. Describe, when set,
	// words them for one match.
	What, Becomes string
	Describe      func(at FormMatch) (what, becomes string)
	Renamed       bool
	// Whole refuses the document with this form's line alone: the document is
	// written in a layout no other form's paths describe.
	Whole bool
	// Migrate rewrites one match in the form that replaces it; nil when the
	// form has no mechanical rewrite, which migrate then refuses with Becomes.
	Migrate func(m *Migration, at FormMatch) error
}

// FormMatch is one place a form occurs: the node at Path, held by Parent
// under Key (a mapping) or at Index (a sequence). Root is the document's
// top-level mapping, for a rewrite that reaches past the match (the signals
// a rewritten plugin reads); Ancestors hold Parent, from the root down.
type FormMatch struct {
	Path      string
	Parent    *yaml.Node
	Key       string
	Index     int
	Node      *yaml.Node
	Root      *yaml.Node
	Ancestors []*yaml.Node
}

// Above is the node levels above the match: 1 is Parent, 2 the node that
// holds it, and so on up to the document root; nil past the root.
func (at FormMatch) Above(levels int) *yaml.Node {
	if levels == 1 {
		return at.Parent
	}
	index := len(at.Ancestors) - (levels - 1)
	if levels < 1 || index < 0 {
		return nil
	}
	return at.Ancestors[index]
}

// configForms lists every removed and renamed form, in the order migrate
// rewrites them, as the concatenation of the per-feature lists: the layouts
// the previous release migrated, which migrate refuses before rewriting
// anything, then the v0.3 layout, the global feature switches and last the
// Looper model names, whose recipe copies the default recipe as migrated.
var configForms = slices.Concat(previousReleaseForms, structureForms, legacyRouterForms, systemOneForms, listenerForms, pluginForms, applicationForms, heuristicForms, algorithmForms, toolsForms, contextForms, responseCacheForms, routerReplayForms, retentionForms, candidateIterationForms, embeddingForms, multimodalForms, modelRuntimeForms, modelResourceForms, switchForms, looperModelNameForms)

// RoutingScopes returns the patterns of suffix in every recipe of a v0.5
// document. Forms match v0.5 paths only: migrate rewrites the v0.3 structure
// first (structureForms), and the Router refuses a v0.3 document by its
// version.
func RoutingScopes(suffix string) []string {
	return []string{"recipes[]." + suffix}
}

// ScopePath is the document path of rest in the recipe of c, for a one-line
// error of a check that runs per recipe; an unscoped config is its default
// recipe.
func (c *RouterConfig) ScopePath(rest string) string {
	scope := c.defaultRecipeName()
	if c != nil && c.RoutingScope != "" {
		scope = c.RoutingScope
	}
	return fmt.Sprintf("recipes[%s].%s", scope, rest)
}

func (f ConfigForm) describe(at FormMatch) (string, string) {
	if f.Describe != nil {
		return f.Describe(at)
	}
	return f.What, f.Becomes
}

// matches returns where the form occurs in a document, in document order.
func (f ConfigForm) matches(root *yaml.Node) []FormMatch {
	var found []FormMatch
	seen := map[*yaml.Node]bool{}
	for _, pattern := range f.Paths {
		for _, at := range matchPattern(root, strings.Split(pattern, ".")) {
			if seen[at.Node] || (f.Match != nil && !f.Match(at.Node, at.Path)) || (f.MatchAt != nil && !f.MatchAt(at)) {
				continue
			}
			seen[at.Node] = true
			found = append(found, at)
		}
	}
	slices.SortStableFunc(found, func(a, b FormMatch) int {
		if a.Node.Line != b.Node.Line {
			return a.Node.Line - b.Node.Line
		}
		return a.Node.Column - b.Node.Column
	})
	return found
}

func matchPattern(root *yaml.Node, segments []string) []FormMatch {
	var found []FormMatch
	var walk func(node *yaml.Node, rest []string, at FormMatch)
	walk = func(node *yaml.Node, rest []string, at FormMatch) {
		for node != nil && node.Kind == yaml.AliasNode {
			node = node.Alias
		}
		if node == nil {
			return
		}
		at.Node, at.Root = node, root
		if len(rest) == 0 {
			found = append(found, at)
			return
		}
		// The nodes that hold node, which hold its children's parent.
		holders := at.Ancestors
		if at.Parent != nil {
			holders = append(slices.Clip(holders), at.Parent)
		}
		segment := rest[0]
		switch {
		case segment == "**":
			walk(node, rest[1:], at)
			forEachChild(node, at.Path, func(child FormMatch) {
				child.Ancestors = holders
				walk(child.Node, rest, child)
			})
		case segment == "*":
			for i := 0; node.Kind == yaml.MappingNode && i+1 < len(node.Content); i += 2 {
				key := node.Content[i].Value
				walk(node.Content[i+1], rest[1:], FormMatch{Path: joinPath(at.Path, key), Parent: node, Key: key, Ancestors: holders})
			}
		case strings.HasSuffix(segment, "[]"):
			key := strings.TrimSuffix(segment, "[]")
			items := mappingValue(node, key)
			for items != nil && items.Kind == yaml.AliasNode {
				items = items.Alias
			}
			if items == nil || items.Kind != yaml.SequenceNode {
				return
			}
			base := joinPath(at.Path, key)
			within := append(slices.Clip(holders), node)
			for i, item := range items.Content {
				walk(item, rest[1:], FormMatch{Path: base + "[" + itemLabel(item, i) + "]", Parent: items, Index: i, Ancestors: within})
			}
		default:
			if value := mappingValue(node, segment); value != nil {
				walk(value, rest[1:], FormMatch{Path: joinPath(at.Path, segment), Parent: node, Key: segment, Ancestors: holders})
			}
		}
	}
	walk(root, segments, FormMatch{})
	return found
}

// forEachChild visits the values of a mapping and the items of a sequence.
func forEachChild(node *yaml.Node, path string, visit func(FormMatch)) {
	switch node.Kind {
	case yaml.MappingNode:
		for i := 0; i+1 < len(node.Content); i += 2 {
			key := node.Content[i].Value
			visit(FormMatch{Path: joinPath(path, key), Parent: node, Key: key, Node: node.Content[i+1]})
		}
	case yaml.SequenceNode:
		for i, item := range node.Content {
			visit(FormMatch{Path: path + "[" + itemLabel(item, i) + "]", Parent: node, Index: i, Node: item})
		}
	}
}

// itemLabel names a sequence item in a path by its name, else its index.
func itemLabel(item *yaml.Node, index int) string {
	if name := scalarAt(item, "name"); name != "" {
		return name
	}
	return strconv.Itoa(index)
}

func documentRoot(document *yaml.Node) *yaml.Node {
	if document.Kind != yaml.DocumentNode || len(document.Content) == 0 || document.Content[0].Kind != yaml.MappingNode {
		return nil
	}
	return document.Content[0]
}

// applyConfigForms reads a document's removed and renamed forms when the
// Router loads it. Every removed form fails the load, one line each. Each
// renamed form is rewritten to its canonical form, which is what the Router
// then reads, and reported as a warning. A document with neither is returned
// as it is.
func applyConfigForms(data []byte, forms []ConfigForm) ([]byte, []ConfigWarning, error) {
	if len(forms) == 0 {
		return data, nil, nil
	}
	var document yaml.Node
	if yaml.Unmarshal(data, &document) != nil {
		return data, nil, nil // the decoder reports the syntax error
	}
	root := documentRoot(&document)
	if root == nil {
		return data, nil, nil
	}
	var failures []error
	var warnings []ConfigWarning
	m := &Migration{root: root}
	renamed := false
	for _, form := range forms {
		for _, at := range form.matches(root) {
			what, becomes := form.describe(at)
			if form.Whole {
				return nil, nil, FormError(at.Path, form.Code, what, becomes)
			}
			if !form.Renamed {
				failures = append(failures, FormError(at.Path, form.Code, what, becomes))
				continue
			}
			warnings = append(warnings, FormWarning(at.Path, form.Code, what, becomes))
			if err := form.Migrate(m, at); err != nil {
				return nil, nil, fmt.Errorf("%s: %w", at.Path, err)
			}
			renamed = true
		}
	}
	if len(failures) > 0 {
		return nil, nil, errors.Join(failures...)
	}
	if !renamed {
		return data, warnings, nil
	}
	canonical, err := encodeDocument(&document)
	if err != nil {
		return nil, nil, err
	}
	return canonical, warnings, nil
}

func encodeDocument(document *yaml.Node) ([]byte, error) {
	var out bytes.Buffer
	encoder := yaml.NewEncoder(&out)
	encoder.SetIndent(2)
	if err := encoder.Encode(document); err != nil {
		return nil, err
	}
	if err := encoder.Close(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// CanonicalConfigBytes returns a document as the Router reads it, for an
// editor that decodes it into the canonical types: each renamed form in its
// canonical form, and an error for a removed one.
func CanonicalConfigBytes(data []byte) ([]byte, error) {
	canonical, _, err := applyConfigForms(data, configForms)
	return canonical, err
}
