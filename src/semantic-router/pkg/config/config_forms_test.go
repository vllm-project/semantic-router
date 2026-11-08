package config

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const formsDocument = `version: v0.5
recipes:
  - name: default
    signals:
      old: [{name: first}, {name: second}]
    decisions:
      - name: route
        rules:
          operator: AND
          conditions:
            - {type: old, name: first}
            - operator: OR
              conditions: [{type: old, name: second}, {type: keyword, name: kw}]
  - name: fast
    signals:
      old: [{name: third}]
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
`

func parseFormsDocument(t *testing.T, data string) *yaml.Node {
	t.Helper()
	var document yaml.Node
	if err := yaml.Unmarshal([]byte(data), &document); err != nil {
		t.Fatal(err)
	}
	return documentRoot(&document)
}

func matchedPaths(form ConfigForm, root *yaml.Node) []string {
	var paths []string
	for _, at := range form.matches(root) {
		paths = append(paths, at.Path)
	}
	return paths
}

// Patterns name sequence items by their name, reach every routing scope,
// and `**` reaches nested rule trees; matches come in document order.
func TestConfigFormPatternsMatchEveryPlace(t *testing.T) {
	root := parseFormsDocument(t, formsDocument)
	collections := ConfigForm{Paths: RoutingScopes("signals.old")}
	if got := matchedPaths(collections, root); !slices.Equal(got, []string{"recipes[default].signals.old", "recipes[fast].signals.old"}) {
		t.Fatalf("collections: %v", got)
	}
	items := ConfigForm{Paths: RoutingScopes("signals.old[]")}
	if got := matchedPaths(items, root); !slices.Equal(got, []string{"recipes[default].signals.old[first]", "recipes[default].signals.old[second]", "recipes[fast].signals.old[third]"}) {
		t.Fatalf("items: %v", got)
	}
	conditions := ConfigForm{
		Paths: RoutingScopes("decisions[].rules.**.type"),
		Match: func(node *yaml.Node, _ string) bool { return node.Value == "old" },
	}
	if got := matchedPaths(conditions, root); !slices.Equal(got, []string{
		"recipes[default].decisions[route].rules.conditions[first].type",
		"recipes[default].decisions[route].rules.conditions[1].conditions[second].type",
	}) {
		t.Fatalf("conditions: %v", got)
	}
	anyKey := ConfigForm{Paths: []string{"recipes[].signals.*"}}
	if got := matchedPaths(anyKey, root); !slices.Equal(got, []string{"recipes[default].signals.old", "recipes[fast].signals.old"}) {
		t.Fatalf("any key: %v", got)
	}
}

// A match reaches the nodes that hold it, up to the document root, so a
// rewrite can edit the routing scope it lies in.
func TestConfigFormMatchesReachTheirHolders(t *testing.T) {
	root := parseFormsDocument(t, formsDocument)
	items := ConfigForm{Paths: RoutingScopes("signals.old[]")}.matches(root)
	if len(items) != 3 {
		t.Fatalf("items: %d", len(items))
	}
	first, third := items[0], items[2]
	recipes := mappingValue(root, "recipes")
	defaultRecipe, fast := recipes.Content[0], recipes.Content[1]
	if first.Above(1) != mappingValue(mappingAt(defaultRecipe, "signals"), "old") || first.Above(2) != mappingAt(defaultRecipe, "signals") ||
		first.Above(3) != defaultRecipe || first.Above(4) != recipes || first.Above(5) != root || first.Above(6) != nil || first.Above(0) != nil {
		t.Fatalf("default recipe holders: %+v", first.Ancestors)
	}
	if third.Above(3) != fast || third.Above(4) != recipes || third.Above(5) != root {
		t.Fatalf("recipe holders: %+v", third.Ancestors)
	}
	nested := ConfigForm{Paths: RoutingScopes("decisions[].rules.**"), Match: hasType("old")}.matches(root)[1]
	rules := mappingAt(sequenceAt(defaultRecipe, "decisions")[0], "rules")
	if nested.Above(2) != sequenceAt(rules, "conditions")[1] || nested.Above(4) != rules || nested.Above(7) != defaultRecipe {
		t.Fatalf("nested condition holders: %s %+v", nested.Path, nested.Ancestors)
	}
}

func renamedOldForm() ConfigForm {
	return ConfigForm{
		Code: CodeRenamedSignalType, Paths: RoutingScopes("signals.old"), Renamed: true,
		What: "the old signal type was renamed", Becomes: "new",
		Migrate: func(m *Migration, at FormMatch) error { return m.Rename(at.Parent, "old", "new") },
	}
}

func removedOldForm() ConfigForm {
	return ConfigForm{Code: CodeRemovedSignalType, Paths: RoutingScopes("signals.old"), What: "the old signal type was removed", Becomes: "a new rule"}
}

// A removed form fails the load with one line per place it occurs; a
// renamed one is read as its canonical form, with a warning; a document
// with neither is read as written.
func TestApplyConfigFormsAtLoad(t *testing.T) {
	_, _, err := applyConfigForms([]byte(formsDocument), []ConfigForm{removedOldForm()})
	var located *LocatedError
	if !errors.As(err, &located) {
		t.Fatalf("a removed form fails the load: %v", err)
	}
	lines := strings.Split(err.Error(), "\n")
	want := "recipes[default].signals.old: removed_signal_type: the old signal type was removed; run `vllm-sr config migrate` (a new rule)"
	if len(lines) != 2 || lines[0] != want || !strings.HasPrefix(lines[1], "recipes[fast].signals.old: ") {
		t.Fatalf("one line per place:\n%s", err)
	}

	canonical, warnings, err := applyConfigForms([]byte(formsDocument), []ConfigForm{renamedOldForm()})
	if err != nil || len(warnings) != 2 || warnings[0].Field != "recipes[default].signals.old" || warnings[0].Code != CodeRenamedSignalType {
		t.Fatalf("renamed: %v %+v", err, warnings)
	}
	root := parseFormsDocument(t, string(canonical))
	recipe := mappingValue(root, "recipes").Content[0]
	if mappingAt(recipe, "signals", "old") != nil || len(sequenceAt(recipe, "signals", "new")) != 2 {
		t.Fatalf("the renamed form is read as the canonical one:\n%s", canonical)
	}

	same, warnings, err := applyConfigForms([]byte("version: v0.5\n"), []ConfigForm{removedOldForm(), renamedOldForm()})
	if err != nil || len(warnings) != 0 || string(same) != "version: v0.5\n" {
		t.Fatalf("a canonical document is read as written: %q %v %v", same, warnings, err)
	}
}

// Migrate rewrites every place, notes each form once, and refuses a form
// with no mechanical rewrite with what it becomes.
func TestMigrateConfigFormsRewritesEveryPlace(t *testing.T) {
	migrated, notes, err := migrateConfigForms([]byte(formsDocument), []ConfigForm{renamedOldForm()})
	if err != nil {
		t.Fatal(err)
	}
	root := parseFormsDocument(t, string(migrated))
	recipes := mappingValue(root, "recipes").Content
	if len(sequenceAt(recipes[0], "signals", "new")) != 2 || len(sequenceAt(recipes[1], "signals", "new")) != 1 {
		t.Fatalf("every place is rewritten:\n%s", migrated)
	}
	if len(notes) != 1 || notes[0].Code != migrationChangeCode || !strings.Contains(notes[0].Message, "(2 places)") {
		t.Fatalf("one note per form: %+v", notes)
	}
	if _, _, err := migrateConfigForms([]byte(formsDocument), []ConfigForm{removedOldForm()}); err == nil || !strings.Contains(err.Error(), "no mechanical rewrite: a new rule") {
		t.Fatalf("a form without a rewrite stops the migration: %v", err)
	}
}

// Renaming into a collection that exists joins the two; any other pair
// is refused.
func TestMigrationRenameJoinsCollections(t *testing.T) {
	root := parseFormsDocument(t, "signals:\n  old: [{name: a}]\n  new: [{name: b}]\n  scalar: x\n  other: y\n")
	signals := mappingValue(root, "signals")
	m := &Migration{}
	if err := m.Rename(signals, "old", "new"); err != nil || len(mappingValue(signals, "new").Content) != 2 || mappingValue(signals, "old") != nil {
		t.Fatalf("join: %v", err)
	}
	if err := m.Rename(signals, "scalar", "other"); err == nil {
		t.Fatal("two scalars under the old and new keys are refused")
	}
}

// A one-line error names its own place, so the recipe validation does not
// prefix it again; other errors keep the recipe.
func TestLocatedErrorsKeepTheirPath(t *testing.T) {
	located := CheckError((&RouterConfig{RoutingScope: "fast"}).ScopePath("signals.systemone[x]"), CodeSpanRequired, "resolves to deployment \"d\"")
	if got := wrapRoutingProfileValidationError("fast", located).Error(); got != `recipes[fast].signals.systemone[x]: span_required: resolves to deployment "d"` {
		t.Fatalf("located: %s", got)
	}
	if got := wrapRoutingProfileValidationError("fast", errors.New("plain")).Error(); got != `recipes[fast]: plain` {
		t.Fatalf("plain: %s", got)
	}
	if got := (&RouterConfig{}).ScopePath("signals.pii[p]"); got != "recipes[default].signals.pii[p]" {
		t.Fatalf("top-level scope: %s", got)
	}
}

// A DSL routing fragment reads the renamed forms as the full loader does.
func TestRoutingFragmentReadsRenamedForms(t *testing.T) {
	fragment := []byte(`recipes:
  - name: default
    signals:
      decision:
        - name: needs_tools
          question: {type: noul, instructions: "Does the request need a tool?"}
    decisions:
      - name: tools
        priority: 100
        rules: {type: decision, name: needs_tools}
        modelRefs: [{model: general}]
`)
	cfg, err := ParseRoutingYAMLBytes(fragment)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.SystemOneRules) != 1 || cfg.SystemOneRules[0].Name != "needs_tools" {
		t.Fatalf("systemone rules = %+v", cfg.SystemOneRules)
	}
	if cfg.Decisions[0].Rules.Type != SignalTypeSystemOne {
		t.Fatalf("condition type = %q", cfg.Decisions[0].Rules.Type)
	}
	if len(cfg.FormWarnings) != 2 {
		t.Fatalf("form warnings = %+v", cfg.FormWarnings)
	}
}
