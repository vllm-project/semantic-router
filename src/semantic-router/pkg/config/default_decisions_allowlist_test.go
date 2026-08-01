package config

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// defaultDecisionsAllowlist is the complete set of production files that may
// reference the flat DefaultDecisions field. The field holds only the default
// routing profile; code that reasons about routing as a whole must use
// AllRoutingDecisions() or GetDecisionByName() instead (#2723 catalogues the
// read sites that silently ignored named recipes by picking this field).
//
// To add a file here, first decide the scope question the entry comment must
// answer: why is the default profile — and not the whole routing surface —
// the right thing for that site to read or write?
var defaultDecisionsAllowlist = map[string]string{
	// Field declaration.
	"src/semantic-router/pkg/config/config.go": "declares the field",

	// Loaders and serializers: the field IS their storage for the top-level
	// routing: block, kept aliased with the default recipe.
	"src/semantic-router/pkg/config/canonical_config.go":         "loader writes the normalized top-level routing block",
	"src/semantic-router/pkg/config/canonical_recipes.go":        "loader keeps the field and the default recipe aliased",
	"src/semantic-router/pkg/config/canonical_routing_loader.go": "fragment loader writes the top-level routing block",
	"src/semantic-router/pkg/config/canonical_export.go":         "export re-emits the field as the top-level routing block",
	"src/semantic-router/pkg/dsl/compiler_routes.go":             "DSL compiler builds the default profile",
	"src/semantic-router/pkg/dsl/decompiler.go":                  "DSL decompiles the default profile; named recipes have no DSL surface yet",
	"src/semantic-router/pkg/dsl/decompiler_decisions.go":        "DSL decompiles the default profile; named recipes have no DSL surface yet",
	"src/semantic-router/pkg/dsl/routing_contract.go":            "DSL contract walks the default profile it decompiles",

	// Whole-surface accessors implemented inside pkg/config: they treat the
	// flat field as the default recipe's storage and add the named recipes.
	"src/semantic-router/pkg/config/recipes.go": "AllRoutingDecisions/HasRoutingDecisions fall back to the field for configs built without the canonical loader",
	"src/semantic-router/pkg/config/helper.go":  "findDecisionBy scans the field as the default-profile portion; GetModelForDecisionIndex is by-index, only meaningful within one profile",

	// Deliberate default-profile reads. Algorithm virtual slugs select
	// decisions without consulting the entrypoint table, so widening them
	// would let a recipe's decisions be selected without going through that
	// recipe's entrypoint; HasXDecision gates whether /v1/models advertises
	// the slug, and advertising one that cannot route would be a lie.
	"src/semantic-router/pkg/config/fusion_config.go":                       "HasFusionDecision gates the fusion slug in /v1/models",
	"src/semantic-router/pkg/config/remom_config.go":                        "HasReMoMDecision gates the remom slug in /v1/models",
	"src/semantic-router/pkg/config/workflows_config.go":                    "HasFlowDecision gates the flow slug in /v1/models",
	"src/semantic-router/pkg/extproc/req_filter_fusion.go":                  "algorithm slugs resolve against the default profile only",
	"src/semantic-router/pkg/extproc/req_filter_classification_runtime.go":  "authz scope for requests without an entrypoint recipe",
	"src/semantic-router/pkg/extproc/router_selection.go":                   "first-match into the process-level selector singleton must not depend on recipe declaration order",
	"src/semantic-router/pkg/classification/classifier_signal_decision.go":  "default candidate set for unscoped requests",
	"src/semantic-router/pkg/classification/classifier_category_entropy.go": "reasoning maps serve the classify API, same scope as its decision engine",
	"src/semantic-router/pkg/classification/mcp_classifier_runtime.go":      "reasoning map serves the classify API, same scope as its decision engine",
	"src/semantic-router/pkg/services/classification.go":                    "gate matches EvaluateDecisionWithEngine's default scope",
	"src/semantic-router/pkg/services/classification_signal_contract.go":    "gate matches EvaluateDecisionWithEngine's default scope",
	"src/semantic-router/pkg/services/classification_recommendation.go":     "recommendation must not name a model only a recipe entrypoint can reach",

	// Known limitation: the dashboard topology test-query panel replays the
	// default profile's rules and does not model recipe scoping yet.
	"dashboard/backend/handlers/topology_response.go": "topology test-query panel does not model recipe scoping yet",
}

// TestDefaultDecisionsReadSitesAreAllowlisted walks every production Go file
// in the repository (all modules) and fails when a file references the flat
// DefaultDecisions field without an allowlist entry, or when an entry goes
// stale. A new site must either switch to a whole-surface accessor or state
// its scope rationale here, in front of the reviewer.
func TestDefaultDecisionsReadSitesAreAllowlisted(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatalf("resolving repo root: %v", err)
	}
	if _, statErr := os.Stat(filepath.Join(repoRoot, ".git")); statErr != nil {
		t.Fatalf("repo root %s does not look like a git checkout: %v", repoRoot, statErr)
	}

	found := scanDefaultDecisionsReferences(t, repoRoot)

	var offenders, stale []string
	for rel := range found {
		if _, ok := defaultDecisionsAllowlist[rel]; !ok {
			offenders = append(offenders, rel)
		}
	}
	for rel := range defaultDecisionsAllowlist {
		if !found[rel] {
			stale = append(stale, rel)
		}
	}
	sort.Strings(offenders)
	sort.Strings(stale)

	if len(offenders) > 0 {
		t.Errorf("files reference the flat DefaultDecisions field without an allowlist entry:\n  %s\n\n"+
			"DefaultDecisions holds only the default routing profile. If the site reasons about routing as a whole, "+
			"use AllRoutingDecisions() or GetDecisionByName(); if the default-profile scope is intended, add the file "+
			"to defaultDecisionsAllowlist with a comment saying why.", strings.Join(offenders, "\n  "))
	}
	if len(stale) > 0 {
		t.Errorf("stale defaultDecisionsAllowlist entries (file gone or no longer references the field), remove them:\n  %s",
			strings.Join(stale, "\n  "))
	}
}

// scanDefaultDecisionsReferences returns the repo-relative paths of every
// production Go file, across all modules, that mentions DefaultDecisions.
func scanDefaultDecisionsReferences(t *testing.T, repoRoot string) map[string]bool {
	t.Helper()
	found := map[string]bool{}
	walkErr := filepath.WalkDir(repoRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			skip := (strings.HasPrefix(d.Name(), ".") && path != repoRoot) || d.Name() == "node_modules"
			if skip {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		content, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		if !strings.Contains(string(content), "DefaultDecisions") {
			return nil
		}
		rel, relErr := filepath.Rel(repoRoot, path)
		if relErr != nil {
			return relErr
		}
		found[filepath.ToSlash(rel)] = true
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walking repo: %v", walkErr)
	}
	return found
}
