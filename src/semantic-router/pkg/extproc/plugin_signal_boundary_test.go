package extproc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// legacyClassifierEntryPoints are the classifier modules' own entry points:
// the PII token classifier, the prompt guard and the hallucination detector.
// Plugins act on named signals, so no plugin package calls them; the signals
// (classification.ResponseConditions, Classifier.PIIRuleSpans) reach the
// modules, and replacing a module changes no plugin.
var legacyClassifierEntryPoints = map[string]bool{
	"ClassifyPII":                         true,
	"ClassifyPIIWithThreshold":            true,
	"ClassifyPIIWithDetails":              true,
	"ClassifyPIIWithDetailsAndThreshold":  true,
	"AnalyzeContentForPII":                true,
	"AnalyzeContentForPIIWithThreshold":   true,
	"IsPIIEnabled":                        true,
	"CheckForJailbreak":                   true,
	"CheckForJailbreakWithThreshold":      true,
	"CheckForJailbreakWithRisk":           true,
	"CheckForJailbreakRiskWithThreshold":  true,
	"CheckForJailbreakVerdict":            true,
	"ScanJailbreak":                       true,
	"ScanJailbreakRisk":                   true,
	"IsJailbreakEnabled":                  true,
	"DetectHallucination":                 true,
	"DetectHallucinationWithExplanations": true,
	"IsHallucinationDetectionEnabled":     true,
	"IsHallucinationDetectorReady":        true,
}

// pluginPackages are the packages that run decision plugins, relative to this
// one.
var pluginPackages = []string{".", "../pluginruntime", "../masking"}

func TestPluginsReachClassifierModulesOnlyThroughSignals(t *testing.T) {
	fset := token.NewFileSet()
	for _, dir := range pluginPackages {
		paths, err := filepath.Glob(filepath.Join(dir, "*.go"))
		if err != nil {
			t.Fatal(err)
		}
		if len(paths) == 0 {
			t.Fatalf("no Go files in %s", dir)
		}
		for _, path := range paths {
			if strings.HasSuffix(path, "_test.go") {
				continue
			}
			file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
			if err != nil {
				t.Fatal(err)
			}
			ast.Inspect(file, func(node ast.Node) bool {
				if selector, ok := node.(*ast.SelectorExpr); ok && legacyClassifierEntryPoints[selector.Sel.Name] {
					t.Errorf("%s: %s is a classifier module's own entry point; a plugin acts on a named signal", fset.Position(selector.Pos()), selector.Sel.Name)
				}
				return true
			})
		}
	}
}
