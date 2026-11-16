package extproc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"
)

// removedClassifierModules are the names of the classifier modules the
// decision model replaced: the PII token classifier masking ran, the prompt
// guard of the response jailbreak check, the hallucination detector, their
// mappings, backends and configuration. Every signal is answered by the
// decision model through its systemone rules, and plugins act on named
// signals (classification.ResponseConditions, Classifier.PIIRuleSpans), so
// no Router package names one of them again.
var removedClassifierModules = map[string]bool{
	"ClassifyPII":                         true,
	"ClassifyPIIWithThreshold":            true,
	"ClassifyPIIWithDetails":              true,
	"ClassifyPIIWithDetailsAndThreshold":  true,
	"AnalyzeContentForPII":                true,
	"AnalyzeContentForPIIWithThreshold":   true,
	"IsPIIEnabled":                        true,
	"IsPIIClassifierEnabled":              true,
	"PIIMapping":                          true,
	"LoadPIIMapping":                      true,
	"PIIModel":                            true,
	"CheckForJailbreak":                   true,
	"CheckForJailbreakWithThreshold":      true,
	"CheckForJailbreakWithRisk":           true,
	"CheckForJailbreakRiskWithThreshold":  true,
	"CheckForJailbreakVerdict":            true,
	"ScanJailbreak":                       true,
	"ScanJailbreakRisk":                   true,
	"IsJailbreakEnabled":                  true,
	"IsPromptGuardEnabled":                true,
	"JailbreakMapping":                    true,
	"LoadJailbreakMapping":                true,
	"PromptGuard":                         true,
	"PromptGuardConfig":                   true,
	"VLLMJailbreakInference":              true,
	"DetectHallucination":                 true,
	"DetectHallucinationWithExplanations": true,
	"IsHallucinationDetectionEnabled":     true,
	"IsHallucinationDetectorReady":        true,
	"HallucinationDetector":               true,
	"EndpointHallucinationDetector":       true,
	"NewHallucinationDetector":            true,
	"HallucinationMitigation":             true,
	"HallucinationModelConfig":            true,
	"HTTPClassifierInference":             true,
	"HTTPTokenClassifierInference":        true,
}

// routerRoot is the Router module, relative to this package.
const routerRoot = "../.."

func TestNoRouterPackageNamesARemovedClassifierModule(t *testing.T) {
	fset := token.NewFileSet()
	files := 0
	err := filepath.WalkDir(routerRoot, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if entry.Name() == "testdata" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		files++
		ast.Inspect(file, func(node ast.Node) bool {
			if ident, ok := node.(*ast.Ident); ok && removedClassifierModules[ident.Name] {
				t.Errorf("%s: %s names a removed classifier module; the decision model answers its signal", fset.Position(ident.Pos()), ident.Name)
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if files == 0 {
		t.Fatalf("no Go files under %s", routerRoot)
	}
}
