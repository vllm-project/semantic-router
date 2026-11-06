package headers

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

var (
	headerNamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)
	ownerPattern      = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)
	versionPattern    = regexp.MustCompile(`^v0\.[1-9][0-9]*$`)
)

const everyDirection = FromClient | ToClient | ToProvider | FromProvider

func TestRegistryEntriesAreWellFormed(t *testing.T) {
	seen := make(map[string]bool)
	for _, header := range Registry() {
		if !headerNamePattern.MatchString(header.Name) {
			t.Errorf("%q is not a lowercase header name", header.Name)
		}
		if seen[header.Name] {
			t.Errorf("%s is registered twice", header.Name)
		}
		seen[header.Name] = true
		if header.Direction == 0 || header.Direction&^everyDirection != 0 {
			t.Errorf("%s: direction %d names no known hop", header.Name, header.Direction)
		}
		if !slices.Contains(surfaceOrder, header.Surface) {
			t.Errorf("%s: unknown surface %q", header.Name, header.Surface)
		}
		if !slices.Contains([]Sensitivity{SensitivityNone, SensitivityIdentity, SensitivitySecret}, header.Sensitivity) {
			t.Errorf("%s: unknown sensitivity %q", header.Name, header.Sensitivity)
		}
		if !ownerPattern.MatchString(header.Owner) {
			t.Errorf("%s: owner %q is not a feature name", header.Name, header.Owner)
		}
		if !versionPattern.MatchString(header.Version) {
			t.Errorf("%s: version %q is not a release such as v0.3", header.Name, header.Version)
		}
		if !tableSafeSentence(header.Doc) {
			t.Errorf("%s: doc %q must be one sentence that a Markdown table and MDX can hold", header.Name, header.Doc)
		}
	}
}

// tableSafeSentence keeps a doc renderable in a GFM table cell of an MDX page:
// one line ending in a period, no pipe even in code, and no markup or JSX
// outside code spans.
func tableSafeSentence(doc string) bool {
	if !strings.HasSuffix(doc, ".") || strings.ContainsAny(doc, "\n|") {
		return false
	}
	parts := strings.Split(doc, "`")
	if len(parts)%2 == 0 {
		return false
	}
	for i := 0; i < len(parts); i += 2 {
		if strings.ContainsAny(parts[i], "<>{}") {
			return false
		}
	}
	return true
}

// The response and provider-bound contract tests in pkg/extproc read these
// rules off the registry, so every entry has to obey them.
func TestRegistrySurfacesAgreeWithDirections(t *testing.T) {
	for _, header := range Registry() {
		onResponses := header.Direction&ToClient != 0
		if (header.Surface == SurfacePublic || header.Surface == SurfaceDebug) && !onResponses {
			t.Errorf("%s is on the %s surface but no response carries it", header.Name, header.Surface)
		}
		if header.Surface == SurfaceControl && header.Direction&FromClient == 0 {
			t.Errorf("%s is a control header but no request carries it", header.Name)
		}
		if header.Surface == SurfaceInternal && onResponses {
			t.Errorf("%s is internal but rides on responses", header.Name)
		}
		if header.Sensitivity != SensitivityNone && header.Surface != SurfaceInternal {
			t.Errorf("%s carries %s values, so it must be internal, not %s", header.Name, header.Sensitivity, header.Surface)
		}
	}
}

func TestLookupIgnoresLetterCase(t *testing.T) {
	header, ok := Lookup(" X-VSR-Debug ")
	if !ok || header.Name != VSRDebug {
		t.Fatalf("Lookup(X-VSR-Debug) = %+v, %t; want %s", header, ok, VSRDebug)
	}
	if _, ok := Lookup("x-vsr-not-a-header"); ok {
		t.Fatal("Lookup found an unregistered name")
	}
}

// Every header constant of this package is registered, and every registered
// name is one of its constants, so a name cannot be added on one side only.
func TestRegistryCoversThePackageConstants(t *testing.T) {
	constants := packageStringConstants(t)
	for _, value := range constants {
		if !strings.HasPrefix(value, "x-") || strings.HasSuffix(value, "-") {
			continue
		}
		if _, ok := Lookup(value); !ok {
			t.Errorf("constant %q is not registered", value)
		}
	}
	for _, header := range Registry() {
		if !slices.Contains(constants, header.Name) {
			t.Errorf("registered name %q is no constant of this package", header.Name)
		}
	}
}

func packageStringConstants(t *testing.T) []string {
	t.Helper()
	fset := token.NewFileSet()
	matches, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var values []string
	for _, path := range matches {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range file.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.CONST {
				continue
			}
			for _, spec := range gen.Specs {
				for _, expr := range spec.(*ast.ValueSpec).Values {
					if literal, ok := expr.(*ast.BasicLit); ok && literal.Kind == token.STRING {
						value, err := strconv.Unquote(literal.Value)
						if err != nil {
							t.Fatal(err)
						}
						values = append(values, value)
					}
				}
			}
		}
	}
	return values
}

// Only this package spells header names: the rest of the module names a
// header through a constant registered here. That is stricter than accepting
// any literal that equals a registered name, and the registry tests above
// cover what it leaves out. Generated files and test-support files (non-test
// files that import testing) are not Router code.
func TestOnlyThisPackageSpellsHeaderNames(t *testing.T) {
	module := os.DirFS(filepath.Join("..", ".."))
	var findings []string
	err := fs.WalkDir(module, ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if path != "." && (path == "pkg/headers" || entry.Name() == "testdata" || strings.HasPrefix(entry.Name(), ".")) {
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		source, err := fs.ReadFile(module, path)
		if err != nil {
			return err
		}
		found, err := headerLiterals(path, source)
		findings = append(findings, found...)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) > 0 {
		t.Errorf("header names spelled outside pkg/headers; register them and use the constant:\n%s", strings.Join(findings, "\n"))
	}
}

func headerLiterals(path string, source []byte) ([]string, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, source, parser.ImportsOnly|parser.ParseComments)
	if err != nil {
		return nil, err
	}
	if ast.IsGenerated(file) || slices.ContainsFunc(file.Imports, func(spec *ast.ImportSpec) bool {
		return spec.Path.Value == strconv.Quote("testing")
	}) {
		return nil, nil
	}
	var lexer scanner.Scanner
	lexer.Init(fset.AddFile(path, -1, len(source)), source, nil, 0)
	var findings []string
	for {
		position, tok, literal := lexer.Scan()
		if tok == token.EOF {
			return findings, nil
		}
		if tok != token.STRING {
			continue
		}
		value, err := strconv.Unquote(literal)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", fset.Position(position), err)
		}
		if lower := strings.ToLower(value); strings.HasPrefix(lower, "x-") && !nonHeaderLiteral(lower) {
			findings = append(findings, fmt.Sprintf("%s: %q", fset.Position(position), value))
		}
	}
}

// nonHeaderLiteral reports the x- strings that are no header names.
func nonHeaderLiteral(value string) bool {
	switch {
	case value == "x-gzip": // a content coding
		return true
	case value == "x-vllm-sr", strings.HasPrefix(value, "x-vllm-sr-"): // OpenAPI and JSON Schema extension keys
		return true
	}
	return false
}

// The model catalog names the credential header each provider reads and the
// per-user key header an authenticator injects for it. Both are secrets.
func TestCatalogCredentialHeadersAreRegisteredSecrets(t *testing.T) {
	catalog, err := modelcatalog.BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	for _, provider := range catalog.Providers() {
		for _, name := range []string{provider.Auth.Header, provider.Auth.InjectedHeader} {
			if name == "" {
				continue
			}
			if header, ok := Lookup(name); !ok || header.Sensitivity != SensitivitySecret {
				t.Errorf("provider %s: credential header %q is not a registered secret", provider.ID, name)
			}
		}
	}
}

func TestEmbedReferenceReplacesOnlyTheMarkedBlock(t *testing.T) {
	page := []byte("intro\n" + ReferenceBegin + "\nstale\n" + ReferenceEnd + "\noutro\n")
	got, err := EmbedReference(page)
	if err != nil {
		t.Fatal(err)
	}
	want := "intro\n" + ReferenceBegin + "\n\n" + MarkdownReference() + "\n" + ReferenceEnd + "\noutro\n"
	if string(got) != want {
		t.Fatalf("EmbedReference() = %q, want %q", got, want)
	}
	if _, err := EmbedReference([]byte("no markers")); err == nil {
		t.Fatal("EmbedReference accepted a page without markers")
	}
}
