package headers

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestWarningCodes(t *testing.T) {
	// Assert the constants directly — a map keyed by the constant would
	// compare each value to itself and never fail.
	for _, tt := range []struct{ got, want string }{
		{WarningHallucination, "hallucination"},
		{WarningUnverifiedFactual, "unverified_factual"},
		{WarningJailbreak, "response_jailbreak"},
		{WarningGuard, "response_guard"},
		{WarningProtocol, "protocol"},
		{WarningTruncated, "truncated"},
	} {
		if tt.got != tt.want {
			t.Errorf("warning code = %q, want %q", tt.got, tt.want)
		}
	}
}

func TestWarningCodesDocumentEachCode(t *testing.T) {
	want := []string{WarningScanPartial, WarningHallucination, WarningUnverifiedFactual, WarningJailbreak, WarningGuard, WarningProtocol, WarningTruncated}
	var got []string
	for _, code := range WarningCodes() {
		got = append(got, code.Code)
		if !tableSafeSentence(code.Doc) {
			t.Errorf("%s: doc %q must be one sentence that a Markdown table and MDX can hold", code.Code, code.Doc)
		}
	}
	if !slices.Equal(got, want) {
		t.Errorf("WarningCodes() = %v, want %v", got, want)
	}
}

// Each x-vsr-response-path value is a ResponsePath constant that Router code
// outside this package sends, and the registry documents exactly those
// values, so the header never promises a value no response carries.
func TestResponsePathValuesAreSentAndDocumented(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "headers.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	values := map[string]string{}
	ast.Inspect(file, func(node ast.Node) bool {
		spec, ok := node.(*ast.ValueSpec)
		if !ok {
			return true
		}
		for i, name := range spec.Names {
			if strings.HasPrefix(name.Name, "ResponsePath") {
				values[name.Name], _ = strconv.Unquote(spec.Values[i].(*ast.BasicLit).Value)
			}
		}
		return false
	})
	header, _ := Lookup(VSRResponsePath)
	var documented []string
	for _, match := range regexp.MustCompile("`([a-z_]+)`").FindAllStringSubmatch(header.Doc, -1) {
		documented = append(documented, match[1])
	}
	declared := slices.Sorted(maps.Values(values))
	slices.Sort(documented)
	if !slices.Equal(documented, declared) {
		t.Errorf("%s documents %v, the ResponsePath constants are %v", VSRResponsePath, documented, declared)
	}
	sent := map[string]bool{}
	module := os.DirFS(filepath.Join("..", ".."))
	err = fs.WalkDir(module, ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			if err == nil && path != "." && (path == "pkg/headers" || strings.HasPrefix(entry.Name(), ".")) {
				return fs.SkipDir
			}
			return err
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		source, err := fs.ReadFile(module, path)
		for name := range values {
			sent[name] = sent[name] || bytes.Contains(source, []byte("headers."+name))
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, value := range values {
		if !sent[name] {
			t.Errorf("%s = %q is a documented %s value that no Router code sends", name, value, VSRResponsePath)
		}
	}
}
