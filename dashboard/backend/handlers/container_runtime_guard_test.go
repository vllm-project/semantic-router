package handlers

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// The Dashboard holds no container runtime: `vllm-sr serve` owns the stack and
// applies every change that needs containers created anew.
var containerCLINames = []string{"crictl", "ctr", "docker", "nerdctl", "podman"}

// trapContainerCLIs puts container CLIs first on PATH that record each call
// and fail, and returns the calls made so far.
func trapContainerCLIs(t *testing.T) func() []string {
	t.Helper()
	directory := t.TempDir()
	calls := filepath.Join(directory, "calls.log")
	for _, name := range containerCLINames {
		script := "#!/bin/sh\nprintf '%s %s\\n' " + name + " \"$*\" >> '" + calls + "'\nexit 1\n"
		if err := os.WriteFile(filepath.Join(directory, name), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", directory+string(os.PathListSeparator)+os.Getenv("PATH"))
	return func() []string {
		data, err := os.ReadFile(calls)
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		return strings.Split(strings.TrimSpace(string(data)), "\n")
	}
}

func TestDashboardBackendRunsNoContainerRuntime(t *testing.T) {
	root := ".."
	files := token.NewFileSet()
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		parsed, err := parser.ParseFile(files, path, nil, 0)
		if err != nil {
			return err
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.CallExpr:
				if program, ok := execProgram(node); ok && isContainerCLI(program) {
					t.Errorf("%s runs the container CLI %q", files.Position(node.Pos()), program)
				}
			case *ast.BasicLit:
				value, unquoteErr := strconv.Unquote(node.Value)
				if node.Kind != token.STRING || unquoteErr != nil {
					return true
				}
				for _, forbidden := range []string{"docker.sock", "podman.sock", "recipe_topology_reconcile"} {
					if strings.Contains(value, forbidden) {
						t.Errorf("%s names %s", files.Position(node.Pos()), forbidden)
					}
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// execProgram returns the literal program an os/exec call starts or looks up.
func execProgram(call *ast.CallExpr) (string, bool) {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	if pkg, isIdent := selector.X.(*ast.Ident); !isIdent || pkg.Name != "exec" {
		return "", false
	}
	argument := 0
	switch selector.Sel.Name {
	case "Command", "LookPath":
	case "CommandContext":
		argument = 1
	default:
		return "", false
	}
	if len(call.Args) <= argument {
		return "", false
	}
	literal, ok := call.Args[argument].(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		return "", false
	}
	program, err := strconv.Unquote(literal.Value)
	return program, err == nil
}

func isContainerCLI(program string) bool {
	for _, name := range containerCLINames {
		if filepath.Base(program) == name {
			return true
		}
	}
	return false
}
