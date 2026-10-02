package dsl

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Constraint fixtures come from rules already on main; the on_unknown/on_error check lands separately.

// constraint fixtures: validation reports DiagConstraint, compilation succeeds.
const constraintPriorityDSL = `SIGNAL keyword urgent { operator: "OR" keywords: ["urgent"] }
ROUTE bad_route { PRIORITY -1 WHEN keyword("urgent") MODEL "m:1b" }
`

const constraintThresholdDSL = `SIGNAL embedding emb { threshold: 1.5 candidates: ["x"] }
ROUTE emb_route { PRIORITY 1 WHEN embedding("emb") MODEL "m:1b" }
`

// warnings-only fixture: undefined signal reference is DiagWarning.
const warningOnlyDSL = `SIGNAL keyword urgent { operator: "OR" keywords: ["urgent"] }
ROUTE warn_route { PRIORITY 1 WHEN keyword("urgent") WHEN domain("nonexistent") MODEL "m:1b" }
`

const cleanDSL = `SIGNAL keyword urgent { operator: "OR" keywords: ["urgent"] }
ROUTE urgent_route { PRIORITY 100 WHEN keyword("urgent") MODEL "m:1b" }
`

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what it wrote; fn must write less than the pipe buffer, since output is drained only after fn returns.
func captureStdout(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = orig }()

	fnErr := fn()
	_ = w.Close()
	os.Stdout = orig

	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read captured stdout: %v", err)
	}
	return string(out), fnErr
}

func writeDSLFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "input.dsl")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write DSL input: %v", err)
	}
	return path
}

func TestCLIValidateExitsNonZeroOnConstraint(t *testing.T) {
	cases := map[string]string{
		"negative priority":    constraintPriorityDSL,
		"out-of-range default": constraintThresholdDSL,
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			path := writeDSLFile(t, input)

			var buf bytes.Buffer
			blocking := CLIValidate(path, &buf)

			if blocking == 0 {
				t.Errorf("expected non-zero exit count for constraint diagnostics, got 0\nOutput: %s", buf.String())
			}
			if !strings.Contains(buf.String(), "🟠 Constraint") {
				t.Errorf("expected constraint diagnostics in output, got: %s", buf.String())
			}
		})
	}
}

func TestCLIValidateExitsZeroOnWarningsOnly(t *testing.T) {
	path := writeDSLFile(t, warningOnlyDSL)

	var buf bytes.Buffer
	blocking := CLIValidate(path, &buf)

	if blocking != 0 {
		t.Errorf("warnings must not block validate, got exit count %d\nOutput: %s", blocking, buf.String())
	}
	if !strings.Contains(buf.String(), "nonexistent") {
		t.Errorf("expected warning about undefined signal, got: %s", buf.String())
	}
}

func TestCLICompileWritesNothingOnConstraint(t *testing.T) {
	cases := map[string]string{
		"negative priority":    constraintPriorityDSL,
		"out-of-range default": constraintThresholdDSL,
	}
	for name, input := range cases {
		t.Run(name+" file output", func(t *testing.T) {
			inPath := writeDSLFile(t, input)
			outPath := filepath.Join(t.TempDir(), "out.yaml")

			err := CLICompile(inPath, outPath, "yaml", "", "", "")

			if err == nil {
				t.Error("expected compile to fail on constraint diagnostics, got nil error")
			}
			if _, statErr := os.Stat(outPath); !os.IsNotExist(statErr) {
				t.Errorf("expected no output file to be written, stat err = %v", statErr)
			}
		})
		t.Run(name+" stdout output", func(t *testing.T) {
			inPath := writeDSLFile(t, input)

			stdout, err := captureStdout(t, func() error {
				return CLICompile(inPath, "-", "yaml", "", "", "")
			})

			if err == nil {
				t.Error("expected compile to fail on constraint diagnostics, got nil error")
			}
			if stdout != "" {
				t.Errorf("expected nothing on stdout, got %d bytes:\n%s", len(stdout), stdout)
			}
		})
	}
}

func TestCLIValidateAndCompileSucceedOnCleanFile(t *testing.T) {
	inPath := writeDSLFile(t, cleanDSL)

	var buf bytes.Buffer
	if blocking := CLIValidate(inPath, &buf); blocking != 0 {
		t.Errorf("expected validate exit count 0 on clean file, got %d\nOutput: %s", blocking, buf.String())
	}

	outPath := filepath.Join(t.TempDir(), "out.yaml")
	if err := CLICompile(inPath, outPath, "yaml", "", "", ""); err != nil {
		t.Fatalf("expected compile to succeed on clean file, got: %v", err)
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("expected compile output to be written: %v", err)
	}
	if len(data) == 0 {
		t.Error("expected non-empty compile output on clean file")
	}
}
