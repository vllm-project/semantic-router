package dsl

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CLICompile reads a DSL file, compiles it, and writes the output in the specified format.
// format can be "yaml" (default), "crd", or "helm".
// basePath, when non-empty, points to a YAML file with infrastructure config
// (version, listeners, providers) that the compiled routing is merged into,
// producing a complete runnable config.
func CLICompile(inputPath, outputPath, format, crdName, crdNamespace, basePath string) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("failed to read input file: %w", err)
	}

	cfg, errs := Compile(string(data))
	if len(errs) > 0 {
		for _, e := range errs {
			fmt.Fprintf(os.Stderr, "  %s\n", e)
		}
		return fmt.Errorf("%d compilation error(s)", len(errs))
	}

	output, err := emitFormat(cfg, format, crdName, crdNamespace, basePath)
	if err != nil {
		return err
	}
	return writeOutput(output, outputPath)
}

func emitFormat(cfg *config.RouterConfig, format, crdName, crdNamespace, basePath string) ([]byte, error) {
	switch format {
	case "yaml", "":
		if basePath != "" {
			return emitMergedConfig(cfg, basePath)
		}
		return EmitRoutingYAMLFromConfig(cfg)
	case "crd":
		if crdName == "" {
			crdName = "router"
		}
		return EmitCRD(cfg, crdName, crdNamespace)
	case "helm":
		return EmitHelm(cfg)
	default:
		return nil, fmt.Errorf("unsupported output format %q (supported: yaml, crd, helm)", format)
	}
}

// emitMergedConfig reads a base YAML (version, listeners, providers) and
// overlays the DSL-compiled routing into it, producing a complete config.
func emitMergedConfig(cfg *config.RouterConfig, basePath string) ([]byte, error) {
	baseData, err := os.ReadFile(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read base config: %w", err)
	}

	return MergeRoutingIntoBase(cfg, baseData)
}

// CLIDecompile reads a YAML config file and converts its routing surface to DSL text.
func CLIDecompile(inputPath, outputPath string) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("failed to read input file: %w", err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		cfg, err = config.ParseRoutingYAMLBytes(data)
		if err != nil {
			return fmt.Errorf("failed to parse YAML: %w", err)
		}
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		return fmt.Errorf("decompilation failed: %w", err)
	}

	return writeOutput([]byte(dslText), outputPath)
}

// TestBlockRunnerFactory constructs a TEST block runner for a parsed program.
type TestBlockRunnerFactory func(prog *Program) (TestBlockRunner, error)

// CLIValidate reads a DSL file and reports diagnostics.
// Returns the number of errors found.
func CLIValidate(inputPath string, w io.Writer) int {
	return cliValidate(inputPath, w, nil)
}

// CLIValidateWithRunner reads a DSL file, reports diagnostics, and executes TEST blocks
// using a runner factory when the parsed program contains them.
func CLIValidateWithRunner(inputPath string, w io.Writer, factory TestBlockRunnerFactory) int {
	return cliValidate(inputPath, w, factory)
}

func cliValidate(inputPath string, w io.Writer, factory TestBlockRunnerFactory) int {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		_, _ = fmt.Fprintf(w, "failed to read input file: %s\n", err)
		return 1
	}

	input := string(data)
	diags, _ := Validate(input)
	diags = appendRuntimeValidationDiagnostics(input, diags, factory)
	if len(diags) == 0 {
		_, _ = fmt.Fprintln(w, "No issues found.")
		return 0
	}

	return writeValidationDiagnostics(w, diags)
}

func appendRuntimeValidationDiagnostics(input string, diags []Diagnostic, factory TestBlockRunnerFactory) []Diagnostic {
	if factory == nil || hasBlockingDiagnostics(diags) {
		return diags
	}

	prog, parseErrs := Parse(input)
	if len(parseErrs) != 0 || prog == nil || !programNeedsRuntimeValidation(prog) {
		return diags
	}

	runner, err := factory(prog)
	if err != nil {
		return append(diags, Diagnostic{
			Level:   DiagError,
			Message: fmt.Sprintf("native runtime validation initialization failed: %v", err),
			Pos:     runtimeValidationPos(prog),
		})
	}
	return append(diags, collectRuntimeValidationDiagnostics(prog, runner)...)
}

func collectRuntimeValidationDiagnostics(prog *Program, runner TestBlockRunner) []Diagnostic {
	diags := make([]Diagnostic, 0)
	if len(prog.TestBlocks) > 0 {
		diags = append(diags, ValidateTestBlocks(prog, runner)...)
	}
	if validator, ok := runner.(ProjectionPartitionRuntimeValidator); ok {
		diags = append(diags, validator.ValidateProjectionPartitions(prog)...)
	}
	return diags
}

func runtimeValidationPos(prog *Program) Position {
	if prog == nil {
		return Position{}
	}
	if len(prog.TestBlocks) > 0 {
		return prog.TestBlocks[0].Pos
	}
	for _, partition := range prog.ProjectionPartitions {
		if programNeedsRuntimeValidation(&Program{ProjectionPartitions: []*ProjectionPartitionDecl{partition}}) {
			return partition.Pos
		}
	}
	return Position{}
}

func writeValidationDiagnostics(w io.Writer, diags []Diagnostic) int {
	var errCount, warnCount, constraintCount int
	for _, d := range diags {
		switch d.Level {
		case DiagError:
			errCount++
		case DiagWarning:
			warnCount++
		case DiagConstraint:
			constraintCount++
		}
		_, _ = fmt.Fprintln(w, d.String())
	}

	_, _ = fmt.Fprintf(w, "\nSummary: 🔴 %d error(s)  🟡 %d warning(s)  🟠 %d constraint(s)\n",
		errCount, warnCount, constraintCount)

	return errCount
}

func hasBlockingDiagnostics(diags []Diagnostic) bool {
	for _, diag := range diags {
		if diag.Level == DiagError || diag.Level == DiagConstraint {
			return true
		}
	}
	return false
}

// CLIFormat reads a DSL file, formats it, and writes the result.
func CLIFormat(inputPath, outputPath string) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("failed to read input file: %w", err)
	}

	formatted, err := Format(string(data))
	if err != nil {
		return fmt.Errorf("formatting failed: %w", err)
	}

	// If no output path specified, overwrite the input file
	if outputPath == "" {
		outputPath = inputPath
	}

	return writeOutput([]byte(formatted), outputPath)
}

// writeOutput writes data to a file or stdout if outputPath is empty or "-".
func writeOutput(data []byte, outputPath string) error {
	if outputPath == "" || outputPath == "-" {
		_, err := os.Stdout.Write(data)
		return err
	}

	return os.WriteFile(filepath.Clean(outputPath), data, 0o644) //nolint:gosec // G703: path sanitized with filepath.Clean
}
