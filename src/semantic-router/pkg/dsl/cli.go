package dsl

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ConfigBytesParser is the application-owned Config parser used by
// decompile. pkg/dsl stays Provider-neutral; the command composition root
// injects the Provider Integration compiler installed in that binary.
type ConfigBytesParser func([]byte) (*config.RouterConfig, error)

// CLICompile reads a DSL file, compiles it, and writes one model-free routing
// fragment. There is deliberately no CRD or Helm translation here: those
// deployment surfaces own their complete runtime resource contracts.
// basePath, when non-empty, points to a YAML file with infrastructure config
// and exactly one Recipe whose routing profile is replaced by the compiled value.
func CLICompile(inputPath, outputPath, basePath string) error {
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

	var output []byte
	if basePath == "" {
		output, err = EmitRoutingYAMLFromConfig(cfg)
	} else {
		output, err = emitMergedConfig(cfg, basePath)
	}
	if err != nil {
		return err
	}
	return writeOutput(output, outputPath)
}

// emitMergedConfig reads a complete base manifest and replaces its sole
// Recipe routing profile with the DSL-compiled value.
func emitMergedConfig(cfg *config.RouterConfig, basePath string) ([]byte, error) {
	baseData, err := os.ReadFile(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read base config: %w", err)
	}

	return MergeRoutingIntoBase(cfg, baseData)
}

// CLIDecompile reads a YAML config file and converts its default routing
// profile plus entrypoint and recipe scopes to DSL text.
func CLIDecompile(inputPath, outputPath string) error {
	return CLIDecompileWithParser(inputPath, outputPath, nil)
}

// CLIDecompileWithParser decompiles either a complete manifest through the
// injected application parser or a narrow model-free Recipe authoring value.
func CLIDecompileWithParser(inputPath, outputPath string, parseConfig ConfigBytesParser) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("failed to read input file: %w", err)
	}

	dslText, err := DecompileYAML(data, parseConfig)
	if err != nil {
		return err
	}
	return writeOutput([]byte(dslText), outputPath)
}

// DecompileYAML converts either a complete manifest through an
// application-provided parser or one of the provider-neutral Recipe values
// owned by the DSL boundary: {routing: ...} or
// {recipes: [{name, description, routing}]}. Callers without a Provider
// Integration compiler can only decompile those narrow forms.
func DecompileYAML(data []byte, parseConfig ConfigBytesParser) (string, error) {
	var cfg *config.RouterConfig
	var manifestErr error
	if parseConfig != nil {
		cfg, manifestErr = parseConfig(data)
		if manifestErr != nil {
			cfg = nil
		}
	}
	if cfg == nil {
		var recipeErr error
		cfg, recipeErr = parseRecipeAuthoringYAML(data)
		if recipeErr != nil {
			if manifestErr != nil {
				return "", fmt.Errorf("failed to parse YAML: %w", errors.Join(
					fmt.Errorf("manifest: %w", manifestErr),
					fmt.Errorf("recipe authoring YAML: %w", recipeErr),
				))
			}
			return "", fmt.Errorf("failed to parse Recipe authoring YAML: %w", recipeErr)
		}
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		return "", fmt.Errorf("decompilation failed: %w", err)
	}
	return dslText, nil
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

	return os.WriteFile(outputPath, data, 0o644)
}
