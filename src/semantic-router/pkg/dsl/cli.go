package dsl

import (
	"fmt"
	"io"
	"os"

	"gopkg.in/yaml.v2"

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

	input := string(data)
	var limits config.DecisionRuleLimits
	if basePath != "" {
		base, readErr := os.ReadFile(basePath)
		if readErr != nil {
			return fmt.Errorf("failed to read base config: %w", readErr)
		}
		limits, err = config.DecisionRuleLimitsFromYAML(base)
		if err != nil {
			return err
		}
	}
	cfg, errs := CompileWithLimits(input, limits)
	if len(errs) > 0 {
		for _, e := range errs {
			fmt.Fprintf(os.Stderr, "  %s\n", e)
		}
		return fmt.Errorf("%d compilation error(s)", len(errs))
	}

	// Refuse to write output on blocking validation diagnostics; parsing twice keeps Compile byte-identical.
	if diags, _ := ValidateWithLimits(input, limits); hasBlockingDiagnostics(diags) {
		blocking := writeValidationDiagnostics(os.Stderr, diags)
		return fmt.Errorf("%d blocking validation diagnostic(s); no output written", blocking)
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

// CLIDecompile reads a YAML config file and converts its default routing
// profile plus entrypoint and recipe scopes to DSL text.
func CLIDecompile(inputPath, outputPath string) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("failed to read input file: %w", err)
	}

	// Decompilation is an offline source operation. Preserve environment
	// references and do not require locally installed inference assets.
	cfg, err := config.ParseYAMLBytesWithoutEnvExpansion(data)
	if err != nil {
		if !isRoutingOnlyFragment(data) {
			return fmt.Errorf("failed to parse YAML: %w", err)
		}
		cfg, err = config.ParseRoutingYAMLBytes(data)
		if err != nil {
			return fmt.Errorf("failed to parse YAML: %w", err)
		}
	}

	dslText, err := Decompile(cfg)
	if err != nil {
		return fmt.Errorf("decompilation failed: %w", err)
	}

	return writeOutput([]byte(dslText), outputPath)
}

// A routing-only fallback intentionally skips provider validation. Never use
// it for a complete document: a fragment holds recipes, entrypoints and
// providers.models[] entries that state only model facts, nothing else.
func isRoutingOnlyFragment(data []byte) bool {
	var fields map[string]interface{}
	if err := yaml.Unmarshal(data, &fields); err != nil || fields["recipes"] == nil {
		return false
	}
	for key, value := range fields {
		switch key {
		case "recipes", "entrypoints":
		case "version":
			if value != config.CanonicalConfigVersion {
				return false
			}
		case "providers":
			if !factOnlyProviders(value) {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func factOnlyProviders(value interface{}) bool {
	providers, ok := stringKeyed(value)
	if !ok || len(providers) != 1 {
		return false
	}
	models, ok := providers["models"].([]interface{})
	if !ok {
		return false
	}
	facts := map[string]bool{"name": true}
	for _, field := range config.ModelFactFields() {
		facts[field] = true
	}
	for _, raw := range models {
		model, ok := stringKeyed(raw)
		if !ok {
			return false
		}
		for field := range model {
			if !facts[field] {
				return false
			}
		}
	}
	return true
}

// stringKeyed reads a YAML mapping, which yaml.v2 decodes with interface keys
// below the top level.
func stringKeyed(value interface{}) (map[string]interface{}, bool) {
	switch typed := value.(type) {
	case map[string]interface{}:
		return typed, true
	case map[interface{}]interface{}:
		mapping := make(map[string]interface{}, len(typed))
		for key, item := range typed {
			name, ok := key.(string)
			if !ok {
				return nil, false
			}
			mapping[name] = item
		}
		return mapping, true
	default:
		return nil, false
	}
}

// TestBlockRunnerFactory constructs a TEST block runner for a parsed program.
type TestBlockRunnerFactory func(prog *Program) (TestBlockRunner, error)

// CLIValidate reports diagnostics for a DSL file and returns the blocking count (errors and constraints, not warnings).
func CLIValidate(inputPath string, w io.Writer) int {
	return cliValidate(inputPath, w, nil)
}

// CLIValidateWithRunner is CLIValidate plus TEST-block runtime checks via the given runner factory.
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
	if len(parseErrs) != 0 || prog == nil || len(prog.TestBlocks) == 0 {
		return diags
	}

	runner, err := factory(prog)
	if err != nil {
		return append(diags, Diagnostic{
			Level:   DiagError,
			Message: fmt.Sprintf("native runtime validation initialization failed: %v", err),
			Pos:     prog.TestBlocks[0].Pos,
		})
	}
	return append(diags, ValidateTestBlocks(prog, runner)...)
}

// writeValidationDiagnostics prints diagnostics with a summary and returns the blocking (error + constraint) count.
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

	return errCount + constraintCount
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
