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
func CLICompile(inputPath, outputPath, format, crdName, crdNamespace string) error {
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
	switch format {
	case "yaml", "":
		output, err = EmitYAMLFromConfig(cfg)
		if err != nil {
			return fmt.Errorf("YAML emission failed: %w", err)
		}
	case "crd":
		if crdName == "" {
			crdName = "router"
		}
		output, err = EmitCRD(cfg, crdName, crdNamespace)
		if err != nil {
			return fmt.Errorf("CRD emission failed: %w", err)
		}
	case "helm":
		output, err = EmitHelm(cfg)
		if err != nil {
			return fmt.Errorf("helm emission failed: %w", err)
		}
	default:
		return fmt.Errorf("unsupported output format %q (supported: yaml, crd, helm)", format)
	}

	return writeOutput(output, outputPath)
}

// CLIDecompile reads a YAML config file and converts it to DSL text.
func CLIDecompile(inputPath, outputPath string) error {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("failed to read input file: %w", err)
	}

	var cfg config.RouterConfig
	if unmarshalErr := yaml.Unmarshal(data, &cfg); unmarshalErr != nil {
		return fmt.Errorf("failed to parse YAML: %w", unmarshalErr)
	}

	dslText, err := Decompile(&cfg)
	if err != nil {
		return fmt.Errorf("decompilation failed: %w", err)
	}

	return writeOutput([]byte(dslText), outputPath)
}

// CLIValidate reads a DSL file and reports diagnostics.
// Returns the number of errors found.
func CLIValidate(inputPath string, w io.Writer) int {
	data, err := os.ReadFile(inputPath)
	if err != nil {
		fmt.Fprintf(w, "failed to read input file: %s\n", err)
		return 1
	}

	diags, _ := Validate(string(data))
	if len(diags) == 0 {
		fmt.Fprintln(w, "✅ No issues found.")
		return 0
	}

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
		fmt.Fprintln(w, d.String())
	}

	fmt.Fprintf(w, "\nSummary: 🔴 %d error(s)  🟡 %d warning(s)  🟠 %d constraint(s)\n",
		errCount, warnCount, constraintCount)

	return errCount
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
