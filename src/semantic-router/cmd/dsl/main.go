package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

const usage = `Usage: sr-dsl <command> [options]

Commands:
  compile    Compile DSL to YAML/CRD
  decompile  Convert YAML config to DSL
  validate   Validate a DSL file
  fmt        Format a DSL file

Examples:
  sr-dsl compile config.dsl -o config.yaml
  sr-dsl compile config.dsl --format crd -o router-config.yaml
  sr-dsl decompile config.yaml -o config.dsl
  sr-dsl validate config.dsl
  sr-dsl fmt config.dsl
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	cmd := os.Args[1]
	os.Args = append(os.Args[:1], os.Args[2:]...)

	switch cmd {
	case "compile":
		runCompile()
	case "decompile":
		runDecompile()
	case "validate":
		runValidate()
	case "fmt", "format":
		runFormat()
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}
}

func runCompile() {
	fs := flag.NewFlagSet("compile", flag.ExitOnError)
	output := fs.String("o", "", "Output file path (default: stdout)")
	format := fs.String("format", "yaml", "Output format: yaml, crd")
	crdName := fs.String("name", "router", "CRD resource name (for --format crd)")
	crdNamespace := fs.String("namespace", "", "CRD namespace (for --format crd, default: \"default\")")
	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl compile <input.dsl> [-o output.yaml] [--format yaml|crd]")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	if err := dsl.CLICompile(inputPath, *output, *format, *crdName, *crdNamespace); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func runDecompile() {
	fs := flag.NewFlagSet("decompile", flag.ExitOnError)
	output := fs.String("o", "", "Output file path (default: stdout)")

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl decompile <input.yaml> [-o output.dsl]")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	if err := dsl.CLIDecompile(inputPath, *output); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func runValidate() {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl validate <input.dsl>")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	errCount := dsl.CLIValidate(inputPath, os.Stdout)
	if errCount > 0 {
		os.Exit(1)
	}
}

func runFormat() {
	fs := flag.NewFlagSet("fmt", flag.ExitOnError)
	output := fs.String("o", "", "Output file path (default: overwrite input)")

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl fmt <input.dsl> [-o output.dsl]")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	if err := dsl.CLIFormat(inputPath, *output); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}
