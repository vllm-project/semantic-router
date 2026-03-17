//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

// CompileResult is the JSON structure returned by signalCompile.
type CompileResult struct {
	YAML        string           `json:"yaml"`
	CRD         string           `json:"crd,omitempty"`
	Diagnostics []DiagnosticJSON `json:"diagnostics"`
	AST         interface{}      `json:"ast,omitempty"`
	Error       string           `json:"error,omitempty"`
}

// DiagnosticJSON is a JSON-serializable diagnostic.
type DiagnosticJSON struct {
	Level   string         `json:"level"`
	Message string         `json:"message"`
	Line    int            `json:"line"`
	Column  int            `json:"column"`
	Fixes   []QuickFixJSON `json:"fixes,omitempty"`
}

// QuickFixJSON is a JSON-serializable quick fix.
type QuickFixJSON struct {
	Description string `json:"description"`
	NewText     string `json:"newText"`
}

// ValidateResult is the JSON structure returned by signalValidate.
type ValidateResult struct {
	Diagnostics []DiagnosticJSON `json:"diagnostics"`
	ErrorCount  int              `json:"errorCount"`
	Symbols     *SymbolTableJSON `json:"symbols,omitempty"`
	Error       string           `json:"error,omitempty"`
}

// SymbolTableJSON is a JSON-serializable symbol table for editor completions.
type SymbolTableJSON struct {
	Signals []SymbolInfoJSON `json:"signals"`
	Models  []string         `json:"models"`
	Plugins []string         `json:"plugins"`
	Routes  []string         `json:"routes"`
}

// SymbolInfoJSON is a named symbol with its type.
type SymbolInfoJSON struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// DecompileResult is the JSON structure returned by signalDecompile.
type DecompileResult struct {
	DSL   string `json:"dsl"`
	Error string `json:"error,omitempty"`
}

// FormatResult is the JSON structure returned by signalFormat.
type FormatResult struct {
	DSL   string `json:"dsl"`
	Error string `json:"error,omitempty"`
}

// ParseASTResult is the JSON structure returned by signalParseAST.
type ParseASTResult struct {
	AST         interface{}      `json:"ast,omitempty"`
	Diagnostics []DiagnosticJSON `json:"diagnostics"`
	Symbols     *SymbolTableJSON `json:"symbols,omitempty"`
	ErrorCount  int              `json:"errorCount"`
	Error       string           `json:"error,omitempty"`
}

func main() {
	js.Global().Set("signalCompile", js.FuncOf(compile))
	js.Global().Set("signalValidate", js.FuncOf(validate))
	js.Global().Set("signalDecompile", js.FuncOf(decompile))
	js.Global().Set("signalFormat", js.FuncOf(format))
	js.Global().Set("signalParseAST", js.FuncOf(parseAST))

	// Keep the Go program alive.
	select {}
}

// compile implements signalCompile(dslSource: string) → string (JSON).
// Full pipeline: DSL → parse → validate → compile → emit YAML + CRD.
func compile(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(CompileResult{Error: "signalCompile requires 1 argument: dslSource"})
	}
	dslSource := args[0].String()

	// 1. Parse → AST (for Visual Builder consumption).
	prog, parseErrs := dsl.Parse(dslSource)
	var astJSON interface{}
	if prog != nil {
		astJSON = dsl.ProgramToJSON(prog)
	}

	// 2. Validate (includes lex + parse + reference + constraint checks).
	diags, valErrs := dsl.Validate(dslSource)
	diagnostics := convertDiagnostics(diags)

	// 3. Compile DSL → RouterConfig.
	cfg, compileErrs := dsl.Compile(dslSource)
	if len(compileErrs) > 0 {
		_ = parseErrs // already captured in diagnostics
		// Still return diagnostics and partial AST even on compile errors.
		return marshalJSON(CompileResult{
			AST:         astJSON,
			Diagnostics: diagnostics,
			Error:       joinErrors(compileErrs),
		})
	}

	// 3. Emit the canonical routing fragment owned by the DSL surface.
	yamlBytes, yamlErr := dsl.EmitRoutingYAMLFromConfig(cfg)
	if yamlErr != nil {
		return marshalJSON(CompileResult{
			Diagnostics: diagnostics,
			Error:       yamlErr.Error(),
		})
	}

	// 4. Emit CRD (with default name/namespace).
	crdBytes, crdErr := dsl.EmitCRD(cfg, "router", "default")
	crdStr := ""
	if crdErr == nil {
		crdStr = string(crdBytes)
	}

	// If there were validation parse errors, include them but still return output.
	errStr := ""
	if len(valErrs) > 0 {
		errStr = joinErrors(valErrs)
	}

	return marshalJSON(CompileResult{
		YAML:        string(yamlBytes),
		CRD:         crdStr,
		Diagnostics: diagnostics,
		AST:         astJSON,
		Error:       errStr,
	})
}

// validate implements signalValidate(dslSource: string) → string (JSON).
// Incremental validation only — faster than full compile.
// Also returns the symbol table extracted from the AST for editor completions.
func validate(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(ValidateResult{Error: "signalValidate requires 1 argument: dslSource"})
	}
	dslSource := args[0].String()

	diags, symbols, valErrs := dsl.ValidateWithSymbols(dslSource)
	diagnostics := convertDiagnostics(diags)

	errorCount := 0
	for _, d := range diags {
		if d.Level == dsl.DiagError {
			errorCount++
		}
	}

	errStr := ""
	if len(valErrs) > 0 {
		errStr = joinErrors(valErrs)
	}

	var symbolsJSON *SymbolTableJSON
	if symbols != nil {
		symbolsJSON = &SymbolTableJSON{
			Models:  symbols.Models,
			Plugins: symbols.Plugins,
			Routes:  symbols.Routes,
		}
		for _, s := range symbols.Signals {
			symbolsJSON.Signals = append(symbolsJSON.Signals, SymbolInfoJSON{Name: s.Name, Type: s.Type})
		}
	}

	return marshalJSON(ValidateResult{
		Diagnostics: diagnostics,
		ErrorCount:  errorCount,
		Symbols:     symbolsJSON,
		Error:       errStr,
	})
}

// decompile implements signalDecompile(yamlSource: string) → string (JSON).
// Converts a full router config YAML or routing fragment YAML back to routing-only DSL text.
func decompile(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(DecompileResult{Error: "signalDecompile requires 1 argument: yamlSource"})
	}
	yamlSource := args[0].String()

	cfg, err := config.ParseYAMLBytes([]byte(yamlSource))
	if err != nil {
		return marshalJSON(DecompileResult{Error: "YAML parse error: " + err.Error()})
	}

	dslText, err := dsl.DecompileRouting(cfg)
	if err != nil {
		return marshalJSON(DecompileResult{Error: err.Error()})
	}

	return marshalJSON(DecompileResult{DSL: dslText})
}

// format implements signalFormat(dslSource: string) → string (JSON).
// Canonical formatting via compile→decompile round-trip.
func format(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(FormatResult{Error: "signalFormat requires 1 argument: dslSource"})
	}
	dslSource := args[0].String()

	formatted, err := dsl.Format(dslSource)
	if err != nil {
		return marshalJSON(FormatResult{Error: err.Error()})
	}

	return marshalJSON(FormatResult{DSL: formatted})
}

// parseAST implements signalParseAST(dslSource: string) → string (JSON).
// Parse + validate only (no compile), returns the full AST with positions
// and symbol table. This is the primary API for the Visual Builder.
func parseAST(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(ParseASTResult{Error: "signalParseAST requires 1 argument: dslSource"})
	}
	dslSource := args[0].String()

	// Parse → AST
	prog, parseErrs := dsl.Parse(dslSource)

	// Validate (includes reference + constraint checks) + symbol table
	diags, symbols, valErrs := dsl.ValidateWithSymbols(dslSource)
	diagnostics := convertDiagnostics(diags)

	errorCount := 0
	for _, d := range diags {
		if d.Level == dsl.DiagError {
			errorCount++
		}
	}

	// Combine parse + validation errors
	allErrs := append(parseErrs, valErrs...)
	errStr := ""
	if len(allErrs) > 0 {
		errStr = joinErrors(allErrs)
	}

	// AST
	var astJSON interface{}
	if prog != nil {
		astJSON = dsl.ProgramToJSON(prog)
	}

	// Symbols
	var symbolsJSON *SymbolTableJSON
	if symbols != nil {
		symbolsJSON = &SymbolTableJSON{
			Models:  symbols.Models,
			Plugins: symbols.Plugins,
			Routes:  symbols.Routes,
		}
		for _, s := range symbols.Signals {
			symbolsJSON.Signals = append(symbolsJSON.Signals, SymbolInfoJSON{Name: s.Name, Type: s.Type})
		}
	}

	return marshalJSON(ParseASTResult{
		AST:         astJSON,
		Diagnostics: diagnostics,
		Symbols:     symbolsJSON,
		ErrorCount:  errorCount,
		Error:       errStr,
	})
}

// --- Helpers ---

func convertDiagnostics(diags []dsl.Diagnostic) []DiagnosticJSON {
	result := make([]DiagnosticJSON, len(diags))
	for i, d := range diags {
		var fixes []QuickFixJSON
		if d.Fix != nil {
			fixes = []QuickFixJSON{{
				Description: d.Fix.Description,
				NewText:     d.Fix.NewText,
			}}
		}
		result[i] = DiagnosticJSON{
			Level:   d.Level.String(),
			Message: d.Message,
			Line:    d.Pos.Line,
			Column:  d.Pos.Column,
			Fixes:   fixes,
		}
	}
	return result
}

func marshalJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		return `{"error":"json marshal failed: ` + err.Error() + `"}`
	}
	return string(b)
}

func joinErrors(errs []error) string {
	msgs := make([]string, len(errs))
	for i, e := range errs {
		msgs[i] = e.Error()
	}
	b, _ := json.Marshal(msgs)
	return string(b)
}
