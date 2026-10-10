// Package editor exposes the DSL editor contract to HTTP and WASM transports.
package editor

import (
	"encoding/json"

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

// compile implements signalCompile(dslSource: string) → string (JSON).
// Full pipeline: DSL → parse → validate → compile → emit YAML + CRD.
func Compile(dslSource string) CompileResult {
	return CompileWithLimits(dslSource, config.DecisionRuleLimits{})
}

// CompileWithBase preserves the rule budget of a full document being edited.
func CompileWithBase(dslSource, baseYAML string) CompileResult {
	limits, err := config.DecisionRuleLimitsFromYAML([]byte(baseYAML))
	if err != nil {
		return CompileResult{Error: err.Error()}
	}
	return CompileWithLimits(dslSource, limits)
}

// CompileWithLimits applies the enclosing config budget in both editor transports.
func CompileWithLimits(dslSource string, limits config.DecisionRuleLimits) CompileResult {
	// 1. Parse → AST (for Visual Builder consumption).
	prog, parseErrs := dsl.Parse(dslSource)
	if prog != nil {
		if err := dsl.ValidateProgramRuleLimits(prog, limits); err != nil {
			return CompileResult{Error: err.Error()}
		}
	}
	var astJSON interface{}
	if prog != nil {
		astJSON = dsl.ProgramToJSON(prog)
	}

	// 2. Validate (includes lex + parse + reference + constraint checks).
	diags, valErrs := dsl.ValidateWithLimits(dslSource, limits)
	diagnostics := convertDiagnostics(diags)
	diagnostics = appendRuntimeValidationWarning(diagnostics, prog)

	// 3. Compile DSL → RouterConfig.
	cfg, compileErrs := dsl.CompileWithLimits(dslSource, limits)
	if len(compileErrs) > 0 {
		_ = parseErrs // already captured in diagnostics
		// Still return diagnostics and partial AST even on compile errors.
		return CompileResult{
			AST:         astJSON,
			Diagnostics: diagnostics,
			Error:       joinErrors(compileErrs),
		}
	}

	// 3. Emit the canonical routing fragment owned by the DSL surface.
	yamlBytes, yamlErr := dsl.EmitRoutingYAMLFromConfig(cfg)
	if yamlErr != nil {
		return CompileResult{
			Diagnostics: diagnostics,
			Error:       yamlErr.Error(),
		}
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

	return CompileResult{
		YAML:        string(yamlBytes),
		CRD:         crdStr,
		Diagnostics: diagnostics,
		AST:         astJSON,
		Error:       errStr,
	}
}

// validate implements signalValidate(dslSource: string) → string (JSON).
// Incremental validation only — faster than full compile.
// Also returns the symbol table extracted from the AST for editor completions.
func Validate(dslSource string) ValidateResult {
	return ValidateWithLimits(dslSource, config.DecisionRuleLimits{})
}

// ValidateWithBase applies the full document's limits to editor diagnostics.
func ValidateWithBase(dslSource, baseYAML string) ValidateResult {
	limits, err := config.DecisionRuleLimitsFromYAML([]byte(baseYAML))
	if err != nil {
		return ValidateResult{Error: err.Error(), ErrorCount: 1}
	}
	return ValidateWithLimits(dslSource, limits)
}

// ValidateWithLimits checks budgets before recursive AST diagnostics.
func ValidateWithLimits(dslSource string, limits config.DecisionRuleLimits) ValidateResult {
	prog, _ := dsl.Parse(dslSource)
	if prog != nil {
		if err := dsl.ValidateProgramRuleLimits(prog, limits); err != nil {
			return ValidateResult{Error: err.Error(), ErrorCount: 1}
		}
	}
	diags, symbols, valErrs := dsl.ValidateWithSymbolsAndLimits(dslSource, limits)
	diagnostics := convertDiagnostics(diags)
	diagnostics = appendRuntimeValidationWarning(diagnostics, prog)

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

	return ValidateResult{
		Diagnostics: diagnostics,
		ErrorCount:  errorCount,
		Symbols:     symbolsJSON,
		Error:       errStr,
	}
}

// decompile implements signalDecompile(yamlSource: string) → string (JSON).
// Converts a full router config YAML or routing fragment YAML back to the
// complete DSL-owned surface, including entrypoints and recipes.
func Decompile(yamlSource string) DecompileResult {
	// Deploy writes the decompiled sections back, so ${VAR} and $$ must stay as
	// written; the Router resolves them with its own environment.
	cfg, err := config.ParseYAMLBytesDeferringEnv([]byte(yamlSource))
	if err != nil {
		return DecompileResult{Error: "YAML parse error: " + err.Error()}
	}

	dslText, err := dsl.Decompile(cfg)
	if err != nil {
		return DecompileResult{Error: err.Error()}
	}

	return DecompileResult{DSL: dslText}
}

// format implements signalFormat(dslSource: string) → string (JSON).
// Canonical formatting via compile→decompile round-trip.
func Format(dslSource string) FormatResult {
	return FormatWithBase(dslSource, "")
}

// FormatWithBase uses the same enclosing limits as compilation and validation.
func FormatWithBase(dslSource, baseYAML string) FormatResult {
	limits, err := config.DecisionRuleLimitsFromYAML([]byte(baseYAML))
	if err != nil {
		return FormatResult{Error: err.Error()}
	}
	formatted, err := dsl.FormatWithLimits(dslSource, limits)
	if err != nil {
		return FormatResult{Error: err.Error()}
	}

	return FormatResult{DSL: formatted}
}

// parseAST implements signalParseAST(dslSource: string) → string (JSON).
// Parse + validate only (no compile), returns the full AST with positions
// and symbol table. This is the primary API for the Visual Builder.
func Parse(dslSource string) ParseASTResult {
	return ParseWithLimits(dslSource, config.DecisionRuleLimits{})
}

// ParseWithBase carries the full document budget into visual-builder analysis.
func ParseWithBase(dslSource, baseYAML string) ParseASTResult {
	limits, err := config.DecisionRuleLimitsFromYAML([]byte(baseYAML))
	if err != nil {
		return ParseASTResult{Error: err.Error(), ErrorCount: 1}
	}
	return ParseWithLimits(dslSource, limits)
}

// ParseWithLimits protects the visual builder's recursive AST serialization.
func ParseWithLimits(dslSource string, limits config.DecisionRuleLimits) ParseASTResult {
	// Parse → AST
	prog, parseErrs := dsl.Parse(dslSource)
	if prog != nil {
		if err := dsl.ValidateProgramRuleLimits(prog, limits); err != nil {
			return ParseASTResult{Error: err.Error(), ErrorCount: 1}
		}
	}

	// Validate (includes reference + constraint checks) + symbol table
	diags, symbols, valErrs := dsl.ValidateWithSymbolsAndLimits(dslSource, limits)
	diagnostics := convertDiagnostics(diags)
	diagnostics = appendRuntimeValidationWarning(diagnostics, prog)

	errorCount := 0
	for _, d := range diags {
		if d.Level == dsl.DiagError {
			errorCount++
		}
	}

	// Combine parse + validation errors
	allErrs := append(append([]error(nil), parseErrs...), valErrs...)
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

	return ParseASTResult{
		AST:         astJSON,
		Diagnostics: diagnostics,
		Symbols:     symbolsJSON,
		ErrorCount:  errorCount,
		Error:       errStr,
	}
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

func appendRuntimeValidationWarning(diagnostics []DiagnosticJSON, prog *dsl.Program) []DiagnosticJSON {
	if prog == nil {
		return diagnostics
	}
	line, column := runtimeValidationWarningPosition(prog)
	if line == 0 {
		return diagnostics
	}
	return append(diagnostics, DiagnosticJSON{
		Level:   dsl.DiagWarning.String(),
		Message: runtimeValidationWarningMessage(prog),
		Line:    line,
		Column:  column,
	})
}

func runtimeValidationWarningMessage(prog *dsl.Program) string {
	if len(prog.TestBlocks) > 0 {
		return "TEST blocks are parsed in browser validation but not executed against the native routing signal pipeline; run sr-dsl validate natively for runtime TEST checks"
	}
	return ""
}

func runtimeValidationWarningPosition(prog *dsl.Program) (int, int) {
	if len(prog.TestBlocks) > 0 {
		return prog.TestBlocks[0].Pos.Line, prog.TestBlocks[0].Pos.Column
	}
	return 0, 0
}

func joinErrors(errs []error) string {
	msgs := make([]string, len(errs))
	for i, e := range errs {
		msgs[i] = e.Error()
	}
	b, _ := json.Marshal(msgs)
	return string(b)
}
