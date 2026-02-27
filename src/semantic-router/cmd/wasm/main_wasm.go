//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
	"gopkg.in/yaml.v3"
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
	Signals  []SymbolInfoJSON `json:"signals"`
	Models   []string         `json:"models"`
	Plugins  []string         `json:"plugins"`
	Backends []SymbolInfoJSON `json:"backends"`
	Routes   []string         `json:"routes"`
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

	// 3. Emit YAML in user-friendly nested format (signals/providers).
	yamlBytes, yamlErr := dsl.EmitUserYAML(cfg)
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
		for _, b := range symbols.Backends {
			symbolsJSON.Backends = append(symbolsJSON.Backends, SymbolInfoJSON{Name: b.Name, Type: b.Type})
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
// Converts YAML RouterConfig back to DSL text.
// Supports both the "user-friendly" format (signals.keywords, providers.models)
// used by deploy configs and the "flat" format (keyword_rules) used by RouterConfig.
func decompile(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(DecompileResult{Error: "signalDecompile requires 1 argument: yamlSource"})
	}
	yamlSource := args[0].String()

	// Normalize: convert user-friendly YAML (signals.keywords etc.) to flat RouterConfig format.
	normalized, normErr := normalizeYAML([]byte(yamlSource))
	if normErr != nil {
		return marshalJSON(DecompileResult{Error: "YAML normalization error: " + normErr.Error()})
	}

	var cfg config.RouterConfig
	if err := yaml.Unmarshal(normalized, &cfg); err != nil {
		return marshalJSON(DecompileResult{Error: "YAML parse error: " + err.Error()})
	}

	dslText, err := dsl.Decompile(&cfg)
	if err != nil {
		return marshalJSON(DecompileResult{Error: err.Error()})
	}

	return marshalJSON(DecompileResult{DSL: dslText})
}

// normalizeYAML converts the user-friendly YAML format (with nested "signals"
// and "providers" sections) into the flat RouterConfig format that Go can unmarshal.
// If the YAML is already in flat format, it passes through unchanged.
func normalizeYAML(data []byte) ([]byte, error) {
	var raw map[string]interface{}
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	changed := false

	// Flatten "signals" section → top-level keys
	if signals, ok := raw["signals"]; ok {
		if signalsMap, ok := signals.(map[string]interface{}); ok {
			signalKeyMap := map[string]string{
				"keywords":       "keyword_rules",
				"embeddings":     "embedding_rules",
				"domains":        "categories",
				"fact_check":     "fact_check_rules",
				"user_feedbacks": "user_feedback_rules",
				"preferences":    "preference_rules",
				"language":       "language_rules",
				"context":        "context_rules",
				"complexity":     "complexity_rules",
				"modality":       "modality_rules",
				"authz":          "role_bindings",
				"jailbreak":      "jailbreak",
				"pii":            "pii",
			}
			for srcKey, dstKey := range signalKeyMap {
				if v, exists := signalsMap[srcKey]; exists {
					if _, already := raw[dstKey]; !already {
						raw[dstKey] = v
						changed = true
					}
				}
			}
			delete(raw, "signals")
			changed = true
		}
	}

	// Flatten "providers" section → vllm_endpoints, model_config, default_model, etc.
	if providers, ok := raw["providers"]; ok {
		if provMap, ok := providers.(map[string]interface{}); ok {
			// Extract models → vllm_endpoints + model_config
			if models, ok := provMap["models"]; ok {
				if modelList, ok := models.([]interface{}); ok {
					var endpoints []interface{}
					modelConfig := make(map[string]interface{})
					for _, m := range modelList {
						mMap, ok := m.(map[string]interface{})
						if !ok {
							continue
						}
						modelName, _ := mMap["name"].(string)
						if modelName == "" {
							continue
						}
						mc := map[string]interface{}{}
						if rf, ok := mMap["reasoning_family"]; ok {
							mc["reasoning_family"] = rf
						}
						if ps, ok := mMap["param_size"]; ok {
							mc["param_size"] = ps
						}
						if ak, ok := mMap["access_key"]; ok {
							mc["access_key"] = ak
						}
						if af, ok := mMap["api_format"]; ok {
							mc["api_format"] = af
						}
						if pr, ok := mMap["pricing"]; ok {
							mc["pricing"] = pr
						}
						modelConfig[modelName] = mc

						if epList, ok := mMap["endpoints"].([]interface{}); ok {
							for _, ep := range epList {
								epMap, ok := ep.(map[string]interface{})
								if !ok {
									continue
								}
								epStr, _ := epMap["endpoint"].(string)
								epName, _ := epMap["name"].(string)
								weight := epMap["weight"]
								protocol, _ := epMap["protocol"].(string)
								if protocol == "" {
									protocol = "http"
								}
								host, port := parseEndpoint(epStr, protocol)
								endpoints = append(endpoints, map[string]interface{}{
									"name":     modelName + "_" + epName,
									"address":  host,
									"port":     port,
									"weight":   weight,
									"protocol": protocol,
									"model":    modelName,
								})
							}
						}
					}
					if _, already := raw["vllm_endpoints"]; !already && len(endpoints) > 0 {
						raw["vllm_endpoints"] = endpoints
						changed = true
					}
					if _, already := raw["model_config"]; !already && len(modelConfig) > 0 {
						raw["model_config"] = modelConfig
						changed = true
					}
				}
			}
			// Hoist simple keys
			for _, key := range []string{"default_model", "reasoning_families", "default_reasoning_effort"} {
				if v, ok := provMap[key]; ok {
					if _, already := raw[key]; !already {
						raw[key] = v
						changed = true
					}
				}
			}
			delete(raw, "providers")
			changed = true
		}
	}

	if !changed {
		return data, nil
	}

	return yaml.Marshal(raw)
}

// parseEndpoint splits "host:port" or "host:port/path" into host and port.
func parseEndpoint(ep string, protocol string) (string, int) {
	// Strip path
	if idx := indexOf(ep, '/'); idx >= 0 {
		ep = ep[:idx]
	}
	if idx := indexOf(ep, ':'); idx >= 0 {
		host := ep[:idx]
		port := 0
		for _, c := range ep[idx+1:] {
			if c >= '0' && c <= '9' {
				port = port*10 + int(c-'0')
			}
		}
		if port == 0 {
			port = 80
		}
		return host, port
	}
	if protocol == "https" {
		return ep, 443
	}
	return ep, 80
}

func indexOf(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
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
		for _, b := range symbols.Backends {
			symbolsJSON.Backends = append(symbolsJSON.Backends, SymbolInfoJSON{Name: b.Name, Type: b.Type})
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
