//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl/editor"
)

type (
	CompileResult   = editor.CompileResult
	DiagnosticJSON  = editor.DiagnosticJSON
	QuickFixJSON    = editor.QuickFixJSON
	ValidateResult  = editor.ValidateResult
	SymbolTableJSON = editor.SymbolTableJSON
	SymbolInfoJSON  = editor.SymbolInfoJSON
	DecompileResult = editor.DecompileResult
	FormatResult    = editor.FormatResult
	ParseASTResult  = editor.ParseASTResult
)

func main() {
	js.Global().Set("signalCompile", js.FuncOf(compile))
	js.Global().Set("signalValidate", js.FuncOf(validate))
	js.Global().Set("signalDecompile", js.FuncOf(decompile))
	js.Global().Set("signalFormat", js.FuncOf(format))
	js.Global().Set("signalParseAST", js.FuncOf(parseAST))

	// Keep the Go program alive.
	select {}
}

func compile(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(CompileResult{Error: "signalCompile requires 1 argument: dslSource"})
	}
	return marshalJSON(editor.CompileWithBase(args[0].String(), optionalBaseYAML(args)))
}

func validate(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(ValidateResult{Error: "signalValidate requires 1 argument: dslSource"})
	}
	return marshalJSON(editor.ValidateWithBase(args[0].String(), optionalBaseYAML(args)))
}

func decompile(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(DecompileResult{Error: "signalDecompile requires 1 argument: yamlSource"})
	}
	return marshalJSON(editor.Decompile(args[0].String()))
}

func format(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(FormatResult{Error: "signalFormat requires 1 argument: dslSource"})
	}
	return marshalJSON(editor.FormatWithBase(args[0].String(), optionalBaseYAML(args)))
}

func parseAST(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(ParseASTResult{Error: "signalParseAST requires 1 argument: dslSource"})
	}
	return marshalJSON(editor.ParseWithBase(args[0].String(), optionalBaseYAML(args)))
}

func optionalBaseYAML(args []js.Value) string {
	if len(args) > 1 && args[1].Type() == js.TypeString {
		return args[1].String()
	}
	return ""
}

func marshalJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		return `{"error":"json marshal failed: ` + err.Error() + `"}`
	}
	return string(b)
}
