//go:build js && wasm

package main

import (
	"encoding/json"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl/editor"
	"syscall/js"
)

type CompileResult = editor.CompileResult
type DiagnosticJSON = editor.DiagnosticJSON
type QuickFixJSON = editor.QuickFixJSON
type ValidateResult = editor.ValidateResult
type SymbolTableJSON = editor.SymbolTableJSON
type SymbolInfoJSON = editor.SymbolInfoJSON
type DecompileResult = editor.DecompileResult
type FormatResult = editor.FormatResult
type ParseASTResult = editor.ParseASTResult

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
	return marshalJSON(editor.Compile(args[0].String()))
}

func validate(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(ValidateResult{Error: "signalValidate requires 1 argument: dslSource"})
	}
	return marshalJSON(editor.Validate(args[0].String()))
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
	return marshalJSON(editor.Format(args[0].String()))
}

func parseAST(_ js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return marshalJSON(ParseASTResult{Error: "signalParseAST requires 1 argument: dslSource"})
	}
	return marshalJSON(editor.Parse(args[0].String()))
}

func marshalJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		return `{"error":"json marshal failed: ` + err.Error() + `"}`
	}
	return string(b)
}
