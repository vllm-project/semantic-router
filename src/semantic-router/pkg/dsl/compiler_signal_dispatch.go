package dsl

// signalCompilerByType dispatches the SIGNAL declarations of the types that
// are not application signals to their compilers.
// Table-driven compileSignals keeps funlen/cyclomatic pressure off the main loop.
var signalCompilerByType = map[string]func(*Compiler, *SignalDecl){
	"keyword":        (*Compiler).compileKeywordSignal,
	"language":       (*Compiler).compileLanguageSignal,
	"context":        (*Compiler).compileContextSignal,
	"authz":          (*Compiler).compileAuthzSignal,
	"conversation":   (*Compiler).compileConversationSignal,
	"metadata":       (*Compiler).compileMetadataSignal,
	"input_modality": (*Compiler).compileInputModalitySignal,
	"systemone":      (*Compiler).compileSystemOneSignal,
}
