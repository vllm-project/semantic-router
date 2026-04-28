package dsl

// signalCompilerByType dispatches SIGNAL declarations to per-type compilers.
// Table-driven compileSignals keeps funlen/cyclomatic pressure off the main loop.
var signalCompilerByType = map[string]func(*Compiler, *SignalDecl){
	"keyword":        (*Compiler).compileKeywordSignal,
	"embedding":      (*Compiler).compileEmbeddingSignal,
	"domain":         (*Compiler).compileDomainSignal,
	"fact_check":     (*Compiler).compileFactCheckSignal,
	"user_feedback":  (*Compiler).compileUserFeedbackSignal,
	"reask":          (*Compiler).compileReaskSignal,
	"preference":     (*Compiler).compilePreferenceSignal,
	"language":       (*Compiler).compileLanguageSignal,
	"context":        (*Compiler).compileContextSignal,
	"structure":      (*Compiler).compileStructureSignal,
	"complexity":     (*Compiler).compileComplexitySignal,
	"modality":       (*Compiler).compileModalitySignal,
	"authz":          (*Compiler).compileAuthzSignal,
	"jailbreak":      (*Compiler).compileJailbreakSignal,
	"pii":            (*Compiler).compilePIISignal,
	"kb":             (*Compiler).compileKBSignal,
	"conversation":   (*Compiler).compileConversationSignal,
	"session_metric": (*Compiler).compileSessionMetricRule,
}
