package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (d *decompiler) decompileSignals() {
	d.decompileApplicationSignals(config.SignalTypeDomain)
	d.decompileKeywordSignals()
	d.decompileApplicationSignals(config.SignalTypeFactCheck)
	d.decompileApplicationSignals(config.SignalTypeUserFeedback)
	d.decompileApplicationSignals(config.SignalTypeTurnRelation)
	d.decompileLanguageSignals()
	d.decompileContextSignals()
	d.decompileConversationSignals()
	d.decompileApplicationSignals(config.SignalTypeComplexity)
	d.decompileApplicationSignals(config.SignalTypeModality)
	d.decompileAuthzSignals()
	d.decompileApplicationSignals(config.SignalTypeJailbreak)
	d.decompileApplicationSignals(config.SignalTypeSafety)
	d.decompileApplicationSignals(config.SignalTypeHallucination)
	d.decompileApplicationSignals(config.SignalTypeRefusal)
	d.decompileApplicationSignals(config.SignalTypePII)
	d.decompileMetadataSignals()
	d.decompileInputModalitySignals()
	d.decompileSystemOneSignals()
	d.decompileProjectionSignals()
}
