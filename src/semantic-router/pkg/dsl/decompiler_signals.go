package dsl

func (d *decompiler) decompileSignals() {
	d.decompileDomainSignals()
	d.decompileKeywordSignals()
	d.decompileEmbeddingSignals()
	d.decompileFactCheckSignals()
	d.decompileUserFeedbackSignals()
	d.decompileReaskSignals()
	d.decompilePreferenceSignals()
	d.decompileLanguageSignals()
	d.decompileContextSignals()
	d.decompileStructureSignals()
	d.decompileConversationSignals()
	d.decompileComplexitySignals()
	d.decompileModalitySignals()
	d.decompileAuthzSignals()
	d.decompileJailbreakSignals()
	d.decompilePIISignals()
	d.decompileKBSignals()
	d.decompileEventSignals()
	d.decompileProjectionSignals()
}
