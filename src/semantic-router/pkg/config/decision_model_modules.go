package config

// The deployment that answers a built-in signal is resolved here, in one
// place, so its authoring surface can change without touching the signals:
//
//  1. a binding the recipe or global.model_catalog.bindings declares;
//  2. else, with a systemone decision model, the decision model, unless the
//     configuration names the module's model itself;
//  3. else the module's model: the decision model's deployment when it is
//     that model, else an implicit model_runtime deployment of its own.
//
// Hazard classification, embeddings and reranking never follow a systemone
// decision model: Vela 1.0 Hazard is a model of its own, and embeddings and
// rerankers are no question a decision model answers.

// DecisionModelResource resolves the decision model's deployment of either
// kind: a model_runtime one, as DecisionModelDeployment does, or a systemone
// one, which answers questions and serves no module model.
func (c *RouterConfig) DecisionModelResource() (name string, deployment ModelDeployment, ok bool, err error) {
	if c.RemoteDecisionModel() {
		name = c.DecisionModel
		if name == "" {
			name = DefaultDecisionDeployment
		}
		return name, c.ModelDeployments[name], true, nil
	}
	return c.DecisionModelDeployment()
}

// remoteDecisionModelFor resolves a module to a systemone decision model: the
// module follows it when its model is the decision model's default for it and
// the configuration does not name the module's model itself.
func (c *RouterConfig) remoteDecisionModelFor(module moduleModel) (string, ModelDeployment, bool) {
	if !c.RemoteDecisionModel() || c.AuthoredModuleModels[module.module] || module.model != c.DecisionModelSpec().Model {
		return "", ModelDeployment{}, false
	}
	name, deployment, _, _ := c.DecisionModelResource()
	return name, deployment, true
}

// authoredModuleModels are the built-in modules whose model a configuration
// names itself: by a global.model_catalog.system line, or by the module's own
// model_id or a model_ref to another module's line. override is the
// configuration's own global section and raw its document.
func authoredModuleModels(override *CanonicalGlobal, raw *StructuredPayload) map[string]bool {
	if override == nil {
		return nil
	}
	authored := make(map[string]bool)
	if raw != nil {
		for key := range explicitSystemLines(raw, override.ModelCatalog.System) {
			authored[key] = true
		}
	} else {
		defaults := DefaultSystemModels()
		for _, line := range systemLines {
			if value := *line.value(&override.ModelCatalog.System); value != "" && value != *line.value(&defaults) {
				authored[line.key] = true
			}
		}
	}
	modules := override.ModelCatalog.Modules
	for line, module := range map[string]struct{ ref, id string }{
		"safety":                 {modules.Safety.Safety.ModelRef, modules.Safety.Safety.ModelID},
		"hazard":                 {modules.Safety.Hazard.ModelRef, modules.Safety.Hazard.ModelID},
		"prompt_guard":           {modules.PromptGuard.ModelRef, modules.PromptGuard.ModelID},
		"domain_classifier":      {modules.Classifier.Domain.ModelRef, modules.Classifier.Domain.ModelID},
		"pii_classifier":         {modules.Classifier.PII.ModelRef, modules.Classifier.PII.ModelID},
		"fact_check_classifier":  {modules.HallucinationMitigation.FactCheck.ModelRef, modules.HallucinationMitigation.FactCheck.ModelID},
		"hallucination_detector": {modules.HallucinationMitigation.Detector.ModelRef, modules.HallucinationMitigation.Detector.ModelID},
		"feedback_detector":      {modules.FeedbackDetector.ModelRef, modules.FeedbackDetector.ModelID},
	} {
		if module.id != "" || (module.ref != "" && module.ref != line) {
			authored[line] = true
		}
	}
	return authored
}
