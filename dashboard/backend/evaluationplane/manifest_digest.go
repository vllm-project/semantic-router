package evaluationplane

import "fmt"

func manifestSemanticDigest(manifest RunManifest) (string, error) {
	// ManifestDigest is deliberately excluded. It is the server-owned digest of
	// this semantic value, not an input to itself.
	target := map[string]any{
		"schema_version": manifest.Target.SchemaVersion,
		"id":             manifest.Target.ID,
		"kind":           manifest.Target.Kind,
	}
	if manifest.Target.RouterAPIURL != "" {
		target["router_api_url"] = manifest.Target.RouterAPIURL
	}
	if manifest.Target.EnvoyURL != "" {
		target["envoy_url"] = manifest.Target.EnvoyURL
	}
	if manifest.Target.RouterAPIKey != nil {
		target["router_api_key"] = manifest.Target.RouterAPIKey
	}
	if manifest.Target.EnvoyAPIKey != nil {
		target["envoy_api_key"] = manifest.Target.EnvoyAPIKey
	}
	if manifest.Target.AgentTaskLedger != nil {
		target["agent_task_ledger"] = manifest.Target.AgentTaskLedger
	}
	if manifest.Target.HardPolicyLedger != nil {
		target["hard_policy_ledger"] = manifest.Target.HardPolicyLedger
	}
	if manifest.Target.FaultRecoveryLedger != nil {
		target["fault_recovery_ledger"] = manifest.Target.FaultRecoveryLedger
	}
	if manifest.Target.ProductionExperimentLedger != nil {
		target["production_experiment_ledger"] = manifest.Target.ProductionExperimentLedger
	}
	if manifest.Target.BackendTopologyDigest != "" {
		target["backend_topology_digest"] = manifest.Target.BackendTopologyDigest
	}
	if manifest.Target.Mixture != nil {
		mixture, err := manifestMixtureCanonicalValue(manifest.Target.Mixture)
		if err != nil {
			return "", err
		}
		target["mixture"] = mixture
	}
	value := map[string]any{
		"schema_version":         manifest.SchemaVersion,
		"run_id":                 manifest.RunID,
		"name":                   manifest.Name,
		"description":            manifest.Description,
		"mode":                   manifest.Mode,
		"target":                 target,
		"change_profile":         manifest.ChangeProfile,
		"gate_contract_version":  manifest.GateContractVersion,
		"suite_ids":              manifest.SuiteIDs,
		"suite_revisions":        manifest.SuiteRevisions,
		"suite_executors":        manifest.SuiteExecutors,
		"track_ids":              manifest.TrackIDs,
		"sample_limit":           manifest.SampleLimit,
		"concurrency":            manifest.Concurrency,
		"seed":                   manifest.Seed,
		"created_at":             manifest.CreatedAt,
		"code_revision":          manifest.CodeRevision,
		"config_digest":          manifest.ConfigDigest,
		"policy_snapshot_digest": manifest.PolicySnapshotDigest,
		"redaction_policy":       manifest.RedactionPolicy,
	}
	if manifest.BaselineRunID != "" {
		value["baseline_run_id"] = manifest.BaselineRunID
	}
	if manifest.CapacitySLO != nil {
		value["capacity_slo"] = manifest.CapacitySLO
	}
	if manifest.CapacityLoadProtocol != nil {
		value["capacity_load_protocol"] = manifest.CapacityLoadProtocol
	}
	return canonicalValueDigest(value)
}

func manifestMixtureCanonicalValue(mixture *ManifestMixture) (map[string]any, error) {
	if err := validateMixtureContract(mixture); err != nil {
		return nil, err
	}
	plan, err := canonicalRoutingRecipePlan(mixture.RoutingRecipePlan)
	if err != nil {
		return nil, fmt.Errorf("canonicalize routing recipe plan: %w", err)
	}
	arms, err := modelArmsCanonicalValue(mixture.ModelArms)
	if err != nil {
		return nil, err
	}
	aliases := append([]string{}, mixture.Aliases...)
	supportModels := supportModelsCanonicalValue(mixture.SupportModels)
	decisions := make([]any, 0, len(mixture.Decisions))
	for _, decision := range mixture.Decisions {
		decisions = append(decisions, map[string]any{
			"name": decision.Name, "algorithm": decision.Algorithm,
			"arm_ids": append([]string{}, decision.ArmIDs...),
		})
	}
	value := map[string]any{
		"schema_version": mixture.SchemaVersion,
		"id":             mixture.ID, "entrypoint_model": mixture.EntrypointModel,
		"aliases": aliases, "recipe_name": mixture.RecipeName,
		"recipe_description": mixture.RecipeDescription,
		"recipe_digest":      mixture.RecipeDigest, "pool_digest": mixture.PoolDigest,
		"selector_policy_digest": mixture.SelectorPolicyDigest, "selector_digest": mixture.SelectorDigest,
		"adaptation_digest": mixture.AdaptationDigest, "binding_digest": mixture.BindingDigest, "model_arms": arms,
		"support_models": supportModels, "decisions": decisions,
		"routing_recipe_plan": copyRoutingRecipePlan(plan),
	}
	if mixture.FallbackArmID != "" {
		value["fallback_arm_id"] = mixture.FallbackArmID
	}
	return value, nil
}

func supportModelsCanonicalValue(models []SupportModel) []any {
	values := make([]any, 0, len(models))
	for _, model := range models {
		value := map[string]any{
			"model":                    model.Model,
			"provider_model_id_digest": model.ProviderModelIDDigest,
			"config_digest":            model.ConfigDigest,
			"backend_topology_digest":  model.BackendTopologyDigest,
		}
		if model.RuntimeRevision != nil {
			value["runtime_revision"] = *model.RuntimeRevision
		}
		values = append(values, value)
	}
	return values
}

func modelArmsCanonicalValue(arms []ModelArm) ([]any, error) {
	values := make([]any, 0, len(arms))
	for _, arm := range arms {
		if arm.ID == "" || arm.Model == "" || !digestPattern.MatchString(arm.ProviderModelIDDigest) {
			return nil, fmt.Errorf("invalid model arm identity")
		}
		capabilities := append([]string(nil), arm.Capabilities...)
		if capabilities == nil {
			capabilities = []string{}
		}
		modalities := append([]string(nil), arm.Modalities...)
		if modalities == nil {
			modalities = []string{}
		}
		value := map[string]any{
			"id": arm.ID, "model": arm.Model, "provider_model_id_digest": arm.ProviderModelIDDigest,
			"input_cost_per_million_tokens_usd":  arm.InputCostPerMillionTokensUSD,
			"output_cost_per_million_tokens_usd": arm.OutputCostPerMillionTokensUSD,
			"capabilities":                       capabilities, "modalities": modalities,
		}
		if arm.ContextWindowTokens != nil {
			value["context_window_tokens"] = *arm.ContextWindowTokens
		}
		if arm.ParameterSize != nil {
			value["parameter_size"] = *arm.ParameterSize
		}
		if arm.RuntimeRevision != nil {
			value["runtime_revision"] = *arm.RuntimeRevision
		}
		if arm.ConfigDigest != nil {
			value["config_digest"] = *arm.ConfigDigest
		}
		values = append(values, value)
	}
	return values, nil
}
