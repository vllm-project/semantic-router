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
	if manifest.Target.BackendTopologyDigest != "" {
		target["backend_topology_digest"] = manifest.Target.BackendTopologyDigest
	}
	arms, err := modelArmsCanonicalValue(manifest.Target.ModelArms)
	if err != nil {
		return "", err
	}
	target["model_arms"] = arms
	value := map[string]any{
		"schema_version":         manifest.SchemaVersion,
		"run_id":                 manifest.RunID,
		"mode":                   manifest.Mode,
		"target":                 target,
		"change_profile":         manifest.ChangeProfile,
		"gate_contract_version":  manifest.GateContractVersion,
		"suite_ids":              manifest.SuiteIDs,
		"suite_revisions":        manifest.SuiteRevisions,
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
	return canonicalValueDigest(value)
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
