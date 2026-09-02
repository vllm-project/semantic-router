package evaluationplane

import "fmt"

// methodMixtureBinding makes each provider ledger carry both the complete
// frozen Mixture snapshot digest and the component identities an operator
// needs to audit without trusting an opaque aggregate alone.
type methodMixtureBinding struct {
	ID                   string `json:"id"`
	SnapshotDigest       string `json:"snapshot_digest"`
	RecipeDigest         string `json:"recipe_digest"`
	PoolDigest           string `json:"pool_digest"`
	SelectorPolicyDigest string `json:"selector_policy_digest"`
	SelectorDigest       string `json:"selector_digest"`
	AdaptationDigest     string `json:"adaptation_digest"`
	BindingDigest        string `json:"binding_digest"`
}

func methodManifestMixtureBinding(manifest RunManifest) (methodMixtureBinding, error) {
	if manifest.Target.Mixture == nil {
		return methodMixtureBinding{}, fmt.Errorf("method ledger requires a frozen Mixture")
	}
	mixture := manifest.Target.Mixture
	normalized := *mixture
	if normalized.Aliases == nil {
		normalized.Aliases = []string{}
	}
	if normalized.ModelArms == nil {
		normalized.ModelArms = []ModelArm{}
	}
	if normalized.SupportModels == nil {
		normalized.SupportModels = []SupportModel{}
	}
	if normalized.Decisions == nil {
		normalized.Decisions = []MixtureDecisionBinding{}
	}
	snapshotDigest, err := canonicalValueDigest(normalized)
	if err != nil {
		return methodMixtureBinding{}, fmt.Errorf("digest frozen Mixture: %w", err)
	}
	return methodMixtureBinding{
		ID: mixture.ID, SnapshotDigest: snapshotDigest, RecipeDigest: mixture.RecipeDigest,
		PoolDigest: mixture.PoolDigest, SelectorPolicyDigest: mixture.SelectorPolicyDigest,
		SelectorDigest: mixture.SelectorDigest, AdaptationDigest: mixture.AdaptationDigest,
		BindingDigest: mixture.BindingDigest,
	}, nil
}

func validMethodMixtureBinding(binding methodMixtureBinding) bool {
	return validMethodID(binding.ID) && validMethodDigest(binding.SnapshotDigest) &&
		validMethodDigest(binding.RecipeDigest) && validMethodDigest(binding.PoolDigest) &&
		validMethodDigest(binding.SelectorPolicyDigest) && validMethodDigest(binding.SelectorDigest) &&
		validMethodDigest(binding.AdaptationDigest) && validMethodDigest(binding.BindingDigest)
}
