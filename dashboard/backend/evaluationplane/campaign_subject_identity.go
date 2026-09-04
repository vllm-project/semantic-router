package evaluationplane

import "fmt"

// CandidateSubjectContractVersion identifies the suite- and execution-independent
// projection used to compose evidence from different evaluation runs.
const CandidateSubjectContractVersion = "evaluation-candidate-subject.v1"

// candidateSubjectDigest derives the identity of the candidate under evaluation.
// Callers must first load the manifest and report through the server-owned durable
// validation path; the checks here deliberately repeat every cross-bundle subject
// binding so campaign composition fails closed if the two validated objects are
// accidentally mixed.
//
// Execution context is not candidate identity. In particular, mode, suites,
// workloads, seeds, sample limits, observations, raw environment snapshots, and
// gate ledger endpoints are excluded. The raw environment digest is still required
// so a report with missing server-validated lineage cannot be projected, while its
// mode- and ledger-dependent value is not hashed. Target and backend topology are
// the stable environment factors that belong to the candidate subject.
func candidateSubjectDigest(manifest RunManifest, report Report) (string, error) {
	value, err := candidateSubjectCanonicalValue(manifest, report)
	if err != nil {
		return "", err
	}
	return canonicalValueDigest(value)
}

func candidateSubjectCanonicalValue(manifest RunManifest, report Report) (map[string]any, error) {
	if manifest.SchemaVersion != SchemaVersion || manifest.Target.SchemaVersion != SchemaVersion ||
		report.SchemaVersion != SchemaVersion || report.Run.SchemaVersion != SchemaVersion ||
		report.Provenance.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("%w: candidate subject schema version mismatch", ErrInvalid)
	}
	if report.AttestationRevision != ServerAttestationRevision {
		return nil, fmt.Errorf("%w: candidate subject report is not server-attested", ErrInvalid)
	}
	if (manifest.Mode != ModeLive && manifest.Mode != ModeReplay) || report.Run.Mode != manifest.Mode {
		return nil, fmt.Errorf("%w: candidate subject manifest/report mode mismatch", ErrInvalid)
	}
	if report.Run.Status != StatusCompleted || report.Run.Error != "" {
		return nil, fmt.Errorf("%w: candidate subject requires a completed report", ErrInvalid)
	}
	if !validClientRequestID(manifest.RunID) || report.Run.ID != manifest.RunID ||
		report.Run.ClientRequestID != manifest.RunID {
		return nil, fmt.Errorf("%w: candidate subject run identity mismatch", ErrInvalid)
	}
	if !digestPattern.MatchString(manifest.ManifestDigest) {
		return nil, fmt.Errorf("%w: candidate subject manifest digest is missing", ErrInvalid)
	}
	manifestDigest, digestErr := manifestSemanticDigest(manifest)
	if digestErr != nil || manifestDigest != manifest.ManifestDigest {
		return nil, fmt.Errorf("%w: candidate subject manifest digest mismatch", ErrInvalid)
	}
	if manifest.Target.Kind != "mixture-of-models" || manifest.Target.Mixture == nil ||
		report.Run.Mixture == nil {
		return nil, fmt.Errorf("%w: candidate subject requires a frozen brokered Mixture", ErrInvalid)
	}
	mixture := manifest.Target.Mixture
	if err := validateManifestMixtureContract(mixture); err != nil {
		return nil, fmt.Errorf("%w: candidate subject Mixture is invalid: %w", ErrInvalid, err)
	}
	if !targetIDMatchesMixture(manifest.Target.ID, mixture.ID) || report.Run.TargetID != manifest.Target.ID ||
		report.Provenance.TargetID != manifest.Target.ID {
		return nil, fmt.Errorf("%w: candidate subject target identity mismatch", ErrInvalid)
	}
	if !validChangeProfile(manifest.ChangeProfile) || report.Run.ChangeProfile != manifest.ChangeProfile {
		return nil, fmt.Errorf("%w: candidate subject change profile mismatch", ErrInvalid)
	}
	if !sourceRevisionPattern.MatchString(manifest.CodeRevision) ||
		report.Provenance.CodeRevision != manifest.CodeRevision {
		return nil, fmt.Errorf("%w: candidate subject code revision mismatch", ErrInvalid)
	}
	if !digestPattern.MatchString(manifest.ConfigDigest) ||
		!digestPattern.MatchString(manifest.PolicySnapshotDigest) ||
		!digestPattern.MatchString(manifest.Target.BackendTopologyDigest) ||
		!digestPattern.MatchString(report.Provenance.EnvironmentSnapshotDigest) {
		return nil, fmt.Errorf("%w: candidate subject required digest is missing", ErrInvalid)
	}
	if manifest.PolicySnapshotDigest != mixture.RecipeDigest ||
		report.Provenance.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		report.Provenance.BindingSnapshotDigest != mixture.BindingDigest ||
		report.Provenance.PoolSnapshotDigest != mixture.PoolDigest {
		return nil, fmt.Errorf("%w: candidate subject provenance factor mismatch", ErrInvalid)
	}

	manifestMixture, manifestMixtureErr := manifestMixtureCanonicalValue(mixture)
	if manifestMixtureErr != nil {
		return nil, fmt.Errorf("%w: candidate subject Mixture is invalid: %w", ErrInvalid, manifestMixtureErr)
	}
	reportManifestMixture := manifestMixtureFromCatalog(report.Run.Mixture)
	if err := validateManifestMixtureContract(reportManifestMixture); err != nil {
		return nil, fmt.Errorf("%w: candidate subject report Mixture is invalid: %w", ErrInvalid, err)
	}
	reportMixture, err := manifestMixtureCanonicalValue(reportManifestMixture)
	if err != nil {
		return nil, fmt.Errorf("%w: candidate subject report Mixture is invalid: %w", ErrInvalid, err)
	}
	manifestMixtureDigest, err := canonicalValueDigest(manifestMixture)
	if err != nil {
		return nil, fmt.Errorf("%w: digest candidate subject manifest Mixture: %w", ErrInvalid, err)
	}
	reportMixtureDigest, err := canonicalValueDigest(reportMixture)
	if err != nil || reportMixtureDigest != manifestMixtureDigest {
		return nil, fmt.Errorf("%w: candidate subject manifest/report Mixture mismatch", ErrInvalid)
	}

	// RecipeDescription is mutable presentation metadata, not executable
	// candidate identity. All behavioral Recipe state remains bound by the
	// recipe/policy digests and the decision bindings below.
	delete(manifestMixture, "recipe_description")
	return map[string]any{
		"schema_version":   SchemaVersion,
		"contract_version": CandidateSubjectContractVersion,
		"change_profile":   manifest.ChangeProfile,
		"target": map[string]any{
			"id":                      manifest.Target.ID,
			"kind":                    manifest.Target.Kind,
			"backend_topology_digest": manifest.Target.BackendTopologyDigest,
		},
		"code_revision":          manifest.CodeRevision,
		"config_digest":          manifest.ConfigDigest,
		"policy_snapshot_digest": manifest.PolicySnapshotDigest,
		"mixture":                manifestMixture,
	}, nil
}
