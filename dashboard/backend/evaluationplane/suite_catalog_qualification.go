package evaluationplane

import (
	"encoding/json"
	"fmt"
	"regexp"
)

var adapterSourceRevisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

type suiteSourceReceiptProjection struct {
	SchemaVersion           string  `json:"schema_version"`
	AdapterID               string  `json:"adapter_id"`
	ExpectedSourceRevision  string  `json:"expected_source_revision"`
	ObservedSourceRevision  string  `json:"observed_source_revision"`
	ExpectedDatasetRevision *string `json:"expected_dataset_revision,omitempty"`
	ObservedDatasetRevision *string `json:"observed_dataset_revision,omitempty"`
	SourceClean             bool    `json:"source_clean"`
	DatasetClean            *bool   `json:"dataset_clean,omitempty"`
	Verified                bool    `json:"verified"`
}

type unqualifiedSuiteEvidence struct {
	SchemaVersion           string `json:"schema_version"`
	Status                  string `json:"status"`
	Origin                  string `json:"origin"`
	ParserVerified          bool   `json:"parser_verified"`
	NativeExecutionAttested bool   `json:"native_execution_attested"`
	PromotionEligible       bool   `json:"promotion_eligible"`
}

func validateSuiteQualification(manifest suiteManifestProjection) error {
	receipt := manifest.QualificationReceipt
	if receipt.SchemaVersion != suiteQualificationContractVersion ||
		receipt.EvidenceLevel != "E0" || receipt.ExecutorID != normalizedSuiteExecutorID ||
		receipt.ExecutorDigest != normalizedSuiteExecutorDigest ||
		!digestPattern.MatchString(receipt.ManifestSubjectDigest) ||
		!digestPattern.MatchString(receipt.SourceReceiptDigest) ||
		!digestPattern.MatchString(receipt.ArtifactSetDigest) {
		return fmt.Errorf("%w: normalized imports require an E0 provenance receipt", ErrInvalid)
	}
	contract, known := normalizedAdapterContracts[manifest.AdapterID]
	if !known || manifest.DecisionUnit != contract.decisionUnit ||
		manifest.ActionSpace != contract.actionSpace ||
		!normalizedAdapterTracksMatch(contract, manifest.TrackIDs) {
		return fmt.Errorf("%w: normalized import workload does not match its adapter", ErrInvalid)
	}
	if _, err := validateSuiteSourceReceipt(manifest, contract); err != nil {
		return err
	}
	sourceDigest, err := canonicalJSONDigest(manifest.SourceReceipt)
	if err != nil || sourceDigest != receipt.SourceReceiptDigest {
		return fmt.Errorf("%w: suite provenance does not bind its source receipt", ErrInvalid)
	}
	artifactDigest, err := canonicalJSONDigest(manifest.Artifacts)
	if err != nil || artifactDigest != receipt.ArtifactSetDigest {
		return fmt.Errorf("%w: suite provenance does not bind its artifacts", ErrInvalid)
	}
	subjectDigest, err := canonicalValueDigest(suiteQualificationSubject(manifest))
	if err != nil || subjectDigest != receipt.ManifestSubjectDigest {
		return fmt.Errorf("%w: suite provenance does not bind its manifest", ErrInvalid)
	}
	var evidence unqualifiedSuiteEvidence
	if err := decodeExactJSON(receipt.Qualification, &evidence); err != nil ||
		evidence.SchemaVersion != suiteQualificationContractVersion ||
		evidence.Status != "exploratory_import" ||
		evidence.NativeExecutionAttested || evidence.PromotionEligible ||
		!validImportOrigin(evidence.Origin, evidence.ParserVerified) {
		return fmt.Errorf("%w: normalized import provenance is invalid", ErrInvalid)
	}
	return nil
}

func validImportOrigin(origin string, parserVerified bool) bool {
	switch origin {
	case "registered_parser_import":
		return parserVerified
	case "user_provided_import":
		return !parserVerified
	default:
		return false
	}
}

func suiteQualificationSubject(manifest suiteManifestProjection) map[string]any {
	subject := map[string]any{
		"schema_version": normalizedSuiteSchemaVersion,
		"id":             manifest.ID, "name": manifest.Name, "adapter_id": manifest.AdapterID,
		"adapter_contract_version": manifest.AdapterContractVersion,
		"decision_unit":            manifest.DecisionUnit, "action_space": manifest.ActionSpace,
		"track_ids": manifest.TrackIDs, "split_protocol": manifest.SplitProtocol,
		"case_count": manifest.CaseCount, "arm_ids": manifest.ArmIDs,
		"data_classification": manifest.DataClassification, "redistribution": manifest.Redistribution,
		"limitations": manifest.Limitations,
	}
	var source, artifacts any
	_ = json.Unmarshal(manifest.SourceReceipt, &source)
	_ = json.Unmarshal(manifest.Artifacts, &artifacts)
	subject["source_receipt"], subject["artifacts"] = source, artifacts
	return subject
}

func validateSuiteSourceReceipt(
	manifest suiteManifestProjection,
	contract normalizedAdapterContract,
) (suiteSourceReceiptProjection, error) {
	var source suiteSourceReceiptProjection
	if err := decodeExactJSON(manifest.SourceReceipt, &source); err != nil ||
		source.SchemaVersion != adapterContractVersion ||
		source.AdapterID != manifest.AdapterID || !source.Verified || !source.SourceClean ||
		source.ExpectedSourceRevision != contract.sourceRevision ||
		source.ObservedSourceRevision != contract.sourceRevision ||
		!adapterSourceRevisionPattern.MatchString(source.ExpectedSourceRevision) ||
		!validDatasetSourceReceipt(source, contract.datasetRevision) {
		return suiteSourceReceiptProjection{}, fmt.Errorf("%w: normalized import source receipt is invalid", ErrInvalid)
	}
	return source, nil
}

func validDatasetSourceReceipt(source suiteSourceReceiptProjection, expected string) bool {
	if expected == "" {
		return source.ExpectedDatasetRevision == nil &&
			source.ObservedDatasetRevision == nil && source.DatasetClean == nil
	}
	return source.ExpectedDatasetRevision != nil && *source.ExpectedDatasetRevision == expected &&
		source.ObservedDatasetRevision != nil && *source.ObservedDatasetRevision == expected &&
		source.DatasetClean != nil && *source.DatasetClean
}
