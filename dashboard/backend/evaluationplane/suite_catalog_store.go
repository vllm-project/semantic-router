package evaluationplane

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	normalizedSuiteSchemaVersion      = "evaluation-suite.v1"
	adapterContractVersion            = "benchmark-adapter.v1"
	benchmarkSourceContractVersion    = "benchmark-source.v1"
	suiteQualificationContractVersion = "evaluation-suite-qualification.v2"
	normalizedSuiteExecutorID         = "normalized-suite-replay.v1"
	normalizedSuiteLiveExecutorID     = "normalized-suite-live.v1"
	normalizedSuiteExecutorDigest     = "sha256:2fcfdac903c5cde4b8964806e00f4288be59414cb8317acf2223f54198ef77d1"
	maxSuiteCatalogDocumentBytes      = int64(4 * 1024 * 1024)
)

var portableSuiteIDPattern = portableIDPattern

type suiteIndexRecord struct {
	ID                string `json:"id"`
	Revision          string `json:"revision"`
	ManifestDigest    string `json:"manifest_digest"`
	ManifestSizeBytes int64  `json:"manifest_size_bytes"`
}

type suiteManifestProjection struct {
	SchemaVersion          string          `json:"schema_version"`
	ID                     string          `json:"id"`
	Name                   string          `json:"name"`
	AdapterID              string          `json:"adapter_id"`
	AdapterContractVersion string          `json:"adapter_contract_version"`
	SourceReceipt          json.RawMessage `json:"source_receipt"`
	Revision               string          `json:"revision"`
	DecisionUnit           string          `json:"decision_unit"`
	ActionSpace            string          `json:"action_space"`
	TrackIDs               []TrackID       `json:"track_ids"`
	QualificationReceipt   struct {
		SchemaVersion         string          `json:"schema_version"`
		EvidenceLevel         EvidenceLevel   `json:"evidence_level"`
		ManifestSubjectDigest string          `json:"manifest_subject_digest"`
		SourceReceiptDigest   string          `json:"source_receipt_digest"`
		ArtifactSetDigest     string          `json:"artifact_set_digest"`
		ExecutorID            string          `json:"executor_id"`
		ExecutorDigest        string          `json:"executor_digest"`
		Qualification         json.RawMessage `json:"qualification"`
	} `json:"qualification_receipt"`
	SplitProtocol      string          `json:"split_protocol"`
	CaseCount          int             `json:"case_count"`
	ArmIDs             []string        `json:"arm_ids"`
	DataClassification string          `json:"data_classification"`
	Redistribution     string          `json:"redistribution"`
	Artifacts          json.RawMessage `json:"artifacts"`
	Limitations        []string        `json:"limitations"`
}

type suiteArtifactReference struct {
	SchemaVersion string `json:"schema_version"`
	Digest        string `json:"digest"`
	SizeBytes     int64  `json:"size_bytes"`
	MediaType     string `json:"media_type"`
}

type installedSuiteDocument struct {
	Index    suiteIndexRecord
	Manifest suiteManifestProjection
	Catalog  CatalogSuite
}

func loadInstalledCatalogSuites(root string) ([]CatalogSuite, error) {
	indexRoot := filepath.Join(root, "index")
	entries, err := os.ReadDir(indexRoot)
	if err != nil {
		return nil, fmt.Errorf("read normalized suite index: %w", err)
	}
	suites := make([]CatalogSuite, 0, len(entries))
	for _, entry := range entries {
		if entry.Type()&os.ModeSymlink != 0 || entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			return nil, fmt.Errorf("%w: normalized suite index contains an unexpected entry", ErrInvalid)
		}
		document, err := loadInstalledSuiteDocument(root, strings.TrimSuffix(entry.Name(), ".json"))
		if err != nil {
			return nil, err
		}
		suites = append(suites, document.Catalog)
	}
	sort.Slice(suites, func(i, j int) bool { return suites[i].ID < suites[j].ID })
	return suites, nil
}

func loadInstalledSuiteDocument(root, suiteID string) (installedSuiteDocument, error) {
	if !portableSuiteIDPattern.MatchString(suiteID) {
		return installedSuiteDocument{}, fmt.Errorf("%w: installed suite identity is invalid", ErrInvalid)
	}
	indexBytes, indexReadErr := readPrivateSuiteFile(filepath.Join(root, "index", suiteID+".json"), maxSuiteCatalogDocumentBytes)
	if indexReadErr != nil {
		return installedSuiteDocument{}, fmt.Errorf("%w: installed suite %q index is unavailable", ErrInvalid, suiteID)
	}
	var index suiteIndexRecord
	if decodeErr := decodeExactJSON(indexBytes, &index); decodeErr != nil || index.ID != suiteID ||
		!digestPattern.MatchString(index.Revision) || !digestPattern.MatchString(index.ManifestDigest) ||
		index.ManifestSizeBytes <= 0 || index.ManifestSizeBytes > maxSuiteCatalogDocumentBytes {
		return installedSuiteDocument{}, fmt.Errorf("%w: installed suite %q index is invalid", ErrInvalid, suiteID)
	}
	manifestPath := filepath.Join(root, "manifests", "sha256", strings.TrimPrefix(index.ManifestDigest, "sha256:"))
	manifestBytes, manifestReadErr := readPrivateSuiteFile(manifestPath, index.ManifestSizeBytes)
	if manifestReadErr != nil || int64(len(manifestBytes)) != index.ManifestSizeBytes || suiteDocumentDigest(manifestBytes) != index.ManifestDigest {
		return installedSuiteDocument{}, fmt.Errorf("%w: installed suite %q manifest is unavailable or corrupt", ErrInvalid, suiteID)
	}
	catalog, validationErr := validateInstalledSuiteManifest(root, manifestBytes, index)
	if validationErr != nil {
		return installedSuiteDocument{}, validationErr
	}
	var manifest suiteManifestProjection
	if decodeErr := decodeExactJSON(manifestBytes, &manifest); decodeErr != nil {
		return installedSuiteDocument{}, fmt.Errorf("%w: installed suite %q manifest is invalid", ErrInvalid, suiteID)
	}
	return installedSuiteDocument{Index: index, Manifest: manifest, Catalog: catalog}, nil
}

func readPrivateSuiteFile(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
		return nil, fmt.Errorf("%w: normalized suite file is not a private regular file", ErrInvalid)
	}
	if info.Size() < 0 || info.Size() > limit {
		return nil, fmt.Errorf("%w: normalized suite file exceeds its size limit", ErrInvalid)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open normalized suite file: %w", err)
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return nil, fmt.Errorf("read normalized suite file: %w", err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("%w: normalized suite file exceeds its size limit", ErrInvalid)
	}
	return data, nil
}

func suiteDocumentDigest(data []byte) string {
	return fmt.Sprintf("sha256:%x", sha256.Sum256(data))
}

func validateInstalledSuiteManifest(root string, data []byte, index suiteIndexRecord) (CatalogSuite, error) {
	var manifest suiteManifestProjection
	if err := decodeExactJSON(data, &manifest); err != nil {
		return CatalogSuite{}, fmt.Errorf("%w: invalid normalized suite manifest", ErrInvalid)
	}
	if manifest.SchemaVersion != normalizedSuiteSchemaVersion || manifest.AdapterContractVersion != adapterContractVersion ||
		manifest.ID != index.ID || manifest.Revision != index.Revision || manifest.Name == "" || strings.TrimSpace(manifest.Name) != manifest.Name ||
		!portableSuiteIDPattern.MatchString(manifest.AdapterID) || manifest.CaseCount <= 0 || len(manifest.TrackIDs) == 0 ||
		!canonicalTrackOrder(manifest.TrackIDs) {
		return CatalogSuite{}, fmt.Errorf("%w: normalized suite manifest identity is invalid", ErrInvalid)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return CatalogSuite{}, fmt.Errorf("%w: invalid normalized suite manifest", ErrInvalid)
	}
	delete(raw, "revision")
	revision, revisionErr := canonicalValueDigest(raw)
	if revisionErr != nil || revision != manifest.Revision {
		return CatalogSuite{}, fmt.Errorf("%w: normalized suite revision is invalid", ErrInvalid)
	}
	if err := validateSuiteQualification(manifest); err != nil {
		return CatalogSuite{}, err
	}
	var importEvidence unqualifiedSuiteEvidence
	if err := decodeExactJSON(manifest.QualificationReceipt.Qualification, &importEvidence); err != nil {
		return CatalogSuite{}, fmt.Errorf("%w: normalized import provenance is invalid", ErrInvalid)
	}
	if err := validateInstalledSuiteArtifacts(root, manifest.Artifacts); err != nil {
		return CatalogSuite{}, err
	}
	importSummary := "The imported records passed the required data checks."
	importTag := "user-provided-import"
	if importEvidence.ParserVerified {
		importSummary = "The registered parser was rerun and produced the same records."
		importTag = "parser-verified"
	}
	methods, err := installedCatalogMethods(root, manifest)
	if err != nil {
		return CatalogSuite{}, err
	}
	executors := map[Mode]string{ModeReplay: normalizedSuiteExecutorID}
	modes := []Mode{ModeReplay}
	tags := []string{
		"external", "pinned", "exploratory-e0", "normalized-replay", importTag,
		"native-run-unattested", "adapter:" + manifest.AdapterID,
		"classification:" + manifest.DataClassification, "redistribution:" + manifest.Redistribution,
	}
	if len(normalizedSuiteLiveMethodTracks(CatalogSuite{Methods: methods})) > 0 {
		executors[ModeLive] = normalizedSuiteLiveExecutorID
		modes = append(modes, ModeLive)
		tags = append(tags, "target-live")
	}
	return CatalogSuite{
		ID: manifest.ID, Name: manifest.Name,
		Description: "Imported benchmark workload pinned to a specific revision. " + importSummary +
			" Imported results are exploratory; release evidence requires a separately supported live evaluation.",
		Executors:     executors,
		TrackIDs:      append([]TrackID(nil), manifest.TrackIDs...),
		Modes:         modes,
		EvidenceLevel: manifest.QualificationReceipt.EvidenceLevel,
		CaseCount:     manifest.CaseCount,
		Revision:      manifest.Revision,
		Tags:          tags,
		Methods:       methods,
	}, nil
}

func installedCatalogMethods(root string, manifest suiteManifestProjection) ([]CatalogMethod, error) {
	methods := make([]CatalogMethod, 0, len(manifest.TrackIDs)+2)
	for _, trackID := range manifest.TrackIDs {
		methods = append(methods, CatalogMethod{
			ID: "normalized." + manifest.AdapterID + "." + string(trackID) + ".v1", TrackID: trackID,
			QualifiedGateIDs: []string{}, EvidenceSource: CatalogMethodEvidenceSourceNormalizedImport, Status: "configured",
		})
	}
	liveMethods, err := installedNormalizedLiveMethods(root, manifest)
	if err != nil {
		return nil, err
	}
	methods = append(methods, liveMethods...)
	return methods, nil
}

func validateInstalledSuiteArtifacts(root string, raw json.RawMessage) error {
	var encodedRefs map[string]json.RawMessage
	if err := json.Unmarshal(raw, &encodedRefs); err != nil {
		return fmt.Errorf("%w: normalized suite artifact set is invalid", ErrInvalid)
	}
	type artifactContract struct{ domain, mediaType string }
	contracts := map[string]artifactContract{
		"visible_cases": {"visible", "application/x-ndjson"}, "grading_cases": {"grading", "application/x-ndjson"},
		"outcomes": {"grading", "application/x-ndjson"}, "decisions": {"grading", "application/x-ndjson"},
		"preferences": {"grading", "application/x-ndjson"}, "trajectories": {"grading", "application/x-ndjson"},
		"perturbations": {"grading", "application/x-ndjson"}, "faults": {"grading", "application/x-ndjson"},
		"multimodal_observations": {"grading", "application/x-ndjson"}, "safety_observations": {"grading", "application/x-ndjson"},
		"capacity_observations": {"grading", "application/x-ndjson"}, "media_manifest": {"metadata", "application/x-ndjson"},
		"license_manifest": {"metadata", "application/json"},
	}
	for _, required := range []string{"visible_cases", "grading_cases", "license_manifest"} {
		if _, ok := encodedRefs[required]; !ok {
			return fmt.Errorf("%w: normalized suite is missing a required artifact", ErrInvalid)
		}
	}
	for role, encoded := range encodedRefs {
		contract, ok := contracts[role]
		var ref suiteArtifactReference
		if !ok || decodeExactJSON(encoded, &ref) != nil || ref.SchemaVersion != SchemaVersion ||
			!digestPattern.MatchString(ref.Digest) || ref.SizeBytes < 0 || ref.MediaType != contract.mediaType {
			return fmt.Errorf("%w: normalized suite artifact identity is invalid", ErrInvalid)
		}
		path := filepath.Join(root, "objects", contract.domain, "sha256", strings.TrimPrefix(ref.Digest, "sha256:"))
		info, err := os.Lstat(path)
		if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 || info.Size() != ref.SizeBytes {
			return fmt.Errorf("%w: normalized suite artifact is unavailable", ErrInvalid)
		}
		if err := verifyInstalledSuiteObject(path, ref); err != nil {
			return err
		}
	}
	return nil
}

func verifyInstalledSuiteObject(path string, ref suiteArtifactReference) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open normalized suite artifact: %w", err)
	}
	defer func() { _ = file.Close() }()
	hash := sha256.New()
	written, err := io.Copy(hash, io.LimitReader(file, ref.SizeBytes+1))
	if err != nil {
		return fmt.Errorf("hash normalized suite artifact: %w", err)
	}
	observed := fmt.Sprintf("sha256:%x", hash.Sum(nil))
	if written != ref.SizeBytes || observed != ref.Digest {
		return fmt.Errorf("%w: normalized suite artifact content is corrupt", ErrInvalid)
	}
	return nil
}

func canonicalTrackOrder(trackIDs []TrackID) bool {
	if len(trackIDs) == 0 {
		return false
	}
	canonical := canonicalTrackIDs(trackIDs)
	if len(canonical) != len(trackIDs) {
		return false
	}
	for index := range trackIDs {
		if canonical[index] != trackIDs[index] {
			return false
		}
	}
	return true
}
