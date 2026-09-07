package evaluationplane

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	normalizedMultimodalLiveMethodID = "multimodal.hidden-answer.server-live.v1"
	benchmarkPackLiveMethodPrefix    = "benchmark-pack.server-live"
)

// installedNormalizedLiveMethods is the admission boundary for
// installed-suite live execution. Source imports remain replay-only E0
// evidence; these methods merely make a fresh server-owned protocol runnable.
func installedNormalizedLiveMethods(
	root string,
	manifest suiteManifestProjection,
) ([]CatalogMethod, error) {
	document := installedSuiteDocument{Manifest: manifest}
	methods := make([]CatalogMethod, 0, len(manifest.TrackIDs)+2)
	declaredShift, err := installedDeclaredShiftSourceEligible(root, document)
	if err != nil {
		return nil, fmt.Errorf("%w: installed suite declared-shift qualification is invalid", err)
	}
	if declaredShift {
		methods = append(methods, CatalogMethod{
			ID: declaredShiftLiveMethodID, TrackID: "routing",
			QualifiedGateIDs: []string{"G4"}, EvidenceSource: CatalogMethodEvidenceSourceServerBrokeredLive, Status: "configured",
		})
	}
	isPack, err := installedSuiteIsBenchmarkPack(manifest)
	if err != nil {
		return nil, err
	}
	if !isPack {
		multimodal, multimodalErr := installedMultimodalLiveSourceEligible(root, document)
		if multimodalErr != nil {
			return nil, fmt.Errorf("%w: installed suite multimodal live qualification is invalid", multimodalErr)
		}
		if multimodal {
			methods = append(methods, CatalogMethod{
				ID: normalizedMultimodalLiveMethodID, TrackID: "multimodal",
				QualifiedGateIDs: []string{}, EvidenceSource: CatalogMethodEvidenceSourceLiveRuntime, Status: "configured",
			})
		}
	}
	packTracks, err := installedBenchmarkPackLiveTracks(root, document)
	if err != nil {
		return nil, fmt.Errorf("%w: installed benchmark pack live qualification is invalid", err)
	}
	for _, trackID := range manifest.TrackIDs {
		if _, admitted := packTracks[trackID]; !admitted {
			continue
		}
		methods = append(methods, CatalogMethod{
			ID: benchmarkPackLiveMethodID(trackID), TrackID: trackID,
			QualifiedGateIDs: []string{}, EvidenceSource: CatalogMethodEvidenceSourceLiveRuntime, Status: "configured",
		})
	}
	return methods, nil
}

func benchmarkPackLiveMethodID(trackID TrackID) string {
	return benchmarkPackLiveMethodPrefix + "." + string(trackID) + ".v1"
}

func benchmarkPackLiveTrack(trackID TrackID) bool {
	switch trackID {
	case "routing", "model_pool", "joint", "multimodal", "capacity":
		return true
	default:
		return false
	}
}

func installedSuiteIsBenchmarkPack(manifest suiteManifestProjection) (bool, error) {
	var source suiteSourceReceiptProjection
	if err := decodeExactJSON(manifest.SourceReceipt, &source); err != nil {
		return false, fmt.Errorf("%w: normalized suite source receipt is invalid", ErrInvalid)
	}
	return source.SourceKind == "benchmark_pack", nil
}

func installedBenchmarkPackLiveTracks(
	root string,
	document installedSuiteDocument,
) (map[TrackID]struct{}, error) {
	isPack, err := installedSuiteIsBenchmarkPack(document.Manifest)
	if err != nil || !isPack {
		return nil, err
	}
	plans, err := installedVisibleCasePlans(root, document)
	if err != nil {
		return nil, err
	}
	labels := make(map[string]gradingCaseEvidence, len(plans))
	if err := scanInstalledSuiteRole(root, document.Manifest, "grading_cases", true, func(line []byte, lineNumber int) error {
		row, decodeErr := decodeInstalledGradingCase(line)
		if decodeErr != nil {
			return fmt.Errorf("%w: installed grading case line %d is invalid", ErrInvalid, lineNumber)
		}
		if _, planned := plans[row.CaseID]; !planned {
			return fmt.Errorf("%w: installed grading case is not present in the visible plan", ErrInvalid)
		}
		if _, duplicate := labels[row.CaseID]; duplicate {
			return fmt.Errorf("%w: installed grading case identity is duplicated", ErrInvalid)
		}
		labels[row.CaseID] = row
		return nil
	}); err != nil {
		return nil, err
	}
	if len(labels) != len(plans) {
		return nil, fmt.Errorf("%w: installed grading cases do not cover the visible plan", ErrInvalid)
	}

	allTrackCases := func(trackID TrackID, qualified func(gradingCaseEvidence) bool) bool {
		count := 0
		for caseID, plan := range plans {
			if !containsTrack(plan.TrackIDs, trackID) {
				continue
			}
			count++
			if !qualified(labels[caseID]) {
				return false
			}
		}
		return count > 0
	}
	admitted := make(map[TrackID]struct{}, len(document.Manifest.TrackIDs))
	for _, trackID := range document.Manifest.TrackIDs {
		if !benchmarkPackLiveTrack(trackID) {
			continue
		}
		switch trackID {
		case "routing":
			if allTrackCases(trackID, func(label gradingCaseEvidence) bool {
				return label.ExpectedRoute != nil && strings.TrimSpace(*label.ExpectedRoute) != ""
			}) {
				admitted[trackID] = struct{}{}
			}
		case "model_pool", "joint":
			if allTrackCases(trackID, func(label gradingCaseEvidence) bool {
				return label.ExpectedAnswer != nil && strings.TrimSpace(*label.ExpectedAnswer) != ""
			}) {
				admitted[trackID] = struct{}{}
			}
		case "capacity":
			if allTrackCases(trackID, func(gradingCaseEvidence) bool { return true }) {
				admitted[trackID] = struct{}{}
			}
		case "multimodal":
			if !allTrackCases(trackID, func(label gradingCaseEvidence) bool {
				return label.ExpectedAnswer != nil && strings.TrimSpace(*label.ExpectedAnswer) != ""
			}) {
				continue
			}
			eligible, eligibilityErr := installedMultimodalLiveSourceEligible(root, document)
			if eligibilityErr != nil {
				return nil, eligibilityErr
			}
			if eligible {
				admitted[trackID] = struct{}{}
			}
		}
	}
	return admitted, nil
}

// installedMultimodalLiveSourceEligible admits a complete hidden-answer image
// cohort from either the maintained MMR parser or a declarative Benchmark Pack.
// Imported observations remain E0; only a fresh brokered run can earn E4.
func installedMultimodalLiveSourceEligible(
	root string,
	document installedSuiteDocument,
) (bool, error) {
	manifest := document.Manifest
	if !containsTrack(manifest.TrackIDs, "multimodal") {
		return false, nil
	}
	isPack, err := installedSuiteIsBenchmarkPack(manifest)
	if err != nil {
		return false, err
	}
	var provenance unqualifiedSuiteEvidence
	if decodeErr := decodeExactJSON(manifest.QualificationReceipt.Qualification, &provenance); decodeErr != nil {
		return false, fmt.Errorf("%w: normalized suite parser qualification is invalid", ErrInvalid)
	}
	registeredMMR := manifest.AdapterID == "mmr-bench" && provenance.ParserVerified
	if !registeredMMR && !isPack {
		return false, nil
	}
	var artifacts map[string]json.RawMessage
	if decodeErr := json.Unmarshal(manifest.Artifacts, &artifacts); decodeErr != nil {
		return false, fmt.Errorf("%w: normalized suite artifact set is invalid", ErrInvalid)
	}
	if _, present := artifacts["media_manifest"]; !present {
		return false, nil
	}
	if _, present := artifacts["multimodal_observations"]; registeredMMR && !present {
		return false, nil
	}

	caseImages, err := installedMultimodalCaseImages(root, document)
	if err != nil {
		return false, err
	}
	media, err := installedMultimodalMedia(root, manifest)
	if err != nil {
		return false, err
	}
	if len(caseImages) == 0 || (registeredMMR && len(caseImages) != manifest.CaseCount) || len(media) == 0 {
		return false, fmt.Errorf("%w: normalized multimodal live cohort is incomplete", ErrInvalid)
	}
	usedMedia := make(map[string]struct{}, len(caseImages))
	for _, image := range caseImages {
		entry, present := media[image.digest]
		if !present || entry.mediaType != image.mediaType || entry.sizeBytes != image.sizeBytes {
			return false, fmt.Errorf("%w: normalized multimodal visible media is not bound to its manifest", ErrInvalid)
		}
		usedMedia[image.digest] = struct{}{}
	}
	if len(usedMedia) != len(media) {
		return false, fmt.Errorf("%w: normalized multimodal media manifest contains an unused object", ErrInvalid)
	}
	if err := validateInstalledMultimodalLabels(root, document, registeredMMR); err != nil {
		return false, err
	}
	return true, nil
}

type installedImageSubject struct {
	digest, mediaType string
	sizeBytes         int64
}

func installedMultimodalCaseImages(
	root string,
	document installedSuiteDocument,
) (map[string]installedImageSubject, error) {
	plans, err := installedVisibleCasePlans(root, document)
	if err != nil {
		return nil, err
	}
	images := make(map[string]installedImageSubject, len(plans))
	err = scanInstalledSuiteRole(root, document.Manifest, "visible_cases", true, func(line []byte, lineNumber int) error {
		var row visibleCaseIdentity
		if decodeErr := decodeStrictJSONLine(line, &row); decodeErr != nil {
			return fmt.Errorf("%w: installed multimodal visible line %d is invalid", ErrInvalid, lineNumber)
		}
		plan, planned := plans[row.ID]
		if !planned {
			return fmt.Errorf("%w: installed multimodal live cohort contains an unknown case", ErrInvalid)
		}
		if !containsTrack(plan.TrackIDs, "multimodal") {
			return nil
		}
		if plan.Modality != "image" {
			return fmt.Errorf("%w: installed multimodal live cohort contains a non-image case", ErrInvalid)
		}
		var image *installedImageSubject
		for _, message := range row.Messages {
			parts := make([]json.RawMessage, 0)
			if decodeErr := json.Unmarshal(message.Content, &parts); decodeErr != nil {
				continue
			}
			for _, encoded := range parts {
				var header struct {
					Type string `json:"type"`
				}
				if json.Unmarshal(encoded, &header) != nil || header.Type != "image_url" {
					continue
				}
				if image != nil {
					return fmt.Errorf("%w: installed multimodal live case must bind exactly one image", ErrInvalid)
				}
				var part imageContentPart
				if decodeErr := decodeStrictJSONLine(encoded, &part); decodeErr != nil {
					return fmt.Errorf("%w: installed multimodal image content is invalid", ErrInvalid)
				}
				subject, subjectErr := installedImageDataSubject(part.ImageURL.URL)
				if subjectErr != nil {
					return subjectErr
				}
				image = &subject
			}
		}
		if image == nil {
			return fmt.Errorf("%w: installed multimodal live case omits its image", ErrInvalid)
		}
		images[row.ID] = *image
		return nil
	})
	if err != nil {
		return nil, err
	}
	return images, nil
}

func installedImageDataSubject(value string) (installedImageSubject, error) {
	metadata, encoded, found := strings.Cut(value, ",")
	if !found || !strings.HasPrefix(metadata, "data:image/") || !strings.HasSuffix(metadata, ";base64") || encoded == "" {
		return installedImageSubject{}, fmt.Errorf("%w: installed multimodal image must be an inline image data URL", ErrInvalid)
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || len(data) == 0 {
		return installedImageSubject{}, fmt.Errorf("%w: installed multimodal image data is invalid", ErrInvalid)
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	return installedImageSubject{
		digest: digest, mediaType: strings.TrimPrefix(strings.TrimSuffix(metadata, ";base64"), "data:"),
		sizeBytes: int64(len(data)),
	}, nil
}

type installedMediaManifestEntry struct {
	SchemaVersion string  `json:"schema_version"`
	ID            string  `json:"id"`
	Digest        string  `json:"digest"`
	MediaType     string  `json:"media_type"`
	SizeBytes     int64   `json:"size_bytes"`
	Modality      string  `json:"modality"`
	LicenseID     *string `json:"license_id,omitempty"`
}

func installedMultimodalMedia(
	root string,
	manifest suiteManifestProjection,
) (map[string]installedImageSubject, error) {
	media := make(map[string]installedImageSubject)
	seenIDs := make(map[string]struct{})
	err := scanInstalledSuiteRole(root, manifest, "media_manifest", true, func(line []byte, lineNumber int) error {
		var row installedMediaManifestEntry
		if err := decodeStrictJSONLine(line, &row); err != nil || row.SchemaVersion != normalizedSuiteSchemaVersion ||
			!portableIDPattern.MatchString(row.ID) || !digestPattern.MatchString(row.Digest) || row.SizeBytes <= 0 ||
			row.Modality != "image" || !strings.HasPrefix(row.MediaType, "image/") ||
			(row.LicenseID != nil && !portableIDPattern.MatchString(*row.LicenseID)) {
			return fmt.Errorf("%w: installed multimodal media line %d is invalid", ErrInvalid, lineNumber)
		}
		if _, duplicate := seenIDs[row.ID]; duplicate {
			return fmt.Errorf("%w: installed multimodal media identity is duplicated", ErrInvalid)
		}
		seenIDs[row.ID] = struct{}{}
		subject := installedImageSubject{digest: row.Digest, mediaType: row.MediaType, sizeBytes: row.SizeBytes}
		if prior, duplicate := media[row.Digest]; duplicate && prior != subject {
			return fmt.Errorf("%w: installed multimodal media digest metadata conflicts", ErrInvalid)
		}
		media[row.Digest] = subject
		return nil
	})
	if err != nil {
		return nil, err
	}
	return media, nil
}

type installedMultimodalObservation struct {
	SchemaVersion      string   `json:"schema_version"`
	CaseID             string   `json:"case_id"`
	Modality           string   `json:"modality"`
	Supported          bool     `json:"supported"`
	Quality            *float64 `json:"quality,omitempty"`
	PrivacyViolations  int      `json:"privacy_violations"`
	SourceRecordDigest string   `json:"source_record_digest"`
}

func validateInstalledMultimodalLabels(
	root string,
	document installedSuiteDocument,
	requireObservations bool,
) error {
	plans, err := installedVisibleCasePlans(root, document)
	if err != nil {
		return err
	}
	multimodalCases := make(map[string]struct{})
	for caseID, plan := range plans {
		if containsTrack(plan.TrackIDs, "multimodal") {
			multimodalCases[caseID] = struct{}{}
		}
	}
	grading := make(map[string]struct{}, len(multimodalCases))
	if err := scanInstalledSuiteRole(root, document.Manifest, "grading_cases", true, func(line []byte, lineNumber int) error {
		row, err := decodeInstalledGradingCase(line)
		if err != nil {
			return fmt.Errorf("%w: installed multimodal grading line %d is invalid", ErrInvalid, lineNumber)
		}
		if _, planned := plans[row.CaseID]; !planned {
			return fmt.Errorf("%w: installed multimodal grading identity is unplanned", ErrInvalid)
		}
		if _, selected := multimodalCases[row.CaseID]; !selected {
			return nil
		}
		if row.ExpectedAnswer == nil || strings.TrimSpace(*row.ExpectedAnswer) == "" {
			return fmt.Errorf("%w: installed multimodal grading line %d lacks an exact hidden answer", ErrInvalid, lineNumber)
		}
		if _, duplicate := grading[row.CaseID]; duplicate {
			return fmt.Errorf("%w: installed multimodal grading identity is duplicated", ErrInvalid)
		}
		grading[row.CaseID] = struct{}{}
		return nil
	}); err != nil {
		return err
	}
	if len(grading) != len(multimodalCases) {
		return fmt.Errorf("%w: installed multimodal labels do not cover the exact case cohort", ErrInvalid)
	}
	if requireObservations {
		observations := make(map[string]struct{}, len(multimodalCases))
		if err := scanInstalledSuiteRole(root, document.Manifest, "multimodal_observations", true, func(line []byte, lineNumber int) error {
			var row installedMultimodalObservation
			if err := decodeStrictJSONLine(line, &row); err != nil || row.SchemaVersion != normalizedSuiteSchemaVersion ||
				row.Modality != "image" || row.PrivacyViolations < 0 || !digestPattern.MatchString(row.SourceRecordDigest) ||
				(row.Quality != nil && (!finiteFloat(*row.Quality) || *row.Quality < 0 || *row.Quality > 1)) {
				return fmt.Errorf("%w: installed multimodal observation line %d is invalid", ErrInvalid, lineNumber)
			}
			if _, planned := multimodalCases[row.CaseID]; !planned {
				return fmt.Errorf("%w: installed multimodal observation identity is unplanned", ErrInvalid)
			}
			if _, duplicate := observations[row.CaseID]; duplicate {
				return fmt.Errorf("%w: installed multimodal observation identity is duplicated", ErrInvalid)
			}
			observations[row.CaseID] = struct{}{}
			return nil
		}); err != nil {
			return err
		}
		if len(observations) != len(multimodalCases) {
			return fmt.Errorf("%w: installed multimodal observations do not cover the exact case cohort", ErrInvalid)
		}
	}
	return nil
}

func normalizedSuiteLiveMethodTracks(suite CatalogSuite) map[TrackID]struct{} {
	tracks := make(map[TrackID]struct{}, 2)
	for _, method := range suite.Methods {
		switch {
		case method.ID == declaredShiftLiveMethodID && method.TrackID == "routing" && method.Status == "configured" &&
			method.EvidenceSource == CatalogMethodEvidenceSourceServerBrokeredLive && len(method.QualifiedGateIDs) == 1 && method.QualifiedGateIDs[0] == "G4":
			tracks["routing"] = struct{}{}
		case method.ID == normalizedMultimodalLiveMethodID && method.TrackID == "multimodal" && method.Status == "configured" &&
			method.EvidenceSource == CatalogMethodEvidenceSourceLiveRuntime && len(method.QualifiedGateIDs) == 0:
			tracks["multimodal"] = struct{}{}
		case benchmarkPackLiveTrack(method.TrackID) && method.ID == benchmarkPackLiveMethodID(method.TrackID) &&
			method.Status == "configured" && method.EvidenceSource == CatalogMethodEvidenceSourceLiveRuntime &&
			len(method.QualifiedGateIDs) == 0:
			tracks[method.TrackID] = struct{}{}
		}
	}
	return tracks
}
