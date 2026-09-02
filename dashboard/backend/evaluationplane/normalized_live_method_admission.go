package evaluationplane

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

const normalizedMultimodalLiveMethodID = "multimodal.hidden-answer.server-live.v1"

// installedFirstPartyNormalizedLiveMethods is the only admission registry for
// installed-suite live execution. Source imports remain replay-only E0
// evidence; these methods merely make a fresh server-owned protocol runnable.
func installedFirstPartyNormalizedLiveMethods(
	root string,
	manifest suiteManifestProjection,
) ([]CatalogMethod, error) {
	document := installedSuiteDocument{Manifest: manifest}
	methods := make([]CatalogMethod, 0, 2)
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
	multimodal, err := installedMultimodalLiveSourceEligible(root, document)
	if err != nil {
		return nil, fmt.Errorf("%w: installed suite multimodal live qualification is invalid", err)
	}
	if multimodal {
		methods = append(methods, CatalogMethod{
			ID: normalizedMultimodalLiveMethodID, TrackID: "multimodal",
			QualifiedGateIDs: []string{}, EvidenceSource: CatalogMethodEvidenceSourceLiveRuntime, Status: "configured",
		})
	}
	return methods, nil
}

// installedMultimodalLiveSourceEligible admits only the maintained MMR
// parser's exact hidden-answer image cohort. The imported observations remain
// E0; a live run earns E4 only later from complete broker receipts and the
// server-side hidden grader.
func installedMultimodalLiveSourceEligible(
	root string,
	document installedSuiteDocument,
) (bool, error) {
	manifest := document.Manifest
	if manifest.AdapterID != "mmr-bench" || !containsTrack(manifest.TrackIDs, "multimodal") {
		return false, nil
	}
	var provenance unqualifiedSuiteEvidence
	if err := decodeExactJSON(manifest.QualificationReceipt.Qualification, &provenance); err != nil {
		return false, fmt.Errorf("%w: normalized suite parser qualification is invalid", ErrInvalid)
	}
	if !provenance.ParserVerified {
		return false, nil
	}
	var artifacts map[string]json.RawMessage
	if err := json.Unmarshal(manifest.Artifacts, &artifacts); err != nil {
		return false, fmt.Errorf("%w: normalized suite artifact set is invalid", ErrInvalid)
	}
	for _, role := range []string{"media_manifest", "multimodal_observations"} {
		if _, present := artifacts[role]; !present {
			return false, nil
		}
	}

	caseImages, err := installedMultimodalCaseImages(root, document)
	if err != nil {
		return false, err
	}
	media, err := installedMultimodalMedia(root, manifest)
	if err != nil {
		return false, err
	}
	if len(caseImages) != manifest.CaseCount || len(media) == 0 {
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
	if err := validateInstalledMultimodalLabels(root, document); err != nil {
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
		if !planned || plan.Modality != "image" || !containsTrack(plan.TrackIDs, "multimodal") {
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

func validateInstalledMultimodalLabels(root string, document installedSuiteDocument) error {
	plans, err := installedVisibleCasePlans(root, document)
	if err != nil {
		return err
	}
	grading := make(map[string]struct{}, len(plans))
	if err := scanInstalledSuiteRole(root, document.Manifest, "grading_cases", true, func(line []byte, lineNumber int) error {
		row, err := decodeInstalledGradingCase(line)
		if err != nil || row.ExpectedAnswer == nil || strings.TrimSpace(*row.ExpectedAnswer) == "" {
			return fmt.Errorf("%w: installed multimodal grading line %d lacks an exact hidden answer", ErrInvalid, lineNumber)
		}
		if _, planned := plans[row.CaseID]; !planned {
			return fmt.Errorf("%w: installed multimodal grading identity is unplanned", ErrInvalid)
		}
		if _, duplicate := grading[row.CaseID]; duplicate {
			return fmt.Errorf("%w: installed multimodal grading identity is duplicated", ErrInvalid)
		}
		grading[row.CaseID] = struct{}{}
		return nil
	}); err != nil {
		return err
	}
	observations := make(map[string]struct{}, len(plans))
	if err := scanInstalledSuiteRole(root, document.Manifest, "multimodal_observations", true, func(line []byte, lineNumber int) error {
		var row installedMultimodalObservation
		if err := decodeStrictJSONLine(line, &row); err != nil || row.SchemaVersion != normalizedSuiteSchemaVersion ||
			row.Modality != "image" || row.PrivacyViolations < 0 || !digestPattern.MatchString(row.SourceRecordDigest) ||
			(row.Quality != nil && (!finiteFloat(*row.Quality) || *row.Quality < 0 || *row.Quality > 1)) {
			return fmt.Errorf("%w: installed multimodal observation line %d is invalid", ErrInvalid, lineNumber)
		}
		if _, planned := plans[row.CaseID]; !planned {
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
	if len(grading) != len(plans) || len(observations) != len(plans) {
		return fmt.Errorf("%w: installed multimodal labels do not cover the exact case cohort", ErrInvalid)
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
		}
	}
	return tracks
}
