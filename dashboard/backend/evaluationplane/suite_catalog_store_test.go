package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type importedSuiteFixtureOptions struct {
	adapterID              string
	sourceKind             string
	sourceRevisionOverride string
	decisionUnit           string
	actionSpace            string
	trackIDs               []TrackID
	evidenceLevel          EvidenceLevel
	origin                 string
	parserVerified         bool
	nativeRunAttested      bool
	promotionEligible      bool
	gradingCaseOverrides   map[string]any
	visibleCaseBytes       []byte
	gradingCaseBytes       []byte
	perturbationBytes      []byte
	multimodalBytes        []byte
	mediaManifestBytes     []byte
	caseCount              int
	armIDs                 []string
}

func normalizedImportedSuiteFixtureOptions(
	t *testing.T,
	custom []importedSuiteFixtureOptions,
) importedSuiteFixtureOptions {
	t.Helper()
	options := importedSuiteFixtureOptions{
		adapterID:     "routerarena",
		sourceKind:    "registered_adapter",
		trackIDs:      []TrackID{"routing"},
		evidenceLevel: "E0",
		origin:        "user_provided_import",
	}
	if len(custom) > 1 {
		t.Fatal("only one imported suite fixture options value is accepted")
	}
	if len(custom) == 1 {
		options = custom[0]
	}
	if options.evidenceLevel == "" {
		options.evidenceLevel = "E0"
	}
	if options.origin == "" {
		options.origin = "user_provided_import"
	}
	if options.sourceKind == "" {
		options.sourceKind = "registered_adapter"
	}
	if options.armIDs == nil {
		options.armIDs = []string{}
	}
	return options
}

func importedSuiteFixtureCases(
	t *testing.T,
	options importedSuiteFixtureOptions,
) ([]byte, []byte) {
	t.Helper()
	visibleModality := "text"
	visibleContent := any("private")
	if containsTrack(options.trackIDs, "multimodal") {
		visibleModality = "image"
		visibleContent = []map[string]any{{
			"type": "image_url", "image_url": map[string]any{
				"url": "data:image/png;base64,AA==", "detail": "low",
			},
		}}
	}
	visibleCase, err := json.Marshal(map[string]any{
		"schema_version": SchemaVersion,
		"id":             "case-1",
		"track_ids":      options.trackIDs,
		"messages":       []map[string]any{{"role": "user", "content": visibleContent}},
		"modality":       visibleModality,
		"tags":           []string{},
	})
	if err != nil {
		t.Fatalf("marshal visible suite fixture: %v", err)
	}
	gradingCase := map[string]any{
		"schema_version": SchemaVersion,
		"case_id":        "case-1",
		"weight":         1.0,
	}
	for field, value := range options.gradingCaseOverrides {
		gradingCase[field] = value
	}
	gradingCaseBytes, err := json.Marshal(gradingCase)
	if err != nil {
		t.Fatalf("marshal grading suite fixture: %v", err)
	}
	return append(visibleCase, '\n'), append(gradingCaseBytes, '\n')
}

func writeImportedSuiteFixtureArtifacts(
	t *testing.T,
	root string,
	visibleCase []byte,
	gradingCase []byte,
	options importedSuiteFixtureOptions,
) map[string]any {
	t.Helper()
	artifacts := map[string]any{}
	type fixtureArtifactContent struct {
		domain, mediaType string
		data              []byte
	}
	contents := map[string]fixtureArtifactContent{
		"visible_cases":    {"visible", "application/x-ndjson", visibleCase},
		"grading_cases":    {"grading", "application/x-ndjson", gradingCase},
		"decisions":        {"grading", "application/x-ndjson", []byte("{\"schema_version\":\"evaluation-suite.v1\",\"case_id\":\"case-1\",\"selected_arm_id\":\"arm-a\",\"selection_status\":\"selected\",\"success\":true,\"fallback\":false,\"source_record_digest\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}\n")},
		"license_manifest": {"metadata", "application/json", []byte("{\"schema_version\":\"evaluation-suite-license.v1\",\"licenses\":[{\"id\":\"upstream\",\"name\":\"fixture\",\"redistribution\":\"metadata_only\"}]}")},
	}
	if len(options.multimodalBytes) != 0 {
		contents["multimodal_observations"] = fixtureArtifactContent{"grading", "application/x-ndjson", options.multimodalBytes}
	}
	if len(options.mediaManifestBytes) != 0 {
		contents["media_manifest"] = fixtureArtifactContent{"metadata", "application/x-ndjson", options.mediaManifestBytes}
	}
	for role, content := range contents {
		digest := suiteDocumentDigest(content.data)
		path := filepath.Join(root, "objects", content.domain, "sha256", digest[len("sha256:"):])
		if err := os.WriteFile(path, content.data, 0o600); err != nil {
			t.Fatalf("write suite object: %v", err)
		}
		artifacts[role] = map[string]any{
			"schema_version": SchemaVersion, "digest": digest,
			"media_type": content.mediaType, "size_bytes": len(content.data),
		}
	}
	return artifacts
}

func writeImportedSuiteFixture(t *testing.T, root, suiteID string, custom ...importedSuiteFixtureOptions) string {
	t.Helper()
	options := normalizedImportedSuiteFixtureOptions(t, custom)
	visibleCase, gradingCaseBytes := importedSuiteFixtureCases(t, options)
	if len(options.visibleCaseBytes) != 0 {
		visibleCase = options.visibleCaseBytes
	}
	if len(options.gradingCaseBytes) != 0 {
		gradingCaseBytes = options.gradingCaseBytes
	}
	contract, knownAdapter := normalizedAdapterContracts[options.adapterID]
	if !knownAdapter {
		if options.sourceKind != "benchmark_pack" || options.sourceRevisionOverride == "" ||
			options.decisionUnit == "" || options.actionSpace == "" {
			t.Fatalf("unknown normalized fixture adapter %q", options.adapterID)
		}
		contract = normalizedAdapterContract{
			sourceRevision: options.sourceRevisionOverride,
			decisionUnit:   options.decisionUnit, actionSpace: options.actionSpace,
			trackIDs: append([]TrackID(nil), options.trackIDs...),
		}
	}
	sourceRevision := contract.sourceRevision
	if options.sourceRevisionOverride != "" {
		sourceRevision = options.sourceRevisionOverride
	}
	artifacts := writeImportedSuiteFixtureArtifacts(t, root, visibleCase, gradingCaseBytes, options)
	addImportedSuitePerturbation(t, root, options.perturbationBytes, artifacts)
	manifest := importedSuiteFixtureManifest(
		t, suiteID, options, contract, sourceRevision, artifacts,
	)
	return writeImportedSuiteManifest(t, root, suiteID, manifest)
}

func addImportedSuitePerturbation(
	t *testing.T,
	root string,
	content []byte,
	artifacts map[string]any,
) {
	t.Helper()
	if len(content) == 0 {
		return
	}
	digest := suiteDocumentDigest(content)
	path := filepath.Join(root, "objects", "grading", "sha256", digest[len("sha256:"):])
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write perturbation suite object: %v", err)
	}
	artifacts["perturbations"] = map[string]any{
		"schema_version": SchemaVersion, "digest": digest,
		"media_type": "application/x-ndjson", "size_bytes": len(content),
	}
}

func importedSuiteFixtureManifest(
	t *testing.T,
	suiteID string,
	options importedSuiteFixtureOptions,
	contract normalizedAdapterContract,
	sourceRevision string,
	artifacts map[string]any,
) map[string]any {
	t.Helper()
	source := map[string]any{
		"schema_version": benchmarkSourceContractVersion, "source_kind": options.sourceKind, "adapter_id": options.adapterID,
		"expected_source_revision": sourceRevision,
		"observed_source_revision": sourceRevision,
		"source_clean":             true, "verified": true,
	}
	if contract.datasetRevision != "" {
		source["expected_dataset_revision"] = contract.datasetRevision
		source["observed_dataset_revision"] = contract.datasetRevision
		source["dataset_clean"] = true
	}
	manifest := map[string]any{
		"schema_version": normalizedSuiteSchemaVersion,
		"id":             suiteID, "name": "Imported normalized suite fixture", "adapter_id": options.adapterID,
		"adapter_contract_version": adapterContractVersion, "source_receipt": source,
		"decision_unit": contract.decisionUnit, "action_space": contract.actionSpace, "track_ids": options.trackIDs,
		"split_protocol": "fixed test split", "case_count": fixtureCaseCount(options), "arm_ids": options.armIDs,
		"data_classification": "restricted", "redistribution": "metadata_only",
		"artifacts": artifacts, "limitations": []string{"test only"},
	}
	subjectDigest, subjectDigestErr := canonicalValueDigest(manifest)
	if subjectDigestErr != nil {
		t.Fatalf("subject digest: %v", subjectDigestErr)
	}
	sourceDigest, _ := canonicalValueDigest(source)
	artifactDigest, _ := canonicalValueDigest(artifacts)
	qualification := map[string]any{
		"schema_version":            suiteQualificationContractVersion,
		"status":                    "exploratory_import",
		"origin":                    options.origin,
		"parser_verified":           options.parserVerified,
		"native_execution_attested": options.nativeRunAttested,
		"promotion_eligible":        options.promotionEligible,
	}
	manifest["qualification_receipt"] = map[string]any{
		"schema_version": suiteQualificationContractVersion, "evidence_level": options.evidenceLevel,
		"manifest_subject_digest": subjectDigest, "source_receipt_digest": sourceDigest,
		"artifact_set_digest": artifactDigest, "executor_id": normalizedSuiteExecutorID,
		"executor_digest": normalizedSuiteExecutorDigest,
		"qualification":   qualification,
	}
	return manifest
}

func writeImportedSuiteManifest(
	t *testing.T,
	root string,
	suiteID string,
	manifest map[string]any,
) string {
	t.Helper()
	revision, revisionErr := canonicalValueDigest(manifest)
	if revisionErr != nil {
		t.Fatalf("suite revision: %v", revisionErr)
	}
	manifest["revision"] = revision
	manifestBytes, manifestJSONErr := canonicalJSON(manifest)
	if manifestJSONErr != nil {
		t.Fatalf("manifest JSON: %v", manifestJSONErr)
	}
	var roundTripped map[string]any
	if unmarshalErr := json.Unmarshal(manifestBytes, &roundTripped); unmarshalErr != nil {
		t.Fatalf("round trip suite manifest: %v", unmarshalErr)
	}
	delete(roundTripped, "revision")
	roundTrippedRevision, roundTripErr := canonicalValueDigest(roundTripped)
	if roundTripErr != nil || roundTrippedRevision != revision {
		t.Fatalf("fixture revision drift: initial=%s round-trip=%s error=%v", revision, roundTrippedRevision, roundTripErr)
	}
	manifestDigest := suiteDocumentDigest(manifestBytes)
	manifestPath := filepath.Join(root, "manifests", "sha256", manifestDigest[len("sha256:"):])
	if writeErr := os.WriteFile(manifestPath, manifestBytes, 0o600); writeErr != nil {
		t.Fatalf("write suite manifest: %v", writeErr)
	}
	indexBytes, indexJSONErr := canonicalJSON(suiteIndexRecord{
		ID: suiteID, Revision: revision, ManifestDigest: manifestDigest, ManifestSizeBytes: int64(len(manifestBytes)),
	})
	if indexJSONErr != nil {
		t.Fatalf("index JSON: %v", indexJSONErr)
	}
	if writeErr := os.WriteFile(filepath.Join(root, "index", suiteID+".json"), indexBytes, 0o600); writeErr != nil {
		t.Fatalf("write suite index: %v", writeErr)
	}
	return revision
}

func fixtureCaseCount(options importedSuiteFixtureOptions) int {
	if options.caseCount > 0 {
		return options.caseCount
	}
	return 1
}

func declaredShiftCatalogFixtureOptions(t *testing.T, parserVerified bool, sourceCaseID string) importedSuiteFixtureOptions {
	t.Helper()
	visible := testJSONLines(t,
		map[string]any{
			"schema_version": SchemaVersion, "id": "source", "track_ids": []TrackID{"routing"},
			"messages": []map[string]any{{"role": "user", "content": "source"}}, "modality": "text", "tags": []string{},
		},
		map[string]any{
			"schema_version": SchemaVersion, "id": "perturbed", "track_ids": []TrackID{"routing"},
			"messages": []map[string]any{{"role": "user", "content": "perturbed"}}, "modality": "text", "tags": []string{},
		},
	)
	grading := testJSONLines(t,
		map[string]any{"schema_version": SchemaVersion, "case_id": "source", "weight": 1.0},
		map[string]any{"schema_version": SchemaVersion, "case_id": "perturbed", "weight": 1.0},
	)
	perturbations := testJSONLines(t, map[string]any{
		"schema_version": normalizedSuiteSchemaVersion, "pair_id": "pair-1",
		"source_case_id": sourceCaseID, "perturbed_case_id": "perturbed", "relation": "invariant",
		"slice_ids": []string{"declared:paraphrase"}, "native_pair_count": 1,
		"source_record_digest": digestString("declared-shift-catalog-source"),
	})
	origin := "user_provided_import"
	if parserVerified {
		origin = "registered_parser_import"
	}
	return importedSuiteFixtureOptions{
		adapterID: "routerarena", trackIDs: []TrackID{"routing"}, origin: origin,
		parserVerified: parserVerified, visibleCaseBytes: visible, gradingCaseBytes: grading,
		perturbationBytes: perturbations, caseCount: 2,
	}
}

func TestInstalledCatalogAdmitsOnlyQualifiedServerLiveDeclaredShift(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(
		t, service.registrySource.suiteStorePath, "qualified-declared-shift",
		declaredShiftCatalogFixtureOptions(t, true, "source"),
	)
	writeImportedSuiteFixture(
		t, service.registrySource.suiteStorePath, "unverified-declared-shift",
		declaredShiftCatalogFixtureOptions(t, false, "source"),
	)

	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	methodsBySuite := make(map[string]map[string]CatalogMethod)
	suitesByID := make(map[string]CatalogSuite)
	for _, suite := range catalog.Suites {
		methods := make(map[string]CatalogMethod, len(suite.Methods))
		for _, method := range suite.Methods {
			methods[method.ID] = method
		}
		methodsBySuite[suite.ID] = methods
		suitesByID[suite.ID] = suite
	}
	qualified := methodsBySuite["qualified-declared-shift"][declaredShiftLiveMethodID]
	if qualified.ID != declaredShiftLiveMethodID || qualified.TrackID != "routing" ||
		qualified.EvidenceSource != "server_brokered_live" || qualified.Status != "configured" ||
		len(qualified.QualifiedGateIDs) != 1 || qualified.QualifiedGateIDs[0] != "G4" {
		t.Fatalf("qualified declared-shift method is not exact: %+v", qualified)
	}
	qualifiedSuite := suitesByID["qualified-declared-shift"]
	if qualifiedSuite.Executors[ModeLive] != normalizedSuiteLiveExecutorID ||
		!containsMode(qualifiedSuite.Modes, ModeLive) {
		t.Fatalf("qualified declared-shift source is not live reachable: %+v", qualifiedSuite)
	}
	if _, present := methodsBySuite["unverified-declared-shift"][declaredShiftLiveMethodID]; present {
		t.Fatal("an unverified normalized import advertised the server-live declared-shift method")
	}
	if _, present := suitesByID["unverified-declared-shift"].Executors[ModeLive]; present ||
		containsMode(suitesByID["unverified-declared-shift"].Modes, ModeLive) {
		t.Fatal("an unverified normalized import advertised live execution")
	}
	for _, suiteID := range []string{"qualified-declared-shift", "unverified-declared-shift"} {
		method := methodsBySuite[suiteID]["normalized.routerarena.routing.v1"]
		if method.EvidenceSource != "normalized_import" || len(method.QualifiedGateIDs) != 0 || method.Status != "configured" {
			t.Fatalf("suite %q promoted its normalized import method: %+v", suiteID, method)
		}
	}
}

func TestInstalledImportReadinessDoesNotInheritResearchNativeStatus(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "xroute-import", importedSuiteFixtureOptions{
		adapterID: "xroutebench", trackIDs: []TrackID{"model_pool"},
	})
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	for _, suite := range catalog.Suites {
		if suite.ID != "xroute-import" {
			continue
		}
		if len(suite.Methods) != 1 || suite.Methods[0].ID != "normalized.xroutebench.model_pool.v1" ||
			suite.Methods[0].Status != "configured" || suite.Methods[0].EvidenceSource != "normalized_import" ||
			len(suite.Methods[0].QualifiedGateIDs) != 0 || suite.Methods[0].Reason != "" {
			t.Fatalf("installed import inherited research-native readiness: %+v", suite.Methods)
		}
		return
	}
	t.Fatal("installed xroute import is missing")
}

func TestInstalledCatalogLoadsADataOnlyBenchmarkPack(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "acme-routing-pack", importedSuiteFixtureOptions{
		adapterID: "acme.routing", sourceKind: "benchmark_pack", sourceRevisionOverride: strings.Repeat("d", 40),
		decisionUnit: "request", actionSpace: "one model", trackIDs: []TrackID{"routing"},
		gradingCaseOverrides: map[string]any{"expected_route": "arm-a"},
	})
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	for _, suite := range catalog.Suites {
		if suite.ID != "acme-routing-pack" {
			continue
		}
		if len(suite.Methods) != 2 || suite.Methods[0].ID != "normalized.acme.routing.routing.v1" ||
			suite.Methods[0].EvidenceSource != CatalogMethodEvidenceSourceNormalizedImport ||
			suite.Methods[0].Status != "configured" || len(suite.Methods[0].QualifiedGateIDs) != 0 ||
			suite.Methods[1].ID != benchmarkPackLiveMethodID("routing") ||
			suite.Methods[1].EvidenceSource != CatalogMethodEvidenceSourceLiveRuntime ||
			suite.Methods[1].Status != "configured" || len(suite.Methods[1].QualifiedGateIDs) != 0 ||
			len(suite.Modes) != 2 || suite.Modes[0] != ModeReplay || suite.Modes[1] != ModeLive ||
			suite.Executors[ModeLive] != normalizedSuiteLiveExecutorID {
			t.Fatalf("benchmark pack catalog projection is invalid: %+v", suite)
		}
		return
	}
	t.Fatal("installed benchmark pack is missing")
}

func TestInstalledBenchmarkPackWithoutHiddenLabelsStaysReplayOnly(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "ungraded-routing-pack", importedSuiteFixtureOptions{
		adapterID: "acme.ungraded", sourceKind: "benchmark_pack", sourceRevisionOverride: strings.Repeat("e", 40),
		decisionUnit: "request", actionSpace: "one model", trackIDs: []TrackID{"routing"},
	})
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	for _, suite := range catalog.Suites {
		if suite.ID != "ungraded-routing-pack" {
			continue
		}
		if len(suite.Methods) != 1 || suite.Methods[0].ID != "normalized.acme.ungraded.routing.v1" ||
			len(suite.Modes) != 1 || suite.Modes[0] != ModeReplay {
			t.Fatalf("ungraded benchmark pack advertised live execution: %+v", suite)
		}
		return
	}
	t.Fatal("installed ungraded benchmark pack is missing")
}

func TestInstalledBenchmarkPackWithoutMultimodalAnswerStaysReplayOnly(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "ungraded-multimodal-pack", importedSuiteFixtureOptions{
		adapterID: "acme.ungraded-multimodal", sourceKind: "benchmark_pack", sourceRevisionOverride: strings.Repeat("e", 40),
		decisionUnit: "request", actionSpace: "one answer", trackIDs: []TrackID{"multimodal"},
	})
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	for _, suite := range catalog.Suites {
		if suite.ID != "ungraded-multimodal-pack" {
			continue
		}
		if len(suite.Methods) != 1 || suite.Methods[0].ID != "normalized.acme.ungraded-multimodal.multimodal.v1" ||
			len(suite.Modes) != 1 || suite.Modes[0] != ModeReplay {
			t.Fatalf("ungraded multimodal pack advertised live execution: %+v", suite)
		}
		return
	}
	t.Fatal("installed ungraded multimodal benchmark pack is missing")
}

func TestInstalledBenchmarkPackAdvertisesOnlyCompletePlatformLiveTracks(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "all-track-pack", importedSuiteFixtureOptions{
		adapterID: "acme.all-tracks", sourceKind: "benchmark_pack", sourceRevisionOverride: strings.Repeat("f", 40),
		decisionUnit: "request", actionSpace: "one model", trackIDs: append([]TrackID(nil), allTrackIDs...),
		gradingCaseOverrides: map[string]any{"expected_route": "arm-a", "expected_answer": "answer"},
		mediaManifestBytes: testJSONLines(t, map[string]any{
			"schema_version": normalizedSuiteSchemaVersion, "id": "fixture-image",
			"digest": suiteDocumentDigest([]byte{0}), "media_type": "image/png", "size_bytes": 1,
			"modality": "image", "license_id": "upstream",
		}),
	})
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	for _, suite := range catalog.Suites {
		if suite.ID != "all-track-pack" {
			continue
		}
		liveTracks := make(map[TrackID]struct{})
		for _, method := range suite.Methods {
			if strings.HasPrefix(method.ID, benchmarkPackLiveMethodPrefix+".") {
				liveTracks[method.TrackID] = struct{}{}
			}
		}
		for _, trackID := range []TrackID{"routing", "model_pool", "joint", "multimodal", "capacity"} {
			if _, present := liveTracks[trackID]; !present {
				t.Fatalf("complete benchmark pack omitted live track %q: %+v", trackID, suite.Methods)
			}
		}
		for _, trackID := range []TrackID{"agentic", "preference", "safety"} {
			if _, present := liveTracks[trackID]; present {
				t.Fatalf("benchmark pack advertised unsupported live track %q", trackID)
			}
		}
		if len(liveTracks) != 5 {
			t.Fatalf("benchmark pack live tracks=%+v, want exactly five", liveTracks)
		}
		if len(suite.Modes) != 2 || suite.Modes[0] != ModeReplay || suite.Modes[1] != ModeLive {
			t.Fatalf("mixed benchmark pack modes=%+v, want replay and live", suite.Modes)
		}
		return
	}
	t.Fatal("installed all-track benchmark pack is missing")
}

func TestInstalledCatalogRejectsQualifiedDeclaredShiftWithUnknownPairCase(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(
		t, service.registrySource.suiteStorePath, "invalid-declared-shift",
		declaredShiftCatalogFixtureOptions(t, true, "missing-source"),
	)
	if _, err := service.Catalog(); err == nil {
		t.Fatal("a parser-qualified declared-shift artifact referencing an unknown case was accepted")
	}
}

func multimodalLiveCatalogFixtureOptions(t *testing.T, parserVerified bool) importedSuiteFixtureOptions {
	t.Helper()
	origin := "user_provided_import"
	if parserVerified {
		origin = "registered_parser_import"
	}
	return importedSuiteFixtureOptions{
		adapterID: "mmr-bench", trackIDs: []TrackID{"model_pool", "multimodal"},
		origin: origin, parserVerified: parserVerified,
		gradingCaseOverrides: map[string]any{"expected_answer": "fixture-answer"},
		multimodalBytes: testJSONLines(t, map[string]any{
			"schema_version": normalizedSuiteSchemaVersion, "case_id": "case-1", "modality": "image",
			"supported": true, "quality": 1.0, "privacy_violations": 0,
			"source_record_digest": digestString("mmr-observation"),
		}),
		mediaManifestBytes: testJSONLines(t, map[string]any{
			"schema_version": normalizedSuiteSchemaVersion, "id": "fixture-image",
			"digest": suiteDocumentDigest([]byte{0}), "media_type": "image/png", "size_bytes": 1,
			"modality": "image", "license_id": "upstream",
		}),
	}
}

func TestInstalledCatalogAdmitsOnlyExactParserVerifiedMultimodalLiveCohort(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "qualified-mmr", multimodalLiveCatalogFixtureOptions(t, true))
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "unverified-mmr", multimodalLiveCatalogFixtureOptions(t, false))

	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	byID := make(map[string]CatalogSuite)
	for _, suite := range catalog.Suites {
		byID[suite.ID] = suite
	}
	qualified := byID["qualified-mmr"]
	if qualified.Executors[ModeLive] != normalizedSuiteLiveExecutorID || !containsMode(qualified.Modes, ModeLive) {
		t.Fatalf("qualified MMR cohort is not live reachable: %+v", qualified)
	}
	methodFound := false
	for _, method := range qualified.Methods {
		if method.ID == normalizedMultimodalLiveMethodID {
			methodFound = method.TrackID == "multimodal" && method.Status == "configured" &&
				method.EvidenceSource == "live_runtime" && len(method.QualifiedGateIDs) == 0
		}
	}
	if !methodFound {
		t.Fatalf("qualified MMR cohort omitted its exact server-live method: %+v", qualified.Methods)
	}
	if _, present := byID["unverified-mmr"].Executors[ModeLive]; present ||
		containsMode(byID["unverified-mmr"].Modes, ModeLive) {
		t.Fatal("an unverified MMR import advertised live execution")
	}
}

func TestInstalledMultimodalLiveAdmissionFailsClosedOnMissingHiddenAnswer(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	options := multimodalLiveCatalogFixtureOptions(t, true)
	options.gradingCaseOverrides = nil
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "mmr-without-answer", options)
	if _, err := service.Catalog(); err == nil {
		t.Fatal("parser-qualified multimodal live source without a hidden answer was accepted")
	}
}

func TestInstalledNormalizedLiveCreateAdmissionIsTrackExact(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	if err := os.WriteFile(service.registrySource.configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write Mixture-of-Models config: %v", err)
	}
	writeImportedSuiteFixture(
		t, service.registrySource.suiteStorePath, "qualified-declared-shift",
		declaredShiftCatalogFixtureOptions(t, true, "source"),
	)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "qualified-mmr", multimodalLiveCatalogFixtureOptions(t, true))
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "routing-pack", importedSuiteFixtureOptions{
		adapterID: "acme.routing", sourceKind: "benchmark_pack", sourceRevisionOverride: strings.Repeat("d", 40),
		decisionUnit: "request", actionSpace: "one model", trackIDs: []TrackID{"routing"},
		gradingCaseOverrides: map[string]any{"expected_route": "Org/Fast Model"},
	})

	declaredShift := validCreateRequest()
	declaredShift.ClientRequestID = newTestClientRequestID()
	declaredShift.Name = "qualified declared shift"
	declaredShift.SuiteIDs = []string{"qualified-declared-shift"}
	declaredShift.TrackIDs = []TrackID{"routing"}
	declaredShift.Mode = ModeLive
	declaredShift.TargetID = mixtureTargetID("default")
	declaredShift.ChangeProfile = "recipe"
	declaredShift.SampleLimit = 2
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), declaredShift); err != nil {
		t.Fatalf("CreateRun qualified declared shift: %v", err)
	}

	multimodal := declaredShift
	multimodal.ClientRequestID = newTestClientRequestID()
	multimodal.Name = "qualified multimodal fidelity cohort"
	multimodal.SuiteIDs = []string{"qualified-mmr"}
	multimodal.TrackIDs = []TrackID{"multimodal"}
	multimodal.ChangeProfile = "agent_multimodal"
	multimodal.SampleLimit = 1
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), multimodal); err != nil {
		t.Fatalf("CreateRun qualified multimodal live cohort: %v", err)
	}

	pack := declaredShift
	pack.ClientRequestID = newTestClientRequestID()
	pack.Name = "declarative benchmark pack"
	pack.SuiteIDs = []string{"routing-pack"}
	pack.SampleLimit = 1
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), pack); err != nil {
		t.Fatalf("CreateRun declarative benchmark pack: %v", err)
	}

	modelPool := multimodal
	modelPool.ClientRequestID = newTestClientRequestID()
	modelPool.Name = "unregistered normalized model pool live method"
	modelPool.TrackIDs = []TrackID{"model_pool"}
	modelPool.ChangeProfile = "model_pool"
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), modelPool); err == nil {
		t.Fatal("MMR live admission leaked from multimodal into model_pool")
	}
}

func TestInstalledSuiteCatalogAndCreateFreezeSameExecutor(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	if err := os.WriteFile(service.registrySource.configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write Mixture-of-Models config: %v", err)
	}
	revision := writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "imported.routing")

	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	var installed *CatalogSuite
	for index := range catalog.Suites {
		if catalog.Suites[index].ID == "imported.routing" {
			installed = &catalog.Suites[index]
			break
		}
	}
	if installed == nil || installed.Executors[ModeReplay] != normalizedSuiteExecutorID ||
		installed.Revision != revision || installed.EvidenceLevel != "E0" {
		t.Fatalf("installed suite is not executable catalog evidence: %+v", installed)
	}

	request := validCreateRequest()
	request.SuiteIDs = []string{"imported.routing"}
	request.TrackIDs = []TrackID{"routing"}
	request.TargetID = "benchmark-source"
	request.SampleLimit = 1
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	manifest, _, err := service.readDurableManifest(run.ID)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.SuiteRevisions["imported.routing"] != revision ||
		manifest.SuiteExecutors["imported.routing"] != normalizedSuiteExecutorID {
		t.Fatalf("manifest did not freeze installed suite: %+v", manifest)
	}

	liveRequest := request
	liveRequest.ClientRequestID = newTestClientRequestID()
	liveRequest.Mode = ModeLive
	liveRequest.TargetID = mixtureTargetID("default")
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), liveRequest); err == nil {
		t.Fatal("exploratory normalized import exposed an unrunnable live mode")
	}
	wrongReplayTarget := request
	wrongReplayTarget.ClientRequestID = newTestClientRequestID()
	wrongReplayTarget.TargetID = mixtureTargetID("default")
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), wrongReplayTarget); err == nil {
		t.Fatal("normalized historical replay accepted the runtime target")
	}

	mixed := validCreateRequest()
	mixed.ClientRequestID = "17d3828d-cfc0-4416-8e67-f639c1ab11b0"
	mixed.SuiteIDs = []string{"evaluation-smoke", "imported.routing"}
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), mixed); err == nil {
		t.Fatal("builtin and installed executor suites were mixed")
	}
}

func TestInstalledSuiteCatalogRejectsImportProvenanceTamper(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	writeImportedSuiteFixture(t, service.registrySource.suiteStorePath, "tampered-routing")
	indexPath := filepath.Join(service.registrySource.suiteStorePath, "index", "tampered-routing.json")
	indexBytes, indexReadErr := os.ReadFile(indexPath)
	if indexReadErr != nil {
		t.Fatal(indexReadErr)
	}
	var index suiteIndexRecord
	if unmarshalErr := json.Unmarshal(indexBytes, &index); unmarshalErr != nil {
		t.Fatal(unmarshalErr)
	}
	manifestPath := filepath.Join(service.registrySource.suiteStorePath, "manifests", "sha256", index.ManifestDigest[len("sha256:"):])
	manifestBytes, manifestReadErr := os.ReadFile(manifestPath)
	if manifestReadErr != nil {
		t.Fatal(manifestReadErr)
	}
	manifestBytes = bytes.Replace(manifestBytes, []byte(normalizedSuiteExecutorDigest), []byte("sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"), 1)
	if err := os.WriteFile(manifestPath, manifestBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Catalog(); err == nil {
		t.Fatal("tampered qualification receipt was accepted")
	}
}
