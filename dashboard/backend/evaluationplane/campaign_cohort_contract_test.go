package evaluationplane

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestMixtureContractRequiresAtLeastTwoFrozenArms(t *testing.T) {
	mixture := brokerTestMixture()
	mixture.ModelArms = mixture.ModelArms[:1]
	mixture.PoolDigest = modelPoolSnapshotDigest(mixture.ModelArms)
	mixture.Decisions[0].ArmIDs = []string{mixture.ModelArms[0].ID}

	if err := validateManifestMixtureContract(mixture); err == nil || !strings.Contains(err.Error(), "at least two") {
		t.Fatalf("single-arm Mixture contract error = %v, want at-least-two rejection", err)
	}
}

func TestCampaignProtocolRequiresAConsistentSuiteContract(t *testing.T) {
	registry, err := NewRegistry("", "")
	if err != nil {
		t.Fatal(err)
	}
	want := builtinSuites()[1]
	want.CampaignProtocol = &CampaignProtocol{
		SchemaVersion: campaignCohortSchemaVersion,
		MinimumCases:  campaignPairedMinimumCases,
	}
	tests := map[string]func(*CatalogSuite){
		"schema": func(suite *CatalogSuite) {
			suite.CampaignProtocol.SchemaVersion = "evaluation-campaign-cohort.v0"
		},
		"executor": func(suite *CatalogSuite) {
			suite.Executors[ModeReplay] = fixtureReplayExecutorID
		},
		"evidence":   func(suite *CatalogSuite) { suite.EvidenceLevel = "E1" },
		"case count": func(suite *CatalogSuite) { suite.CaseCount = 0 },
		"minimum": func(suite *CatalogSuite) {
			suite.CampaignProtocol.MinimumCases = 0
		},
		"oversized minimum": func(suite *CatalogSuite) {
			suite.CampaignProtocol.MinimumCases = suite.CaseCount + 1
		},
		"tracks": func(suite *CatalogSuite) {
			suite.TrackIDs = []TrackID{"routing", "model_pool"}
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			suite := copyCatalogSuite(want)
			suite.ID = "campaign-test-" + strings.ReplaceAll(name, " ", "-")
			mutate(&suite)
			if registerErr := registry.registerSuite(suite); registerErr == nil {
				t.Fatalf("invalid campaign suite was registered: %+v", suite)
			}
		})
	}
}

func TestCampaignProtocolIsPresentOnlyOnDeclaredSuitesAndRejectsLegacyFields(t *testing.T) {
	suites := builtinSuites()
	campaignSuite := copyCatalogSuite(suites[1])
	campaignSuite.CampaignProtocol = &CampaignProtocol{
		SchemaVersion: campaignCohortSchemaVersion,
		MinimumCases:  campaignPairedMinimumCases,
	}
	campaignJSON, err := json.Marshal(campaignSuite)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(campaignJSON, []byte(`"campaign_protocol"`)) ||
		bytes.Contains(campaignJSON, []byte(`"campaign_eligible"`)) ||
		bytes.Contains(campaignJSON, []byte(`"campaign_minimum_cases"`)) {
		t.Fatalf("campaign suite did not use the clean protocol contract: %s", campaignJSON)
	}
	nonCampaignJSON, err := json.Marshal(suites[1])
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(nonCampaignJSON, []byte(`"campaign_protocol"`)) {
		t.Fatalf("diagnostic suite emitted a campaign protocol: %s", nonCampaignJSON)
	}

	legacyJSON := bytes.Replace(
		campaignJSON,
		[]byte(`"campaign_protocol":`),
		[]byte(`"campaign_eligible":true,"campaign_minimum_cases":59,"campaign_protocol":`),
		1,
	)
	decoder := json.NewDecoder(bytes.NewReader(legacyJSON))
	decoder.DisallowUnknownFields()
	var decoded CatalogSuite
	if decodeErr := decoder.Decode(&decoded); decodeErr == nil ||
		!strings.Contains(decodeErr.Error(), "unknown field") {
		t.Fatalf("legacy campaign fields were not rejected: %v", decodeErr)
	}
}
