package evaluationplane

import (
	"errors"
	"reflect"
)

const campaignCohortSchemaVersion = "evaluation-campaign-cohort.v1"

// CampaignProtocol declares that a catalog suite can supply one Campaign
// comparison cohort. Its presence is the eligibility signal.
type CampaignProtocol struct {
	SchemaVersion string `json:"schema_version"`
	MinimumCases  int    `json:"minimum_cases"`
}

func validateCampaignProtocol(suite CatalogSuite) error {
	protocol := suite.CampaignProtocol
	if protocol == nil {
		return nil
	}
	requiredExecutors := map[Mode]string{
		ModeReplay: momReplayExecutorID,
		ModeLive:   liveRuntimeExecutorID,
	}
	if protocol.SchemaVersion != campaignCohortSchemaVersion || protocol.MinimumCases <= 0 ||
		!reflect.DeepEqual(suite.Modes, []Mode{ModeReplay, ModeLive}) ||
		!reflect.DeepEqual(suite.Executors, requiredExecutors) ||
		!reflect.DeepEqual(suite.TrackIDs, []TrackID{"routing", "model_pool", "joint"}) ||
		suite.EvidenceLevel != "E0" || suite.CaseCount <= 0 || protocol.MinimumCases > suite.CaseCount {
		return errors.New("campaign protocol is inconsistent with its evaluation suite")
	}
	return nil
}
