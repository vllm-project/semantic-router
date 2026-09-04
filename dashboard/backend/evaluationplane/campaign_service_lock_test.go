package evaluationplane

import (
	"context"
	"testing"
	"time"
)

func TestCreateCampaignDoesNotReenterEvidencePublicationLock(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-lock-owner", false)
	request := campaignV2Request("schema_adapter")
	for _, runID := range []string{request.GateBindings.G2RunID, request.GateBindings.G4RunID} {
		runRequest := validCreateRequest()
		runRequest.ClientRequestID = runID
		if _, err := service.CreateRunAs(context.Background(), owner, runRequest); err != nil {
			t.Fatalf("create campaign evidence owner fixture: %v", err)
		}
	}

	completed := make(chan error, 1)
	go func() {
		_, err := service.CreateCampaignAs(owner, request)
		completed <- err
	}()
	select {
	case err := <-completed:
		if err == nil {
			t.Fatal("pending run bundles unexpectedly satisfied campaign evidence")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("campaign creation deadlocked while refreshing the complete run ledger")
	}
}
