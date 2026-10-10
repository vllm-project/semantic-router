package testcases

import (
	"encoding/json"
	"os"
	"testing"
)

func TestDashboardOpenAPICheckAcceptsCommittedArtifact(t *testing.T) {
	body, err := os.ReadFile("../../dashboard/backend/router/dashboard.openapi.json")
	if err != nil {
		t.Fatal(err)
	}
	var document dashboardOpenAPIDocument
	if err := json.Unmarshal(body, &document); err != nil {
		t.Fatal(err)
	}
	if err := checkDashboardOpenAPIDocument(document); err != nil {
		t.Fatal(err)
	}

	delete(document.Paths, "/api/settings")
	if err := checkDashboardOpenAPIDocument(document); err == nil {
		t.Fatal("check accepted a document without /api/settings")
	}
}
