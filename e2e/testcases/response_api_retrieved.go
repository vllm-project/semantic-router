package testcases

import (
	"fmt"
	"reflect"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func validateRetrievedResponse(created, retrieved *fixtures.ResponseAPIResponse) error {
	if created == nil || retrieved == nil || created.ID == "" || retrieved.ID != created.ID ||
		retrieved.Object != "response" || retrieved.Status != created.Status ||
		retrieved.PreviousResponseID != created.PreviousResponseID {
		return fmt.Errorf("retrieved response identity, status or lineage differs from creation")
	}
	if !reflect.DeepEqual(created.Output, retrieved.Output) {
		return fmt.Errorf("retrieved response output differs from the public creation result")
	}
	return nil
}
