package managementserver

import (
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
)

func newPolicyBulkOperation(value policybulk.Operation) managementapi.Operation {
	itemErrors := make([]managementapi.OperationItemFailure, len(value.ItemErrors))
	for index, item := range value.ItemErrors {
		itemErrors[index] = managementapi.OperationItemFailure{ItemID: item.ItemID, Code: item.Code, Reason: item.Reason}
	}
	return managementapi.Operation{
		OperationID: value.ID, Kind: value.Kind, State: managementapi.OperationState(value.State),
		Progress: managementapi.OperationProgress{
			Total:     managementapi.WholeQuantity(strconv.FormatUint(value.Total, 10)),
			Completed: managementapi.WholeQuantity(strconv.FormatUint(value.Completed, 10)),
			Failed:    managementapi.WholeQuantity(strconv.FormatUint(value.Failed, 10)),
		},
		Revisions: managementapi.RevisionState{
			DesiredRevision:     int64(value.DesiredRevision),
			PublicationRevision: int64(value.PublicationRevision),
			AppliedRevision:     int64(value.AppliedRevision),
		},
		TargetIDs: append([]string(nil), value.TargetIDs...), ItemErrors: itemErrors,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt, CompletedAt: cloneResponseTime(value.CompletedAt),
	}
}

func newPolicyBulkOperationPage(value policybulk.Page) managementapi.OperationPage {
	items := make([]managementapi.Operation, len(value.Items))
	for index := range value.Items {
		items[index] = newPolicyBulkOperation(value.Items[index])
	}
	return managementapi.OperationPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
	}}
}
