package managementserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	credentialmanagement "github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/management"
)

func newProviderCredential(value credentialmanagement.Metadata) managementapi.ProviderCredential {
	return managementapi.ProviderCredential{
		CredentialID: value.CredentialID, Name: value.Name, ProviderID: value.ProviderID,
		CatalogRevision: value.CatalogRevision, NormalizedOrigin: value.NormalizedOrigin,
		Status: string(value.Status), Revision: value.Revision,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt, DeletedAt: cloneResponseTime(value.DeletedAt),
	}
}

func newProviderCredentialPage(value credentialmanagement.ListResult) managementapi.ProviderCredentialPage {
	items := make([]managementapi.ProviderCredential, len(value.Credentials))
	for index := range value.Credentials {
		items[index] = newProviderCredential(value.Credentials[index])
	}
	return managementapi.ProviderCredentialPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
	}}
}
