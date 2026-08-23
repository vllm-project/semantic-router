package managementserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func newAPIKey(value accesscontrol.APIKey) managementapi.APIKey {
	return managementapi.APIKey{
		KeyID: string(value.ID), Name: value.Name,
		Owner:         managementapi.APIKeyOwner{Type: string(value.Owner.Kind), ID: string(value.Owner.ID)},
		ContextTeamID: string(value.ContextTeamID), Status: string(value.Status),
		ExpiresAt: cloneResponseTime(value.ExpiresAt), LastUsedAt: cloneResponseTime(value.LastUsedAt),
		Revision: uint64(value.Revision), CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
		DeletedAt: cloneResponseTime(value.DeletedAt),
	}
}

func newAPIKeyPage(value apikeymanagement.KeyPage) managementapi.APIKeyPage {
	items := make([]managementapi.APIKey, len(value.Items))
	for index := range value.Items {
		items[index] = newAPIKey(value.Items[index])
	}
	return managementapi.APIKeyPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
	}}
}

func newAPIKeyCredential(value apikeymanagement.CredentialMetadata) managementapi.APIKeyCredential {
	return managementapi.APIKeyCredential{
		CredentialID: value.ID, KeyID: value.KeyID, KID: value.KID,
		Status: string(value.Status), Revealable: value.Revealable, NotBefore: value.NotBefore,
		ExpiresAt: cloneResponseTime(value.ExpiresAt), RevokedAt: cloneResponseTime(value.RevokedAt),
		CreatedAt: value.CreatedAt,
	}
}

func newAPIKeyCredentialPage(value apikeymanagement.CredentialPage) managementapi.APIKeyCredentialPage {
	items := make([]managementapi.APIKeyCredential, len(value.Items))
	for index := range value.Items {
		items[index] = newAPIKeyCredential(value.Items[index])
	}
	return managementapi.APIKeyCredentialPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
	}}
}
