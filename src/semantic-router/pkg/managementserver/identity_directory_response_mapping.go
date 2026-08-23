package managementserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func principalDirectoryEntryDTO(
	value managementidentity.PrincipalDirectoryEntry,
	includeVerifiedEmail bool,
) managementapi.PrincipalDirectoryEntry {
	result := managementapi.PrincipalDirectoryEntry{
		PrincipalID: string(value.PrincipalID), DisplayName: value.DisplayName,
		Status: string(value.Status), Linked: value.Linked(), UserID: string(value.UserID),
	}
	if includeVerifiedEmail {
		result.VerifiedEmail = value.VerifiedEmail
	}
	if value.LinkRevision > 0 {
		revision := uint64(value.LinkRevision)
		result.LinkRevision = &revision
	}
	return result
}

func principalUserLinkDTO(value managementidentity.PrincipalUserLink) managementapi.PrincipalUserLink {
	return managementapi.PrincipalUserLink{
		PrincipalID: string(value.PrincipalID), NamespaceID: string(value.NamespaceID),
		UserID: string(value.UserID), Revision: uint64(value.Revision),
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}
