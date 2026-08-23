package managementapi

import "time"

type PrincipalDirectoryEntry struct {
	PrincipalID   string  `json:"principalId"`
	DisplayName   string  `json:"displayName"`
	VerifiedEmail string  `json:"verifiedEmail,omitempty"`
	Status        string  `json:"status"`
	Linked        bool    `json:"linked"`
	UserID        string  `json:"userId,omitempty"`
	LinkRevision  *uint64 `json:"linkRevision,omitempty"`
}

type PrincipalDirectoryDetail struct {
	Data PrincipalDirectoryEntry `json:"data"`
}

type PrincipalDirectoryPage struct {
	Data []PrincipalDirectoryEntry `json:"data"`
	Page PageInfo                  `json:"page"`
}

type PrincipalUserLink struct {
	PrincipalID string    `json:"principalId"`
	NamespaceID string    `json:"namespaceId"`
	UserID      string    `json:"userId"`
	Revision    uint64    `json:"revision"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`
}

type PrincipalUserLinkDetail struct {
	Data PrincipalUserLink `json:"data"`
}

type PrincipalUserLinkPage struct {
	Data []PrincipalUserLink `json:"data"`
	Page PageInfo            `json:"page"`
}

type PrincipalUserLinkPutRequest struct {
	UserID string `json:"userId"`
}
