package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const principalDirectoryColumns = `principal.id::text, principal.display_name,
       COALESCE(principal.verified_email,''), principal.status,
       link.user_id::text, link.revision`

func (store *Store) GetPrincipalDirectoryEntry(
	ctx context.Context,
	namespaceID string,
	principalID string,
) (managementidentity.PrincipalDirectoryEntry, error) {
	if !canonicalUUID(namespaceID) || !canonicalUUID(principalID) {
		return managementidentity.PrincipalDirectoryEntry{}, managementidentity.ErrNotFound
	}
	entry, err := scanPrincipalDirectoryEntry(store.database.QueryRowContext(ctx, `SELECT `+principalDirectoryColumns+`
FROM access_namespaces namespace
JOIN management_principals principal ON TRUE
LEFT JOIN management_principal_user_links link
  ON link.namespace_id=namespace.id AND link.principal_id=principal.id
WHERE namespace.id=$1 AND principal.id=$2`, namespaceID, principalID))
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.PrincipalDirectoryEntry{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.PrincipalDirectoryEntry{}, fmt.Errorf("load namespace principal directory entry: %w", err)
	}
	return entry, nil
}

func (store *Store) ListPrincipalDirectory(
	ctx context.Context,
	request managementidentity.PrincipalDirectoryRequest,
) (managementidentity.PrincipalDirectoryPage, error) {
	if !canonicalUUID(request.NamespaceID) || request.Limit < 1 || request.Limit > 200 ||
		(request.AfterID != "" && !canonicalUUID(request.AfterID)) {
		return managementidentity.PrincipalDirectoryPage{}, managementidentity.ErrInvalidLifecycleRequest
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+principalDirectoryColumns+`
FROM access_namespaces namespace
JOIN management_principals principal ON TRUE
LEFT JOIN management_principal_user_links link
  ON link.namespace_id=namespace.id AND link.principal_id=principal.id
WHERE namespace.id=$1
  AND ($2='' OR lower(principal.display_name) LIKE lower($2)||'%'
       OR lower(COALESCE(principal.verified_email,'')) LIKE lower($2)||'%')
  AND ($3='' OR principal.id>NULLIF($3,'')::uuid)
ORDER BY principal.id LIMIT $4`, request.NamespaceID, request.Search, request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.PrincipalDirectoryPage{}, fmt.Errorf("list namespace principal directory: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.PrincipalDirectoryEntry, 0, request.Limit+1)
	for rows.Next() {
		entry, err := scanPrincipalDirectoryEntry(rows)
		if err != nil {
			return managementidentity.PrincipalDirectoryPage{}, fmt.Errorf("scan namespace principal directory: %w", err)
		}
		items = append(items, entry)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.PrincipalDirectoryPage{}, fmt.Errorf("iterate namespace principal directory: %w", err)
	}
	page := managementidentity.PrincipalDirectoryPage{Items: items}
	if len(items) > request.Limit {
		page.Items = items[:request.Limit]
		page.NextCursor = string(page.Items[len(page.Items)-1].PrincipalID)
	}
	return page, nil
}

func (store *Store) ListPrincipalUserLinks(
	ctx context.Context,
	request managementidentity.PrincipalUserLinkListRequest,
) (managementidentity.PrincipalUserLinkPage, error) {
	if !canonicalUUID(request.NamespaceID) || request.Limit < 1 || request.Limit > 200 ||
		(request.PrincipalID != "" && !canonicalUUID(request.PrincipalID)) ||
		(request.UserID != "" && !canonicalUUID(request.UserID)) ||
		(request.AfterID != "" && !canonicalUUID(request.AfterID)) {
		return managementidentity.PrincipalUserLinkPage{}, managementidentity.ErrInvalidLifecycleRequest
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+linkColumns+`
FROM management_principal_user_links
WHERE namespace_id=$1
  AND ($2='' OR principal_id=NULLIF($2,'')::uuid)
  AND ($3='' OR user_id=NULLIF($3,'')::uuid)
  AND ($4='' OR principal_id>NULLIF($4,'')::uuid)
ORDER BY principal_id LIMIT $5`, request.NamespaceID, request.PrincipalID, request.UserID,
		request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.PrincipalUserLinkPage{}, fmt.Errorf("list namespace principal User links: %w", err)
	}
	return scanPrincipalUserLinkPage(rows, request.Limit, true)
}

func (store *Store) ListPrincipalLinks(
	ctx context.Context,
	principalID string,
	request managementidentity.ListRequest,
) (managementidentity.PrincipalUserLinkPage, error) {
	if !canonicalUUID(principalID) || request.Limit < 1 || request.Limit > 200 ||
		(request.AfterID != "" && !canonicalUUID(request.AfterID)) {
		return managementidentity.PrincipalUserLinkPage{}, managementidentity.ErrInvalidLifecycleRequest
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+linkColumns+`
FROM management_principal_user_links
WHERE principal_id=$1 AND ($2='' OR namespace_id>NULLIF($2,'')::uuid)
ORDER BY namespace_id LIMIT $3`, principalID, request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.PrincipalUserLinkPage{}, fmt.Errorf("list principal User links: %w", err)
	}
	return scanPrincipalUserLinkPage(rows, request.Limit, false)
}

func scanPrincipalDirectoryEntry(scanner scanner) (managementidentity.PrincipalDirectoryEntry, error) {
	var entry managementidentity.PrincipalDirectoryEntry
	var userID sql.NullString
	var revision sql.NullInt64
	if err := scanner.Scan(&entry.PrincipalID, &entry.DisplayName, &entry.VerifiedEmail,
		&entry.Status, &userID, &revision); err != nil {
		return managementidentity.PrincipalDirectoryEntry{}, err
	}
	if !canonicalUUID(string(entry.PrincipalID)) || entry.DisplayName == "" || !entry.Status.Valid() ||
		userID.Valid != revision.Valid || (revision.Valid && revision.Int64 < 1) {
		return managementidentity.PrincipalDirectoryEntry{}, errors.New("stored namespace principal directory entry is invalid")
	}
	if userID.Valid {
		if !canonicalUUID(userID.String) {
			return managementidentity.PrincipalDirectoryEntry{}, errors.New("stored principal User link is invalid")
		}
		entry.UserID = accesscontrol.UserID(userID.String)
		entry.LinkRevision = accesscontrol.Revision(revision.Int64)
	}
	return entry, nil
}

func scanPrincipalUserLinkPage(
	rows *sql.Rows,
	limit int,
	principalCursor bool,
) (managementidentity.PrincipalUserLinkPage, error) {
	defer rows.Close()
	items := make([]managementidentity.PrincipalUserLink, 0, limit+1)
	for rows.Next() {
		link, err := scanLink(rows)
		if err != nil {
			return managementidentity.PrincipalUserLinkPage{}, fmt.Errorf("scan principal User-link page: %w", err)
		}
		items = append(items, link)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.PrincipalUserLinkPage{}, fmt.Errorf("iterate principal User-link page: %w", err)
	}
	page := managementidentity.PrincipalUserLinkPage{Items: items}
	if len(items) > limit {
		page.Items = items[:limit]
		last := page.Items[len(page.Items)-1]
		page.NextCursor = string(last.NamespaceID)
		if principalCursor {
			page.NextCursor = string(last.PrincipalID)
		}
	}
	return page, nil
}
