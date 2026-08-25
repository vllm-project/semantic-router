package postgres

import (
	"context"
	"fmt"
	"sort"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

func (store *Store) ProviderCredentialIDsByName(
	ctx context.Context, namespaceID string, names []string,
) (map[string]string, error) {
	canonical, err := canonicalCredentialNames(names)
	if err != nil {
		return nil, err
	}
	if len(canonical) == 0 {
		return map[string]string{}, nil
	}
	rows, err := store.db.QueryContext(ctx, `SELECT name,id::text
FROM provider_credentials
WHERE namespace_id=$1 AND name=ANY($2::text[])`, namespaceID, pq.Array(canonical))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	resolved := make(map[string]string, len(canonical))
	for rows.Next() {
		var name, id string
		if scanErr := rows.Scan(&name, &id); scanErr != nil {
			return nil, scanErr
		}
		resolved[name] = id
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(resolved) != len(canonical) {
		return nil, fmt.Errorf("%w: ProviderCredential name is unavailable in this Namespace", routingmanagement.ErrManifest)
	}
	return resolved, nil
}

func (store *Store) ProviderCredentialNamesByID(
	ctx context.Context, namespaceID string, ids []string,
) (map[string]string, error) {
	canonical, err := canonicalCredentialIDs(ids)
	if err != nil {
		return nil, err
	}
	if len(canonical) == 0 {
		return map[string]string{}, nil
	}
	rows, err := store.db.QueryContext(ctx, `SELECT id::text,name
FROM provider_credentials
WHERE namespace_id=$1 AND id=ANY($2::uuid[])`, namespaceID, pq.Array(canonical))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	resolved := make(map[string]string, len(canonical))
	for rows.Next() {
		var id, name string
		if scanErr := rows.Scan(&id, &name); scanErr != nil {
			return nil, scanErr
		}
		resolved[id] = name
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(resolved) != len(canonical) {
		return nil, fmt.Errorf("%w: ProviderCredential identity is unavailable in this Namespace", routingmanagement.ErrPublication)
	}
	return resolved, nil
}

func canonicalCredentialNames(values []string) ([]string, error) {
	canonical := append([]string(nil), values...)
	for _, value := range canonical {
		if err := providercredential.ValidateName(value); err != nil {
			return nil, fmt.Errorf("%w: invalid ProviderCredential name", routingmanagement.ErrManifest)
		}
	}
	sort.Strings(canonical)
	return compactStrings(canonical), nil
}

func canonicalCredentialIDs(values []string) ([]string, error) {
	canonical := append([]string(nil), values...)
	for _, value := range canonical {
		if _, err := uuid.Parse(value); err != nil {
			return nil, fmt.Errorf("%w: invalid ProviderCredential identity", routingmanagement.ErrPublication)
		}
	}
	sort.Strings(canonical)
	return compactStrings(canonical), nil
}

func compactStrings(values []string) []string {
	if len(values) == 0 {
		return values
	}
	write := 1
	for read := 1; read < len(values); read++ {
		if values[read] == values[write-1] {
			continue
		}
		values[write] = values[read]
		write++
	}
	return values[:write]
}
