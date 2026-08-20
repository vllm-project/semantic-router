package workflowstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"strings"
	"time"
)

// RecipeDraft is Dashboard-owned authoring state. It is intentionally separate
// from the live router configuration until an Entrypoint publishes it.
type RecipeDraft struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Routing     map[string]any `json:"routing"`
	CreatedAt   string         `json:"createdAt"`
	UpdatedAt   string         `json:"updatedAt"`
}

func normalizeRecipeDraft(item RecipeDraft) (RecipeDraft, error) {
	item.Name = strings.TrimSpace(item.Name)
	item.Description = strings.TrimSpace(item.Description)
	if item.Name == "" {
		return RecipeDraft{}, errors.New("recipe name is required")
	}
	if item.Name == "default" {
		return RecipeDraft{}, errors.New("default is reserved for the live routing profile")
	}
	if item.Routing == nil {
		item.Routing = map[string]any{}
	}
	return item, nil
}

func (s *Store) ListRecipeDrafts(ctx context.Context) ([]RecipeDraft, error) {
	rows, err := s.db.QueryContext(ctx, `
SELECT name,description,routing_json,created_at,updated_at
FROM recipe_draft ORDER BY updated_at DESC, name ASC`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	items := []RecipeDraft{}
	for rows.Next() {
		var item RecipeDraft
		var routingJSON string
		if err := rows.Scan(
			&item.Name,
			&item.Description,
			&routingJSON,
			&item.CreatedAt,
			&item.UpdatedAt,
		); err != nil {
			return nil, err
		}
		if err := json.Unmarshal([]byte(routingJSON), &item.Routing); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Store) GetRecipeDraft(ctx context.Context, name string) (RecipeDraft, error) {
	var item RecipeDraft
	var routingJSON string
	err := s.db.QueryRowContext(ctx, `
SELECT name,description,routing_json,created_at,updated_at
FROM recipe_draft WHERE name=?`, strings.TrimSpace(name)).
		Scan(&item.Name, &item.Description, &routingJSON, &item.CreatedAt, &item.UpdatedAt)
	if err != nil {
		return RecipeDraft{}, err
	}
	if err := json.Unmarshal([]byte(routingJSON), &item.Routing); err != nil {
		return RecipeDraft{}, err
	}
	return item, nil
}

func (s *Store) SaveRecipeDraft(ctx context.Context, item RecipeDraft) (RecipeDraft, error) {
	item, err := normalizeRecipeDraft(item)
	if err != nil {
		return RecipeDraft{}, err
	}
	routingJSON, err := json.Marshal(item.Routing)
	if err != nil {
		return RecipeDraft{}, errors.New("recipe routing is not valid JSON")
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	_, err = s.db.ExecContext(ctx, `
INSERT INTO recipe_draft(name,description,routing_json,created_at,updated_at)
VALUES(?,?,?,?,?)
ON CONFLICT(name) DO UPDATE SET
 description=excluded.description,
 routing_json=excluded.routing_json,
 updated_at=excluded.updated_at`,
		item.Name,
		item.Description,
		string(routingJSON),
		now,
		now,
	)
	if err != nil {
		return RecipeDraft{}, err
	}
	return s.GetRecipeDraft(ctx, item.Name)
}

func (s *Store) DeleteRecipeDraft(ctx context.Context, name string) error {
	result, err := s.db.ExecContext(
		ctx,
		`DELETE FROM recipe_draft WHERE name=?`,
		strings.TrimSpace(name),
	)
	if err != nil {
		return err
	}
	if affected, _ := result.RowsAffected(); affected == 0 {
		return sql.ErrNoRows
	}
	return nil
}
