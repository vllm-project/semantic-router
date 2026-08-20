package workflowstore

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"testing"
)

func TestRecipeDraftLifecycle(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "workflow.db"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	created, err := store.SaveRecipeDraft(context.Background(), RecipeDraft{
		Name:        "speed-first",
		Description: "Fast path",
		Routing: map[string]any{
			"signals":   map[string]any{"keywords": []any{}},
			"decisions": []any{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if created.Name != "speed-first" || created.CreatedAt == "" || created.UpdatedAt == "" {
		t.Fatalf("created draft = %#v", created)
	}

	created.Description = "Updated"
	updated, err := store.SaveRecipeDraft(context.Background(), created)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Description != "Updated" || updated.CreatedAt != created.CreatedAt {
		t.Fatalf("updated draft = %#v", updated)
	}

	items, err := store.ListRecipeDrafts(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Name != created.Name {
		t.Fatalf("drafts = %#v", items)
	}

	if err := store.DeleteRecipeDraft(context.Background(), created.Name); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetRecipeDraft(context.Background(), created.Name); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("GetRecipeDraft() error = %v, want sql.ErrNoRows", err)
	}
}
