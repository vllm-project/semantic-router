package configschema

import (
	"encoding/json"
	"testing"
)

func TestRenderIndexIsCompactAndDiscoverable(t *testing.T) {
	representation, err := Render(ViewOptions{})
	if err != nil {
		t.Fatalf("render index: %v", err)
	}
	if representation.ContentType != "application/json" {
		t.Fatalf("content type=%q", representation.ContentType)
	}
	if len(representation.Body) >= len(Document())/4 {
		t.Fatalf("index is not compact: index=%d full=%d", len(representation.Body), len(Document()))
	}
	var index schemaIndex
	if err := json.Unmarshal(representation.Body, &index); err != nil {
		t.Fatalf("decode index: %v", err)
	}
	if index.ConfigVersion != ConfigVersion || len(index.Sections) == 0 {
		t.Fatalf("unexpected index: %#v", index)
	}
	if index.Surfaces["signal"].Count == 0 || index.Surfaces["algorithm"].Count == 0 {
		t.Fatalf("routing surface directory is incomplete: %#v", index.Surfaces)
	}
}

func TestRenderFocusedViewsIncludeOnlyReferencedDefinitions(t *testing.T) {
	section, err := Render(ViewOptions{View: ViewSection, Path: "global.router.learning"})
	if err != nil {
		t.Fatalf("render section: %v", err)
	}
	if len(section.Body) >= len(Document()) {
		t.Fatalf("section=%d full=%d", len(section.Body), len(Document()))
	}
	var sectionDocument map[string]any
	if decodeErr := json.Unmarshal(section.Body, &sectionDocument); decodeErr != nil {
		t.Fatalf("decode section: %v", decodeErr)
	}
	metadata, _ := sectionDocument["x-vllm-sr-view"].(map[string]any)
	if metadata["path"] != "global.router.learning" {
		t.Fatalf("section metadata=%v", metadata)
	}

	surface, err := Render(ViewOptions{View: ViewSurface, SurfaceKind: "signal", SurfaceName: "keyword"})
	if err != nil {
		t.Fatalf("render surface: %v", err)
	}
	var surfaceDocument map[string]any
	if decodeErr := json.Unmarshal(surface.Body, &surfaceDocument); decodeErr != nil {
		t.Fatalf("decode surface: %v", decodeErr)
	}
	metadata, _ = surfaceDocument["x-vllm-sr-surface"].(map[string]any)
	if metadata["type"] != "keyword" {
		t.Fatalf("surface metadata=%v", metadata)
	}
}

func TestRenderRejectsUnknownViews(t *testing.T) {
	for _, options := range []ViewOptions{
		{View: "everything"},
		{View: ViewSection, Path: "global.unknown"},
		{View: ViewSurface, SurfaceKind: "signal", SurfaceName: "unknown"},
	} {
		if _, err := Render(options); err == nil {
			t.Fatalf("Render(%#v) succeeded, want error", options)
		}
	}
}
