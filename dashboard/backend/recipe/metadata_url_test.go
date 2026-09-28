package recipe

import "testing"

func TestMetadataURLsExcludeCredentialsQueryAndFragment(t *testing.T) {
	tests := []struct {
		name  string
		value string
		valid bool
	}{
		{name: "clean", value: "https://example.com/org/recipe", valid: true},
		{name: "userinfo", value: "https://user:secret@example.com/recipe"},
		{name: "query", value: "https://example.com/recipe?token=secret"},
		{name: "empty_query", value: "https://example.com/recipe?"},
		{name: "fragment", value: "https://example.com/recipe#private"},
		{name: "http", value: "http://example.com/recipe"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isHTTPSURI(test.value); got != test.valid {
				t.Fatalf("isHTTPSURI(%q) = %v, want %v", test.value, got, test.valid)
			}
		})
	}
}

func TestMetadataURLFieldsUsePrivateComponentPolicy(t *testing.T) {
	clean := "https://example.com/author"
	metadata := Metadata{
		SchemaVersion: MetadataSchemaVersion,
		ID:            "url-test",
		Name:          "URL Test",
		Version:       "0.1.0",
		Description:   "Metadata URL validation.",
		Authors:       []Author{{Name: "Author", URL: &clean}},
		License:       "Apache-2.0",
		Tags:          []string{"test"},
		Links:         MetadataLinks{Source: clean},
	}
	if err := validateMetadata(metadata); err != nil {
		t.Fatalf("clean metadata rejected: %v", err)
	}

	for _, privateURL := range []string{
		"https://user:secret@example.com/source",
		"https://example.com/source?token=secret",
		"https://example.com/source#private",
	} {
		t.Run(privateURL, func(t *testing.T) {
			candidate := metadata
			candidate.Links.Source = privateURL
			candidate.Links.Homepage = &privateURL
			candidate.Links.Documentation = &privateURL
			candidate.Authors = []Author{{Name: "Author", URL: &privateURL}}
			if err := validateMetadata(candidate); err == nil {
				t.Fatalf("metadata accepted private URL %q", privateURL)
			}
		})
	}
}
