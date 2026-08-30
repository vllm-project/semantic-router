package llmprotocol

import (
	"strings"
	"testing"
)

func TestCitationValidationIsBoundedAndTextScoped(t *testing.T) {
	limits := DefaultPolicy().Limits
	valid := validSemanticRequest()
	valid.Messages[0].Content[0].Citations = []Citation{{
		URL: "https://example.com/source", Title: "Source", StartIndex: 0, EndIndex: 5,
	}}
	if err := ValidateRequest(valid, limits); err != nil {
		t.Fatalf("valid citation rejected: %v", err)
	}

	for name, mutate := range map[string]func(*Content){
		"range":       func(content *Content) { content.Citations[0].EndIndex = 6 },
		"url":         func(content *Content) { content.Citations[0].URL = "file:///tmp/source" },
		"title bound": func(content *Content) { content.Citations[0].Title = strings.Repeat("x", limits.CitationTitleBytes+1) },
		"count bound": func(content *Content) { content.Citations = make([]Citation, limits.Citations+1) },
	} {
		t.Run(name, func(t *testing.T) {
			request := validSemanticRequest()
			request.Messages[0].Content[0].Citations = append([]Citation(nil), valid.Messages[0].Content[0].Citations...)
			mutate(&request.Messages[0].Content[0])
			if err := ValidateRequest(request, limits); err == nil {
				t.Fatal("invalid citation was accepted")
			}
		})
	}

	nonText := valid
	nonText.Messages = []Message{{Role: RoleUser, Content: []Content{{
		Kind: ContentImage, URL: "https://example.com/image.png",
		Citations: []Citation{{URL: "https://example.com/source"}},
	}}}}
	if err := ValidateRequest(nonText, limits); err == nil {
		t.Fatal("citation on non-text content was accepted")
	}
}
