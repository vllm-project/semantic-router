package llmprotocol

import (
	"net/url"
	"unicode"
	"unicode/utf8"
)

// ValidateTextCitations applies the same bounded URL and Unicode-offset
// contract to a complete text block assembled from buffered or streaming wire
// input. Streaming codecs call this after each citation addition so partial
// transport frames cannot bypass semantic response validation.
func ValidateTextCitations(text string, citations []Citation, limits Limits) error {
	if limits.TextBytes > 0 && len(text) > limits.TextBytes {
		return NewError(ErrorInvalidRequest, "text_limit", "content text exceeds the configured limit", nil)
	}
	return ValidateCitationBatch(int64(utf8.RuneCountInString(text)), len(citations), citations, limits)
}

// ValidateCitationBatch validates newly observed citations against the
// cumulative Unicode text length and citation count for one streaming item.
// Callers update their cumulative state only after this function succeeds.
func ValidateCitationBatch(
	textLength int64,
	totalCitations int,
	citations []Citation,
	limits Limits,
) error {
	if textLength < 0 || totalCitations < len(citations) {
		return NewError(ErrorInvalidRequest, "citation_state", "citation stream state is invalid", nil)
	}
	if limits.Citations > 0 && totalCitations > limits.Citations {
		return NewError(ErrorInvalidRequest, "citation_limit", "text citations exceed the configured limit", nil)
	}
	for _, citation := range citations {
		if exceeds(citation.URL, limits.CitationURLBytes) || exceeds(citation.Title, limits.CitationTitleBytes) {
			return NewError(ErrorInvalidRequest, "citation_field_limit", "citation URL or title exceeds the configured limit", nil)
		}
		if citation.StartIndex < 0 || citation.EndIndex < citation.StartIndex || citation.EndIndex > textLength {
			return NewError(ErrorInvalidRequest, "citation_range", "citation range is outside its text block", nil)
		}
		if err := validateCitationURL(citation.URL); err != nil {
			return err
		}
	}
	return nil
}

func validateCitationURL(raw string) error {
	for _, character := range raw {
		if unicode.IsControl(character) {
			return NewError(ErrorInvalidRequest, "invalid_citation_url", "citation URL contains control characters", nil)
		}
	}
	parsed, err := url.Parse(raw)
	if err != nil || !parsed.IsAbs() || parsed.Host == "" || parsed.User != nil ||
		(parsed.Scheme != "https" && parsed.Scheme != "http") {
		return NewError(ErrorInvalidRequest, "invalid_citation_url", "citation URL must be an absolute HTTP(S) URL without credentials", err)
	}
	return nil
}

func validateMediaURL(raw string) error {
	for _, character := range raw {
		if unicode.IsControl(character) {
			return NewError(ErrorInvalidRequest, "invalid_media_url", "media URL contains control characters", nil)
		}
	}
	parsed, err := url.Parse(raw)
	if err != nil || !parsed.IsAbs() || parsed.Host == "" || parsed.User != nil ||
		(parsed.Scheme != "https" && parsed.Scheme != "http") {
		return NewError(ErrorInvalidRequest, "invalid_media_url", "media URL must be an absolute HTTP(S) URL without credentials", err)
	}
	return nil
}
