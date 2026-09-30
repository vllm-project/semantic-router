package config

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Supported external-API request formats. Response-body limits are NOT
// declared here: main now owns that via ExternalAPIRAGConfig.MaxResponseBytes
// and httputil.ReadLimitedBody, so this PR no longer defines its own.
const (
	ExternalAPIRequestFormatPinecone      = "pinecone"
	ExternalAPIRequestFormatWeaviate      = "weaviate"
	ExternalAPIRequestFormatElasticsearch = "elasticsearch"
	ExternalAPIRequestFormatCustom        = "custom"
)

// ExternalAPICustomRequestTemplate is the validated representation of a
// custom external API RAG request body. Templates must be JSON objects or
// arrays so user-controlled substitutions cannot alter the configured shape.
type ExternalAPICustomRequestTemplate struct {
	document interface{}
}

// ParseExternalAPICustomRequestTemplate validates and compiles a custom
// external API RAG request template. The same parser is used during config
// validation and request execution so the accepted template language cannot
// drift from runtime behavior.
func ParseExternalAPICustomRequestTemplate(template string) (*ExternalAPICustomRequestTemplate, error) {
	if strings.TrimSpace(template) == "" {
		return nil, fmt.Errorf("request template is required for custom format")
	}

	normalized := quoteBareExternalAPICustomRequestPlaceholders(template)
	document, err := decodeExternalAPICustomRequestTemplate(normalized)
	if err != nil {
		return nil, fmt.Errorf("invalid custom request template JSON: %w", err)
	}

	switch document.(type) {
	case map[string]interface{}, []interface{}:
	default:
		return nil, fmt.Errorf("custom request template must be a JSON object or array")
	}
	if err := validateExternalAPICustomRequestTemplateValue(document); err != nil {
		return nil, err
	}

	return &ExternalAPICustomRequestTemplate{document: document}, nil
}

// Render substitutes typed request values into a validated custom template.
func (t *ExternalAPICustomRequestTemplate) Render(query string, topK int, threshold float64) ([]byte, error) {
	if t == nil || t.document == nil {
		return nil, fmt.Errorf("custom request template is not compiled")
	}

	document := substituteExternalAPICustomRequestPlaceholders(t.document, query, topK, threshold)
	requestBody, err := json.Marshal(document)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal custom request template: %w", err)
	}
	return requestBody, nil
}

func decodeExternalAPICustomRequestTemplate(template string) (interface{}, error) {
	decoder := json.NewDecoder(strings.NewReader(template))
	decoder.UseNumber()

	var document interface{}
	if err := decoder.Decode(&document); err != nil {
		return nil, err
	}

	var trailing interface{}
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("template contains multiple JSON values")
		}
		return nil, fmt.Errorf("invalid trailing data: %w", err)
	}
	return document, nil
}

const (
	externalAPIUserContentPlaceholder = "${user_content}"
	externalAPITopKPlaceholder        = "${top_k}"
	externalAPIThresholdPlaceholder   = "${threshold}"
)

var externalAPICustomRequestPlaceholders = []string{
	"{{.Query}}",
	"{{.TopK}}",
	"{{.Threshold}}",
	externalAPIUserContentPlaceholder,
	externalAPITopKPlaceholder,
	externalAPIThresholdPlaceholder,
}

func quoteBareExternalAPICustomRequestPlaceholders(template string) string {
	var result strings.Builder
	result.Grow(len(template))

	inString := false
	escaped := false
	for i := 0; i < len(template); {
		if inString {
			ch := template[i]
			result.WriteByte(ch)
			i++
			if escaped {
				escaped = false
				continue
			}
			switch ch {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}

		if template[i] == '"' {
			inString = true
			result.WriteByte(template[i])
			i++
			continue
		}

		if placeholder := externalAPICustomRequestPlaceholderAt(template, i); placeholder != "" {
			encoded, _ := json.Marshal(placeholder)
			result.Write(encoded)
			i += len(placeholder)
			continue
		}
		if length := externalAPIEnvironmentReferenceAt(template, i); length > 0 {
			encoded, _ := json.Marshal(template[i : i+length])
			result.Write(encoded)
			i += length
			continue
		}

		result.WriteByte(template[i])
		i++
	}

	return result.String()
}

func externalAPICustomRequestPlaceholderAt(value string, offset int) string {
	for _, placeholder := range externalAPICustomRequestPlaceholders {
		if strings.HasPrefix(value[offset:], placeholder) {
			return placeholder
		}
	}
	return ""
}

func validateExternalAPICustomRequestTemplateValue(value interface{}) error {
	switch typed := value.(type) {
	case map[string]interface{}:
		for key, child := range typed {
			if strings.Contains(key, "${") || strings.Contains(key, "{{") {
				return fmt.Errorf("custom request template placeholders are not allowed in object keys: %q", key)
			}
			if err := validateExternalAPICustomRequestTemplateValue(child); err != nil {
				return err
			}
		}
	case []interface{}:
		for _, child := range typed {
			if err := validateExternalAPICustomRequestTemplateValue(child); err != nil {
				return err
			}
		}
	case string:
		return validateExternalAPICustomRequestPlaceholderTokens(typed)
	}
	return nil
}

func validateExternalAPICustomRequestPlaceholderTokens(value string) error {
	for i := 0; i < len(value); {
		switch {
		case strings.HasPrefix(value[i:], "$$"):
			i += 2
		case strings.HasPrefix(value[i:], "${"):
			end := strings.IndexByte(value[i+2:], '}')
			if end < 0 {
				return fmt.Errorf("malformed custom request template placeholder at byte %d", i)
			}
			placeholder := value[i : i+2+end+1]
			if externalAPICustomRequestPlaceholderAt(placeholder, 0) != placeholder &&
				externalAPIEnvironmentReferenceAt(placeholder, 0) != len(placeholder) {
				return fmt.Errorf("unsupported custom request template placeholder %q", placeholder)
			}
			i += len(placeholder)
		case strings.HasPrefix(value[i:], "{{"):
			end := strings.Index(value[i+2:], "}}")
			if end < 0 {
				return fmt.Errorf("malformed custom request template placeholder at byte %d", i)
			}
			placeholder := value[i : i+2+end+2]
			if externalAPICustomRequestPlaceholderAt(placeholder, 0) != placeholder {
				return fmt.Errorf("unsupported custom request template placeholder %q", placeholder)
			}
			i += len(placeholder)
		default:
			i++
		}
	}
	return nil
}

func substituteExternalAPICustomRequestPlaceholders(value interface{}, query string, topK int, threshold float64) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(typed))
		for key, child := range typed {
			result[key] = substituteExternalAPICustomRequestPlaceholders(child, query, topK, threshold)
		}
		return result
	case []interface{}:
		result := make([]interface{}, len(typed))
		for i, child := range typed {
			result[i] = substituteExternalAPICustomRequestPlaceholders(child, query, topK, threshold)
		}
		return result
	case string:
		if replacement, ok := exactExternalAPICustomRequestPlaceholder(typed, query, topK, threshold); ok {
			return replacement
		}
		if externalAPIEnvironmentReferenceAt(typed, 0) == len(typed) && typed != "" {
			return resolveExternalAPIEnvironmentReference(typed)
		}
		return interpolateExternalAPICustomRequestPlaceholders(typed, query, topK, threshold)
	default:
		return value
	}
}

func exactExternalAPICustomRequestPlaceholder(value, query string, topK int, threshold float64) (interface{}, bool) {
	switch value {
	case "{{.Query}}", externalAPIUserContentPlaceholder:
		return query, true
	case "{{.TopK}}", externalAPITopKPlaceholder:
		return topK, true
	case "{{.Threshold}}", externalAPIThresholdPlaceholder:
		return json.Number(fmt.Sprintf("%.3f", threshold)), true
	default:
		return nil, false
	}
}

func interpolateExternalAPICustomRequestPlaceholders(value, query string, topK int, threshold float64) string {
	var result strings.Builder
	result.Grow(len(value))

	for i := 0; i < len(value); {
		placeholder := externalAPICustomRequestPlaceholderAt(value, i)
		if placeholder == "" {
			// One pass over the template text: request values written below
			// are never rescanned, so user content cannot name an environment
			// variable and have it resolved.
			if strings.HasPrefix(value[i:], "$$") {
				result.WriteByte('$')
				i += 2
				continue
			}
			if length := externalAPIEnvironmentReferenceAt(value, i); length > 0 {
				result.WriteString(resolveExternalAPIEnvironmentReference(value[i : i+length]))
				i += length
				continue
			}
			result.WriteByte(value[i])
			i++
			continue
		}

		switch placeholder {
		case "{{.Query}}", externalAPIUserContentPlaceholder:
			result.WriteString(query)
		case "{{.TopK}}", externalAPITopKPlaceholder:
			result.WriteString(strconv.Itoa(topK))
		case "{{.Threshold}}", externalAPIThresholdPlaceholder:
			result.WriteString(strconv.FormatFloat(threshold, 'f', 3, 64))
		}
		i += len(placeholder)
	}

	return result.String()
}

// externalAPIEnvironmentReferenceAt returns the length of an environment
// reference starting at offset, or 0. Accepted forms are ${NAME},
// ${NAME:-default}, ${NAME-default} and $NAME, where NAME uses uppercase
// environment syntax; lowercase braced tokens stay reserved for the runtime
// placeholders above. References are resolved when a request is rendered and
// always land inside a JSON string value, so an environment value can never
// add keys or change the request's shape.
func externalAPIEnvironmentReferenceAt(value string, offset int) int {
	if offset >= len(value) || value[offset] != '$' || offset+1 >= len(value) {
		return 0
	}
	rest := value[offset+1:]
	if rest[0] == '{' {
		closeOffset := strings.IndexByte(rest, '}')
		if closeOffset < 0 || !isUppercaseRequestTemplateEnvReference(rest[1:closeOffset]) {
			return 0
		}
		return closeOffset + 2
	}
	if !isUppercaseEnvironmentNameStart(rest[0]) {
		return 0
	}
	end := 1
	for end < len(rest) && isUppercaseEnvironmentNameByte(rest[end]) {
		end++
	}
	if end < len(rest) && rest[end] >= 'a' && rest[end] <= 'z' {
		// $TENANTid is not a reference to TENANT; leave it literal.
		return 0
	}
	return end + 1
}

func resolveExternalAPIEnvironmentReference(reference string) string {
	if strings.HasPrefix(reference, "${") {
		return resolveBracedEnvReference(reference[2 : len(reference)-1])
	}
	return os.Getenv(reference[1:])
}
