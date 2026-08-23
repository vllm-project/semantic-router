package routingmanagement

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const maximumRecipeDocumentBytes = 2 << 20

var (
	resourceIDPattern = regexp.MustCompile(`^[a-z][a-z0-9_-]{2,127}$`)
	digestPattern     = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
)

func CompileRecipeDocument(recipeID string, document json.RawMessage) (json.RawMessage, []routingsnapshot.Decision, error) {
	if err := ValidateResourceID(recipeID); err != nil {
		return nil, nil, fmt.Errorf("%w: Recipe identity is invalid", ErrInvalid)
	}
	if len(document) > maximumRecipeDocumentBytes {
		return nil, nil, fmt.Errorf("%w: recipe document exceeds %d bytes", ErrInvalid, maximumRecipeDocumentBytes)
	}
	parsed, canonical, err := config.ParseManagedRecipeDocument(document)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	decisions := make([]routingsnapshot.Decision, len(parsed.Decisions))
	seenNames := make(map[string]struct{}, len(parsed.Decisions))
	for index, decision := range parsed.Decisions {
		if !canonicalText(decision.Name, 1, 128) {
			return nil, nil, fmt.Errorf("%w: Decision %d requires a canonical name", ErrInvalid, index)
		}
		if _, duplicate := seenNames[decision.Name]; duplicate {
			return nil, nil, fmt.Errorf("%w: duplicate Decision name %q", ErrInvalid, decision.Name)
		}
		seenNames[decision.Name] = struct{}{}
		algorithmType := ""
		if decision.Algorithm != nil {
			algorithmType = decision.Algorithm.Type
		}
		cardinality, known := config.DecisionAlgorithmDispatchCardinality(algorithmType)
		if !known {
			return nil, nil, fmt.Errorf("%w: Decision %q uses an unknown dispatch algorithm", ErrInvalid, decision.Name)
		}
		decisions[index] = routingsnapshot.Decision{
			ID: config.DeterministicRoutingResourceID("dec", recipeID, decision.Name), Name: decision.Name,
			DispatchCardinality: routingsnapshot.DispatchCardinality(cardinality),
		}
	}
	sort.Slice(decisions, func(i, j int) bool { return decisions[i].ID < decisions[j].ID })
	return canonical, decisions, nil
}

func ValidateResourceID(value string) error {
	if !resourceIDPattern.MatchString(value) {
		return fmt.Errorf("%w: routing resource id is not canonical", ErrInvalid)
	}
	return nil
}

func canonicalText(value string, minimum, maximum int) bool {
	return utf8.ValidString(value) && len(value) >= minimum && len(value) <= maximum &&
		strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n\t")
}

func validateIdentity(id, name string) error {
	if err := ValidateResourceID(id); err != nil {
		return err
	}
	if !canonicalText(name, 1, 256) {
		return fmt.Errorf("%w: routing resource name is invalid", ErrInvalid)
	}
	return nil
}

func validateCatalogRevision(value string) error {
	if !digestPattern.MatchString(value) {
		return fmt.Errorf("%w: Provider catalog revision is invalid", ErrInvalid)
	}
	return nil
}

func uniqueCanonical(values []string, maximum int) ([]string, error) {
	if len(values) > maximum {
		return nil, fmt.Errorf("%w: list exceeds %d entries", ErrInvalid, maximum)
	}
	result := append([]string(nil), values...)
	sort.Strings(result)
	for index, value := range result {
		if !canonicalText(value, 1, 256) || index > 0 && result[index-1] == value {
			return nil, fmt.Errorf("%w: list contains an invalid or duplicate value", ErrInvalid)
		}
	}
	return result, nil
}
