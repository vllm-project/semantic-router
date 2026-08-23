package accessmanagement

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const maximumRoutingClaims = 16

var routingClaimNamePattern = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_.-]{0,63}$`)

func ValidateSchema(schema RoutingClaimSchema) error {
	if schema.Revision == 0 && len(schema.Definitions) != 0 {
		return fmt.Errorf("%w: routing claim schema revision is missing", ErrInvalidRequest)
	}
	if len(schema.Definitions) > maximumRoutingClaims {
		return fmt.Errorf("%w: routing claim schema exceeds %d definitions", ErrInvalidRequest, maximumRoutingClaims)
	}
	for name, definition := range schema.Definitions {
		if !routingClaimNamePattern.MatchString(name) {
			return fmt.Errorf("%w: routing claim name %q is invalid", ErrInvalidRequest, name)
		}
		switch definition.Kind {
		case "string":
			if definition.Minimum != nil || definition.Maximum != nil {
				return fmt.Errorf("%w: string claim %q cannot define integer bounds", ErrInvalidRequest, name)
			}
			if definition.MaxLength != nil && (*definition.MaxLength < 1 || *definition.MaxLength > 4096) {
				return fmt.Errorf("%w: string claim %q maxLength is invalid", ErrInvalidRequest, name)
			}
		case "boolean":
			if definition.Minimum != nil || definition.Maximum != nil || definition.MaxLength != nil {
				return fmt.Errorf("%w: boolean claim %q cannot define bounds", ErrInvalidRequest, name)
			}
		case "integer":
			if definition.MaxLength != nil || definition.Minimum == nil || definition.Maximum == nil ||
				*definition.Minimum > *definition.Maximum {
				return fmt.Errorf("%w: integer claim %q requires valid minimum and maximum", ErrInvalidRequest, name)
			}
		default:
			return fmt.Errorf("%w: routing claim %q has unsupported kind", ErrInvalidRequest, name)
		}
	}
	return nil
}

func ValidateContextValues(schema RoutingClaimSchema, values map[string]routingsnapshot.ClaimValue) error {
	if err := ValidateSchema(schema); err != nil {
		return err
	}
	if len(values) > maximumRoutingClaims {
		return fmt.Errorf("%w: routing context exceeds %d values", ErrInvalidRequest, maximumRoutingClaims)
	}
	for name, value := range values {
		definition, found := schema.Definitions[name]
		if !found {
			return fmt.Errorf("%w: routing claim %q is not defined", ErrInvalidRequest, name)
		}
		if err := value.Validate(); err != nil || value.Kind != definition.Kind {
			return fmt.Errorf("%w: routing claim %q has the wrong value kind", ErrInvalidRequest, name)
		}
		switch value.Kind {
		case "string":
			maximum := int64(256)
			if definition.MaxLength != nil {
				maximum = *definition.MaxLength
			}
			if int64(len(value.String)) > maximum || strings.ContainsRune(value.String, '\x00') {
				return fmt.Errorf("%w: routing claim %q exceeds its string contract", ErrInvalidRequest, name)
			}
		case "integer":
			if value.Integer < *definition.Minimum || value.Integer > *definition.Maximum {
				return fmt.Errorf("%w: routing claim %q is outside its integer bounds", ErrInvalidRequest, name)
			}
		}
	}
	return nil
}
