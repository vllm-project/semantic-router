package config

import "fmt"

type optionalNonNegativeIntField struct {
	name  string
	value *int
}

func validateOptionalNonNegativeIntFields(fields []optionalNonNegativeIntField) error {
	for _, field := range fields {
		if field.value != nil && *field.value < 0 {
			return fmt.Errorf("%s must be >= 0, got %d", field.name, *field.value)
		}
	}
	return nil
}

type optionalNonNegativeFloatField struct {
	name  string
	value *float64
}

func validateOptionalNonNegativeFloatFields(fields []optionalNonNegativeFloatField) error {
	for _, field := range fields {
		if field.value != nil && *field.value < 0 {
			return fmt.Errorf("%s must be >= 0, got %v", field.name, *field.value)
		}
	}
	return nil
}
