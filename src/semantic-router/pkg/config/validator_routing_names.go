package config

import (
	"fmt"
	"reflect"
	"strings"
)

// validateRoutingLocalNames enforces uniqueness inside one recipe. The caller
// supplies a recipe-scoped RouterConfig, so identical names in other recipes
// are intentionally invisible here.
func validateRoutingLocalNames(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.RoutingScope == "" {
		return nil
	}
	if err := validateNamedRoutingCollections("routing.signals", canonicalSignalsFromSignals(cfg.Signals)); err != nil {
		return err
	}
	if err := validateNamedRoutingCollections("routing.projections", cfg.Projections); err != nil {
		return err
	}

	decisionNames := make([]string, 0, len(cfg.Decisions))
	for i := range cfg.Decisions {
		decisionNames = append(decisionNames, cfg.Decisions[i].Name)
	}
	return validateUniqueRoutingNames("routing.decisions", decisionNames)
}

// validateNamedRoutingCollections walks a schema struct whose fields are
// slices of named objects. This keeps the uniqueness contract exhaustive when
// a new signal family or projection collection is added, without duplicating a
// long per-vendor/per-signal switch.
func validateNamedRoutingCollections(prefix string, collections interface{}) error {
	value := reflect.ValueOf(collections)
	typeOfValue := value.Type()
	for i := 0; i < value.NumField(); i++ {
		field := value.Field(i)
		if field.Kind() != reflect.Slice {
			continue
		}
		collectionName := strings.Split(typeOfValue.Field(i).Tag.Get("yaml"), ",")[0]
		if collectionName == "" || collectionName == "-" {
			collectionName = typeOfValue.Field(i).Name
		}

		names := make([]string, 0, field.Len())
		for j := 0; j < field.Len(); j++ {
			item := field.Index(j)
			if item.Kind() == reflect.Pointer {
				if item.IsNil() {
					names = append(names, "")
					continue
				}
				item = item.Elem()
			}
			if item.Kind() != reflect.Struct {
				return fmt.Errorf("%s.%s: schema item must be a named struct", prefix, collectionName)
			}
			name := item.FieldByName("Name")
			if !name.IsValid() || name.Kind() != reflect.String {
				return fmt.Errorf("%s.%s: schema item does not expose a string Name field", prefix, collectionName)
			}
			names = append(names, name.String())
		}
		if err := validateUniqueRoutingNames(prefix+"."+collectionName, names); err != nil {
			return err
		}
	}
	return nil
}

func validateUniqueRoutingNames(path string, names []string) error {
	seen := make(map[string]struct{}, len(names))
	for index, rawName := range names {
		name := strings.TrimSpace(rawName)
		if name == "" {
			return fmt.Errorf("%s[%d].name cannot be empty", path, index)
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("%s: duplicate local name %q", path, name)
		}
		seen[name] = struct{}{}
	}
	return nil
}
