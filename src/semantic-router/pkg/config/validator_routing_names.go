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
		path := prefix + "." + routingCollectionName(typeOfValue.Field(i))
		names, err := namedRoutingCollectionNames(path, field)
		if err != nil {
			return err
		}
		if err := validateUniqueRoutingNames(path, names); err != nil {
			return err
		}
	}
	return nil
}

func routingCollectionName(field reflect.StructField) string {
	name := strings.Split(field.Tag.Get("yaml"), ",")[0]
	if name == "" || name == "-" {
		return field.Name
	}
	return name
}

func namedRoutingCollectionNames(path string, collection reflect.Value) ([]string, error) {
	names := make([]string, 0, collection.Len())
	for i := 0; i < collection.Len(); i++ {
		name, err := namedRoutingCollectionItem(path, collection.Index(i))
		if err != nil {
			return nil, err
		}
		names = append(names, name)
	}
	return names, nil
}

func namedRoutingCollectionItem(path string, item reflect.Value) (string, error) {
	if item.Kind() == reflect.Pointer {
		if item.IsNil() {
			return "", nil
		}
		item = item.Elem()
	}
	if item.Kind() != reflect.Struct {
		return "", fmt.Errorf("%s: schema item must be a named struct", path)
	}
	name := item.FieldByName("Name")
	if !name.IsValid() || name.Kind() != reflect.String {
		return "", fmt.Errorf("%s: schema item does not expose a string Name field", path)
	}
	return name.String(), nil
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
