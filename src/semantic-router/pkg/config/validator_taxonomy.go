package config

import (
	"fmt"
	"path/filepath"
)

func validateTaxonomyContracts(cfg *RouterConfig) error {
	classifiers, taxonomies, err := taxonomyClassifierDefinitions(cfg)
	if err != nil {
		return err
	}

	signalNames := make(map[string]struct{}, len(cfg.TaxonomyRules))
	for _, rule := range cfg.TaxonomyRules {
		if rule.Name == "" {
			return fmt.Errorf("routing.signals.taxonomy: name cannot be empty")
		}
		if _, exists := signalNames[rule.Name]; exists {
			return fmt.Errorf("routing.signals.taxonomy[%q]: duplicate signal name", rule.Name)
		}
		signalNames[rule.Name] = struct{}{}

		classifier, ok := classifiers[rule.Classifier]
		if !ok {
			return fmt.Errorf("routing.signals.taxonomy[%q]: classifier %q is not declared in global.model_catalog.classifiers", rule.Name, rule.Classifier)
		}
		taxonomy, hasTaxonomy := taxonomies[classifier.Name]
		if !hasTaxonomy {
			continue
		}

		switch rule.Bind.Kind {
		case TaxonomyBindKindTier:
			if _, ok := taxonomy.Tiers[rule.Bind.Value]; ok {
				continue
			}
			if taxonomyHasTier(taxonomy, rule.Bind.Value) {
				continue
			}
			return fmt.Errorf("routing.signals.taxonomy[%q]: bind.value %q is not a declared taxonomy tier for classifier %q", rule.Name, rule.Bind.Value, rule.Classifier)
		case TaxonomyBindKindCategory:
			if _, ok := taxonomy.Categories[rule.Bind.Value]; !ok {
				return fmt.Errorf("routing.signals.taxonomy[%q]: bind.value %q is not a declared taxonomy category for classifier %q", rule.Name, rule.Bind.Value, rule.Classifier)
			}
		default:
			return fmt.Errorf("routing.signals.taxonomy[%q]: bind.kind %q is unsupported (supported: tier, category)", rule.Name, rule.Bind.Kind)
		}
	}

	return nil
}

func taxonomyClassifierDefinitions(cfg *RouterConfig) (map[string]TaxonomyClassifierConfig, map[string]TaxonomyDefinition, error) {
	classifiers := make(map[string]TaxonomyClassifierConfig, len(cfg.TaxonomyClassifiers))
	taxonomies := make(map[string]TaxonomyDefinition, len(cfg.TaxonomyClassifiers))
	for _, classifier := range cfg.TaxonomyClassifiers {
		if classifier.Name == "" {
			return nil, nil, fmt.Errorf("global.model_catalog.classifiers: name cannot be empty")
		}
		if _, exists := classifiers[classifier.Name]; exists {
			return nil, nil, fmt.Errorf("global.model_catalog.classifiers[%q]: duplicate classifier name", classifier.Name)
		}
		if classifier.NormalizedType() != ClassifierTypeTaxonomy {
			return nil, nil, fmt.Errorf("global.model_catalog.classifiers[%q]: unsupported type %q (supported: taxonomy)", classifier.Name, classifier.Type)
		}
		if classifier.Source.Path == "" {
			return nil, nil, fmt.Errorf("global.model_catalog.classifiers[%q]: source.path cannot be empty", classifier.Name)
		}
		if cfg.ConfigBaseDir == "" && !filepath.IsAbs(classifier.Source.Path) {
			classifiers[classifier.Name] = classifier
			continue
		}
		taxonomy, err := LoadTaxonomyDefinition(cfg.ConfigBaseDir, classifier.Source)
		if err != nil {
			return nil, nil, fmt.Errorf("global.model_catalog.classifiers[%q]: failed to load taxonomy manifest: %w", classifier.Name, err)
		}
		classifiers[classifier.Name] = classifier
		taxonomies[classifier.Name] = taxonomy
	}
	return classifiers, taxonomies, nil
}

func taxonomyHasTier(taxonomy TaxonomyDefinition, tier string) bool {
	for _, category := range taxonomy.Categories {
		if category.Tier == tier {
			return true
		}
	}
	for _, resolvedTier := range taxonomy.CategoryToTier {
		if resolvedTier == tier {
			return true
		}
	}
	return false
}
