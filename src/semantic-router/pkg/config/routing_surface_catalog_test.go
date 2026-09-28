package config

import (
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestDecisionAlgorithmCatalog_AllTypesHaveTier(t *testing.T) {
	catalog := DecisionAlgorithmCatalog()
	if len(catalog) == 0 {
		t.Fatal("DecisionAlgorithmCatalog() returned empty catalog")
	}

	for _, entry := range catalog {
		if entry.Type == "" {
			t.Error("Catalog entry has empty Type")
		}
		if entry.Tier != "supported" && entry.Tier != "experimental" {
			t.Errorf("Catalog entry %q has invalid Tier %q", entry.Type, entry.Tier)
		}
		if entry.Execution != AlgorithmExecutionSelector && entry.Execution != AlgorithmExecutionLooper {
			t.Errorf("Catalog entry %q has invalid Execution %q", entry.Type, entry.Execution)
		}
		if entry.PayloadShape != "" && entry.PayloadShape != AlgorithmPayloadNested {
			t.Errorf("Catalog entry %q has invalid payload shape %q", entry.Type, entry.PayloadShape)
		}
		if entry.PayloadShape == AlgorithmPayloadNested && entry.ConfigField == "" {
			t.Errorf("Catalog entry %q has a nested payload without a config field", entry.Type)
		}
	}
}

func TestDecisionAlgorithmRegistryOwnsEveryPayloadBlock(t *testing.T) {
	algorithmType := reflect.TypeOf(AlgorithmConfig{})
	for _, entry := range decisionAlgorithmRegistry {
		configField := entry.Catalog.ConfigField
		if configField == "" {
			if entry.IsConfigured != nil {
				t.Errorf("blockless algorithm %q has a payload detector", entry.Catalog.Type)
			}
			continue
		}
		if entry.IsConfigured == nil {
			t.Errorf("algorithm %q has config field %q without a payload detector", entry.Catalog.Type, configField)
			continue
		}

		fieldIndex := -1
		for index := 0; index < algorithmType.NumField(); index++ {
			yamlName := strings.Split(algorithmType.Field(index).Tag.Get("yaml"), ",")[0]
			if yamlName == configField {
				fieldIndex = index
				break
			}
		}
		if fieldIndex < 0 {
			t.Errorf("algorithm %q config field %q is missing from AlgorithmConfig", entry.Catalog.Type, configField)
			continue
		}

		configured := reflect.New(algorithmType)
		field := configured.Elem().Field(fieldIndex)
		if field.Kind() != reflect.Pointer {
			t.Errorf("algorithm %q config field %q is not a pointer", entry.Catalog.Type, configField)
			continue
		}
		field.Set(reflect.New(field.Type().Elem()))
		blocks := configuredDecisionAlgorithmBlocks(configured.Interface().(*AlgorithmConfig))
		if !slices.Equal(blocks, []string{configField}) {
			t.Errorf("algorithm %q configured blocks = %v, want [%s]", entry.Catalog.Type, blocks, configField)
		}
	}
}

func TestSupportedLooperAlgorithmTypes(t *testing.T) {
	want := []string{
		DecisionAlgorithmConfidence,
		DecisionAlgorithmFusion,
		DecisionAlgorithmRatings,
		DecisionAlgorithmReMoM,
		DecisionAlgorithmWorkflows,
	}
	got := SupportedLooperAlgorithmTypes()

	if len(got) != len(want) {
		t.Fatalf("SupportedLooperAlgorithmTypes() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SupportedLooperAlgorithmTypes() = %v, want %v", got, want)
		}
		if !IsLooperAlgorithmType(got[i]) {
			t.Errorf("IsLooperAlgorithmType(%q) = false, want true", got[i])
		}
		if !IsSupportedDecisionAlgorithmType(got[i]) {
			t.Errorf("Looper algorithm %q is missing from the public decision catalog", got[i])
		}
	}

	for _, algorithmType := range []string{DecisionAlgorithmStatic, "rl_driven", "unknown"} {
		if IsLooperAlgorithmType(algorithmType) {
			t.Errorf("IsLooperAlgorithmType(%q) = true, want false", algorithmType)
		}
	}
}

func TestDecisionAlgorithmCatalog_PublicAlgorithmSurface(t *testing.T) {
	publicTypes := []string{
		"automix", "confidence", "fusion", "hybrid", "kmeans",
		"knn", "latency_aware", "mlp", "multi_factor", "ratings",
		"remom", "router_dc", "static", "svm",
	}

	for _, algType := range publicTypes {
		if !IsSupportedDecisionAlgorithmType(algType) {
			t.Errorf("public algorithm type %q is not supported", algType)
		}
	}
	for _, migratedType := range []string{"elo", "rl_driven", "gmtrouter", "session_aware"} {
		if IsSupportedDecisionAlgorithmType(migratedType) {
			t.Errorf("learning-owned algorithm type %q should not be public", migratedType)
		}
	}
}

func TestSignalCatalogSeparatesResponseOnlyObservationsFromDecisionInputs(t *testing.T) {
	decisionTypes := SupportedDecisionSignalTypes()
	if slices.Contains(decisionTypes, SignalTypeHallucination) {
		t.Fatal("response-stage hallucination signal must not be exposed as a decision input")
	}
	if !slices.Contains(SupportedSignalTypes(), SignalTypeHallucination) {
		t.Fatal("hallucination signal must remain available as a configurable observation")
	}
}

func TestSignalCatalogOwnsRuntimeObservationAndQualification(t *testing.T) {
	for _, entry := range SignalCatalog() {
		if entry.DecisionReferenceable && entry.ObservationKey == "" {
			t.Errorf("decision-referenceable signal %q has no observation key", entry.Type)
		}
		lookup, present := LookupSignalCatalog(entry.Type)
		if !present || !reflect.DeepEqual(lookup, entry) {
			t.Errorf("LookupSignalCatalog(%q) = %#v, want %#v", entry.Type, lookup, entry)
		}
	}

	inputModality, _ := LookupSignalCatalog(SignalTypeInputModality)
	if inputModality.ObservationKey != "input_modality" {
		t.Fatalf("input_modality observation key = %q", inputModality.ObservationKey)
	}
	complexity, _ := LookupSignalCatalog(SignalTypeComplexity)
	if complexity.ReferenceQualifier != SignalReferenceQualifierFixedSuffix {
		t.Fatalf("complexity reference qualifier = %q", complexity.ReferenceQualifier)
	}
	classifier, _ := LookupSignalCatalog(SignalTypeClassifier)
	if classifier.ReferenceQualifier != SignalReferenceQualifierLabel {
		t.Fatalf("classifier reference qualifier = %q", classifier.ReferenceQualifier)
	}
}

func TestGetAlgorithmTier(t *testing.T) {
	tests := []struct {
		algType      string
		expectedTier string
	}{
		{"static", "supported"},
		{"router_dc", "supported"},
		{"latency_aware", "supported"},
		{"hybrid", "supported"},
		{"automix", "experimental"},
		{"fusion", "experimental"},
		{"knn", "experimental"},
		{"kmeans", "experimental"},
		{"svm", "experimental"},
		{"mlp", "experimental"},
	}

	for _, tt := range tests {
		t.Run(tt.algType, func(t *testing.T) {
			tier := GetAlgorithmTier(tt.algType)
			if tier != tt.expectedTier {
				t.Errorf("GetAlgorithmTier(%q) = %q, want %q", tt.algType, tier, tt.expectedTier)
			}
		})
	}
}
