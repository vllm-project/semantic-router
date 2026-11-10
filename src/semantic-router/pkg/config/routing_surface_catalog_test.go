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
		if entry.Kind != AlgorithmKindSelector && entry.Kind != AlgorithmKindExecutor {
			t.Errorf("Catalog entry %q has invalid Kind %q", entry.Type, entry.Kind)
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
	for _, entry := range registeredSpecs(decisionAlgorithms) {
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

func TestAlgorithmCatalogSeparatesSelectorsFromExecutors(t *testing.T) {
	selectors := []string{DecisionAlgorithmMultiFactor, DecisionAlgorithmRandom, DecisionAlgorithmStatic, DecisionAlgorithmSystemOne}
	executors := []string{DecisionAlgorithmEscalate, DecisionAlgorithmFusion, DecisionAlgorithmRatings, DecisionAlgorithmReMoM, DecisionAlgorithmWorkflows}
	if got := SupportedExecutorAlgorithmTypes(); !slices.Equal(got, executors) {
		t.Fatalf("SupportedExecutorAlgorithmTypes() = %v, want %v", got, executors)
	}
	if got, want := SupportedDecisionAlgorithmTypes(), slices.Sorted(slices.Values(slices.Concat(selectors, executors))); !slices.Equal(got, want) {
		t.Fatalf("SupportedDecisionAlgorithmTypes() = %v, want %v", got, want)
	}
	for _, algorithmType := range append(selectors, "unknown") {
		if IsExecutorAlgorithmType(algorithmType) {
			t.Errorf("IsExecutorAlgorithmType(%q) = true, want false", algorithmType)
		}
	}
	for _, entry := range DecisionAlgorithmCatalog() {
		wantAPIs := []InferenceAPI{ChatAPI}
		if entry.Type == DecisionAlgorithmEscalate {
			wantAPIs = []InferenceAPI{ChatAPI, SystemOneAPI}
		}
		if !slices.Equal(entry.APIs, wantAPIs) {
			t.Errorf("algorithm %q serves %v, want %v", entry.Type, entry.APIs, wantAPIs)
		}
		if entry.Kind == AlgorithmKindExecutor && (!IsExecutorAlgorithmType(entry.Type) || entry.ConfigField != entry.Type) {
			t.Errorf("executor %q must nest its settings under its type, got config field %q", entry.Type, entry.ConfigField)
		}
	}
}

func TestDecisionAlgorithmCatalog_RemovedAlgorithmsStayRemoved(t *testing.T) {
	for _, removedType := range removedAlgorithmTypes {
		if IsSupportedDecisionAlgorithmType(removedType) {
			t.Errorf("removed algorithm type %q is still public", removedType)
		}
		spec := NewDecisionAlgorithmType(AlgorithmCatalogEntry{Type: removedType, DisplayName: "Removed", Tier: "experimental"}, AlgorithmOptions[struct{}]{})
		if err := RegisterDecisionAlgorithm(spec); err == nil {
			t.Errorf("a Router build registered the removed algorithm type %q", removedType)
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

func TestGetAlgorithmTier(t *testing.T) {
	tests := []struct {
		algType      string
		expectedTier string
	}{
		{"static", "supported"},
		{"multi_factor", "supported"},
		{"systemone", "supported"},
		{"escalate", "supported"},
		{"fusion", "experimental"},
		{"workflows", "experimental"},
		{"router_dc", ""},
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
	for _, signalType := range []string{SignalTypeComplexity, SignalTypeTurnRelation, SignalTypeSafety, SignalTypeSystemOne} {
		if entry, _ := LookupSignalCatalog(signalType); entry.ReferenceQualifier != SignalReferenceQualifierLabel {
			t.Fatalf("%s reference qualifier = %q", signalType, entry.ReferenceQualifier)
		}
	}
	if turnRelation, _ := LookupSignalCatalog(SignalTypeTurnRelation); turnRelation.ObservationKey != "turn_relation" {
		t.Fatalf("turn_relation observation key = %q", turnRelation.ObservationKey)
	}
}
