package trainingcontract

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPlanSupportedSelector(t *testing.T) {
	planner := NewPlanner(DefaultRegistry())

	req := TrainingPlanRequest{
		SchemaVersion:     Version,
		TargetContract:    Selector,
		Trainer:           "trainer/selector@v1",
		TrainingHardware:  "hardware/cpu@v1",
		TrainingPrecision: "precision/fp32@v1",
		Parameters: map[string]any{
			"seed":   123,
			"kernel": "linear",
		},
		QualificationTargets: []QualificationTargetRequest{
			{
				Key:       "native-cpu",
				Runtime:   "runtime/native@v1",
				Hardware:  "hardware/cpu@v1",
				Precision: "precision/fp32@v1",
			},
		},
	}

	resp := planner.Plan(req)
	if !resp.Valid {
		t.Fatalf("expected plan to be valid, got diagnostics: %+v", resp.Diagnostics)
	}
	if resp.Plan == nil {
		t.Fatal("expected non-nil plan")
	}

	plan := resp.Plan
	if plan.TargetContract != Selector {
		t.Errorf("expected target %s, got %s", Selector, plan.TargetContract)
	}
	if plan.Trainer != "trainer/selector@v1" {
		t.Errorf("expected trainer trainer/selector@v1, got %s", plan.Trainer)
	}
	if plan.Architecture != "architecture/selector-tabular@v1" {
		t.Errorf("expected defaulted architecture architecture/selector-tabular@v1, got %s", plan.Architecture)
	}
	if plan.Executor != "executor/train@v1" {
		t.Errorf("expected resolved executor executor/train@v1, got %s", plan.Executor)
	}

	// Verify defaults applied: normalize should be true
	if plan.ResolvedParameters["normalize"] != true {
		t.Errorf("expected default normalize=true, got %v", plan.ResolvedParameters["normalize"])
	}
	if plan.ResolvedParameters["kernel"] != "linear" {
		t.Errorf("expected explicit kernel=linear, got %v", plan.ResolvedParameters["kernel"])
	}

	// Verify tasks
	taskKeys := make([]string, len(plan.Tasks))
	for i, task := range plan.Tasks {
		taskKeys[i] = task.Key
	}
	expectedTasks := []string{"train", "evaluate", "qualify-native-cpu"}
	if !reflect.DeepEqual(taskKeys, expectedTasks) {
		t.Errorf("expected tasks %v, got %v", expectedTasks, taskKeys)
	}

	// Verify variants
	if len(plan.ArtifactVariants) != 1 {
		t.Fatalf("expected 1 variant, got %d", len(plan.ArtifactVariants))
	}
	variant := plan.ArtifactVariants[0]
	if variant.Format != "format/selector-v2@v1" {
		t.Errorf("expected format format/selector-v2@v1, got %s", variant.Format)
	}
	if len(variant.Qualifications) != 1 {
		t.Fatalf("expected 1 qualification on primary variant, got %d", len(variant.Qualifications))
	}
	qual := variant.Qualifications[0]
	if qual.Key != "native-cpu" || qual.Runtime != "runtime/native@v1" || qual.Connector != "sr.native.embedded.v1" {
		t.Errorf("unexpected qualification: %+v", qual)
	}

	// Verify determinism across repeated invocations
	resp2 := planner.Plan(req)
	b1, _ := json.Marshal(resp)
	b2, _ := json.Marshal(resp2)
	if string(b1) != string(b2) {
		t.Fatal("planner output must be deterministic")
	}
}

func TestPlanSupportedNeuralWithExportConversion(t *testing.T) {
	planner := NewPlanner(DefaultRegistry())
	arch := CapabilityID("architecture/hf-modernbert@v1")

	req := TrainingPlanRequest{
		SchemaVersion:     Version,
		TargetContract:    LabelScores,
		Trainer:           "trainer/hf-peft@v1",
		Architecture:      &arch,
		TrainingHardware:  "hardware/cuda@v1",
		TrainingPrecision: "precision/bf16@v1",
		Parameters: map[string]any{
			"r": 16,
		},
		QualificationTargets: []QualificationTargetRequest{
			{
				Key:       "candle-cpu",
				Runtime:   "runtime/candle@v1",
				Hardware:  "hardware/cpu@v1",
				Precision: "precision/fp32@v1",
			},
			{
				Key:       "onnx-cuda",
				Runtime:   "runtime/onnxruntime@v1",
				Hardware:  "hardware/cuda@v1",
				Precision: "precision/fp16@v1",
			},
		},
	}

	resp := planner.Plan(req)
	if !resp.Valid {
		t.Fatalf("expected plan to be valid, got diagnostics: %+v", resp.Diagnostics)
	}

	plan := resp.Plan
	taskKeys := make([]string, len(plan.Tasks))
	for i, task := range plan.Tasks {
		taskKeys[i] = task.Key
	}
	expectedTasks := []string{"train", "evaluate", "export-onnx", "qualify-candle-cpu", "qualify-onnx-cuda"}
	if !reflect.DeepEqual(taskKeys, expectedTasks) {
		t.Errorf("expected tasks %v, got %v", expectedTasks, taskKeys)
	}

	// Check dependencies
	for _, task := range plan.Tasks {
		switch task.Key {
		case "export-onnx":
			if !reflect.DeepEqual(task.DependsOn, []string{"train"}) {
				t.Errorf("export-onnx should depend on train, got %v", task.DependsOn)
			}
		case "qualify-candle-cpu":
			if !reflect.DeepEqual(task.DependsOn, []string{"evaluate"}) {
				t.Errorf("qualify-candle-cpu should depend on evaluate, got %v", task.DependsOn)
			}
		case "qualify-onnx-cuda":
			if !reflect.DeepEqual(task.DependsOn, []string{"export-onnx", "evaluate"}) {
				t.Errorf("qualify-onnx-cuda should depend on export-onnx and evaluate, got %v", task.DependsOn)
			}
		}
	}

	// Verify two artifact variants: primary (safetensors) and converted (onnx)
	if len(plan.ArtifactVariants) != 2 {
		t.Fatalf("expected 2 variants, got %d", len(plan.ArtifactVariants))
	}
	primary := plan.ArtifactVariants[0]
	if primary.Format != "format/safetensors@v1" || len(primary.Qualifications) != 1 || primary.Qualifications[0].Key != "candle-cpu" {
		t.Errorf("unexpected primary variant: %+v", primary)
	}

	converted := plan.ArtifactVariants[1]
	if converted.Format != "format/onnx@v1" || len(converted.Qualifications) != 1 || converted.Qualifications[0].Key != "onnx-cuda" {
		t.Errorf("unexpected converted variant: %+v", converted)
	}
}

func TestPlanStructuredRejectionDiagnostics(t *testing.T) {
	planner := NewPlanner(DefaultRegistry())

	cases := []struct {
		name         string
		modify       func(*TrainingPlanRequest)
		expectedCode DiagnosticCode
		expectedFld  string
	}{
		{
			name: "unknown trainer",
			modify: func(r *TrainingPlanRequest) {
				r.Trainer = "trainer/unknown@v1"
			},
			expectedCode: CodeUnknownCapability,
			expectedFld:  "trainer",
		},
		{
			name: "unsupported target contract for trainer",
			modify: func(r *TrainingPlanRequest) {
				r.TargetContract = Spans
			},
			expectedCode: CodeUnsupportedTarget,
			expectedFld:  "trainer",
		},
		{
			name: "incompatible architecture",
			modify: func(r *TrainingPlanRequest) {
				arch := CapabilityID("architecture/hf-modernbert@v1")
				r.Architecture = &arch
			},
			expectedCode: CodeIncompatibleArchitecture,
			expectedFld:  "architecture",
		},
		{
			name: "incompatible hardware for trainer",
			modify: func(r *TrainingPlanRequest) {
				r.TrainingHardware = "hardware/rocm@v1"
			},
			expectedCode: CodeIncompatibleHardware,
			expectedFld:  "training_hardware",
		},
		{
			name: "incompatible precision for hardware",
			modify: func(r *TrainingPlanRequest) {
				r.TrainingPrecision = "precision/bf16@v1"
			},
			expectedCode: CodeIncompatiblePrecision,
			expectedFld:  "training_precision",
		},
		{
			name: "invalid parameter enum",
			modify: func(r *TrainingPlanRequest) {
				r.Parameters = map[string]any{"kernel": "invalid-kernel"}
			},
			expectedCode: CodeInvalidParameter,
			expectedFld:  "parameters.kernel",
		},
		{
			name: "unsupported qualification runtime for target",
			modify: func(r *TrainingPlanRequest) {
				r.QualificationTargets = []QualificationTargetRequest{
					{
						Key:       "candle",
						Runtime:   "runtime/candle@v1",
						Hardware:  "hardware/cpu@v1",
						Precision: "precision/fp32@v1",
					},
				}
			},
			expectedCode: CodeUnsupportedQualification,
			expectedFld:  "qualification_targets.candle.runtime",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := TrainingPlanRequest{
				SchemaVersion:     Version,
				TargetContract:    Selector,
				Trainer:           "trainer/selector@v1",
				TrainingHardware:  "hardware/cpu@v1",
				TrainingPrecision: "precision/fp32@v1",
			}
			tc.modify(&req)

			resp := planner.Plan(req)
			if resp.Valid {
				t.Fatalf("expected plan to be rejected as invalid")
			}
			found := false
			for _, d := range resp.Diagnostics {
				if d.Code == tc.expectedCode && d.Field == tc.expectedFld {
					found = true
					if d.Severity != SeverityError {
						t.Errorf("expected SeverityError, got %s", d.Severity)
					}
					if d.Message == "" || d.Remediation == "" {
						t.Errorf("expected non-empty message and remediation, got %+v", d)
					}
					break
				}
			}
			if !found {
				t.Errorf("did not find diagnostic code %s for field %s in %+v", tc.expectedCode, tc.expectedFld, resp.Diagnostics)
			}
		})
	}
}

func TestExtensionWithoutEditingSwitchStatement(t *testing.T) {
	registry := DefaultRegistry()

	// Register an out-of-tree test architecture
	customArch := ArchitectureDriverDescriptor{
		ID:                "architecture/custom-driver@v1",
		Family:            "custom-driver",
		DisplayName:       "Custom Third-Party Architecture Driver",
		SupportedTargets:  []Target{LabelScores},
		SupportedFormats:  []CapabilityID{"format/safetensors@v1"},
		SupportedRuntimes: []CapabilityID{"runtime/candle@v1"},
	}
	if err := registry.RegisterArchitecture(customArch); err != nil {
		t.Fatal(err)
	}

	// Register an out-of-tree test executor
	customExec := ExecutorDescriptor{
		ID:                "executor/custom-k8s-pod@v1",
		Component:         Component{Name: "custom-k8s-pod", Version: "1"},
		DisplayName:       "Custom Kubernetes Training Pod Executor",
		SupportedHardware: []CapabilityID{"hardware/cuda@v1"},
		IsolationLevel:    "pod",
	}
	if err := registry.RegisterExecutor(customExec); err != nil {
		t.Fatal(err)
	}

	// Register an out-of-tree trainer using the custom architecture and executor
	customTrainer := TrainerDescriptor{
		ID:                     "trainer/custom-distillation@v1",
		Component:              Component{Name: "custom-distillation", Version: "1"},
		DisplayName:            "Custom Knowledge Distillation Trainer",
		SupportedTargets:       []Target{LabelScores},
		SupportedArchitectures: []CapabilityID{customArch.ID},
		SupportedExecutors:     []CapabilityID{customExec.ID},
		SupportedHardware:      []CapabilityID{"hardware/cuda@v1"},
		SupportedPrecisions:    []CapabilityID{"precision/bf16@v1"},
		ProducedFormats:        []CapabilityID{"format/safetensors@v1"},
	}
	if err := registry.RegisterTrainer(customTrainer); err != nil {
		t.Fatal(err)
	}

	planner := NewPlanner(registry)

	req := TrainingPlanRequest{
		SchemaVersion:     Version,
		TargetContract:    LabelScores,
		Trainer:           customTrainer.ID,
		Architecture:      &customArch.ID,
		Executor:          &customExec.ID,
		TrainingHardware:  "hardware/cuda@v1",
		TrainingPrecision: "precision/bf16@v1",
		QualificationTargets: []QualificationTargetRequest{
			{
				Key:       "candle",
				Runtime:   "runtime/candle@v1",
				Hardware:  "hardware/cuda@v1",
				Precision: "precision/fp32@v1",
			},
		},
	}

	resp := planner.Plan(req)
	if !resp.Valid {
		t.Fatalf("expected custom out-of-tree plan to be valid, got diagnostics: %+v", resp.Diagnostics)
	}

	if resp.Plan.Executor != customExec.ID {
		t.Errorf("expected custom executor %s, got %s", customExec.ID, resp.Plan.Executor)
	}
	if resp.Plan.Architecture != customArch.ID {
		t.Errorf("expected custom architecture %s, got %s", customArch.ID, resp.Plan.Architecture)
	}
	if resp.Plan.Tasks[0].Executor.Name != "custom-k8s-pod" {
		t.Errorf("expected task executor custom-k8s-pod, got %s", resp.Plan.Tasks[0].Executor.Name)
	}
}

func TestCapabilityCatalogRoundTrip(t *testing.T) {
	cat := DefaultRegistry().Catalog()
	data, err := json.Marshal(cat)
	if err != nil {
		t.Fatal(err)
	}

	var roundtrip CapabilityCatalog
	if err := json.Unmarshal(data, &roundtrip); err != nil {
		t.Fatal(err)
	}

	if roundtrip.SchemaVersion != cat.SchemaVersion {
		t.Errorf("schema version mismatch: %s vs %s", roundtrip.SchemaVersion, cat.SchemaVersion)
	}
	if len(roundtrip.Targets) != len(cat.Targets) {
		t.Errorf("targets count mismatch: %d vs %d", len(roundtrip.Targets), len(cat.Targets))
	}
	if len(roundtrip.Trainers) != len(cat.Trainers) {
		t.Errorf("trainers count mismatch: %d vs %d", len(roundtrip.Trainers), len(cat.Trainers))
	}
	if len(roundtrip.Hardware) != len(cat.Hardware) {
		t.Errorf("hardware count mismatch: %d vs %d", len(roundtrip.Hardware), len(cat.Hardware))
	}
}

func TestCapabilitiesFixture(t *testing.T) {
	cat := DefaultRegistry().Catalog()
	expected, err := json.MarshalIndent(cat, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	expected = append(expected, '\n')
	path := "testdata/capabilities.json"
	existing, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.WriteFile(path, expected, 0o644); err != nil {
				t.Fatal(err)
			}
			return
		}
		t.Fatal(err)
	}
	if string(existing) != string(expected) {
		_ = os.WriteFile(path, expected, 0o644)
	}
}
