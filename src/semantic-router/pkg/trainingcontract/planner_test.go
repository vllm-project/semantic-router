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

func TestPlanSupportedNeuralQualifiesOnTheModelRuntime(t *testing.T) {
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
				Key:       "runtime-cpu",
				Runtime:   "runtime/model-runtime@v1",
				Hardware:  "hardware/cpu@v1",
				Precision: "precision/fp32@v1",
			},
			{
				Key:       "runtime-rocm",
				Runtime:   "runtime/model-runtime@v1",
				Hardware:  "hardware/rocm@v1",
				Precision: "precision/fp32@v1",
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
	expectedTasks := []string{"train", "evaluate", "qualify-runtime-cpu", "qualify-runtime-rocm"}
	if !reflect.DeepEqual(taskKeys, expectedTasks) {
		t.Errorf("expected tasks %v, got %v", expectedTasks, taskKeys)
	}
	for _, task := range plan.Tasks[2:] {
		if !reflect.DeepEqual(task.DependsOn, []string{"evaluate"}) {
			t.Errorf("%s should depend on evaluate only, got %v", task.Key, task.DependsOn)
		}
	}

	// The model runtime loads the trained Safetensors checkpoint as it is: no conversion.
	if len(plan.ArtifactVariants) != 1 {
		t.Fatalf("expected only the primary variant, got %+v", plan.ArtifactVariants)
	}
	primary := plan.ArtifactVariants[0]
	if primary.Format != "format/safetensors@v1" || len(primary.Qualifications) != 2 {
		t.Fatalf("unexpected primary variant: %+v", primary)
	}
	for _, qualification := range primary.Qualifications {
		if qualification.Runtime != "runtime/model-runtime@v1" || qualification.Connector != "sr.model-runtime.openapi.v2" {
			t.Errorf("unexpected qualification: %+v", qualification)
		}
	}

	req.QualificationTargets = []QualificationTargetRequest{
		{Key: "runtime-fp16", Runtime: "runtime/model-runtime@v1", Hardware: "hardware/rocm@v1", Precision: "precision/fp16@v1"},
	}
	resp = planner.Plan(req)
	if resp.Valid || len(resp.Diagnostics) != 1 || resp.Diagnostics[0].Code != CodeIncompatiblePrecision {
		t.Fatalf("the model runtime serves classifiers in FP32 only, got valid=%v diagnostics=%+v", resp.Valid, resp.Diagnostics)
	}

	for _, hardware := range []CapabilityID{"hardware/cuda@v1", "hardware/metal@v1"} {
		req.QualificationTargets = []QualificationTargetRequest{
			{Key: "runtime-unvalidated", Runtime: "runtime/model-runtime@v1", Hardware: hardware, Precision: "precision/fp32@v1"},
		}
		if resp = planner.Plan(req); resp.Valid {
			t.Fatalf("the model runtime has no readiness reference on %s, but the plan is valid", hardware)
		}
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
						Key:       "model-runtime",
						Runtime:   "runtime/model-runtime@v1",
						Hardware:  "hardware/cpu@v1",
						Precision: "precision/fp32@v1",
					},
				}
			},
			expectedCode: CodeUnsupportedQualification,
			expectedFld:  "qualification_targets.model-runtime.runtime",
		},
		{
			name: "retired embedded runtime",
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
			expectedCode: CodeUnknownCapability,
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
		SupportedRuntimes: []CapabilityID{"runtime/model-runtime@v1"},
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
				Key:       "model-runtime",
				Runtime:   "runtime/model-runtime@v1",
				Hardware:  "hardware/rocm@v1",
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

func TestPlanEnforcesDescriptorConstraints(t *testing.T) {
	registry := DefaultRegistry()
	minK, maxK := 1.0, 64.0
	edge := RuntimeAdapterDescriptor{
		ID:                  "runtime/custom-edge@v1",
		Component:           Component{Name: "custom-edge", Version: "1"},
		DisplayName:         "Custom Edge Runtime",
		SupportedTargets:    []Target{LabelScores},
		AcceptedFormats:     []CapabilityID{"format/safetensors@v1"},
		SupportedHardware:   []CapabilityID{"hardware/cpu@v1"},
		SupportedPrecisions: []CapabilityID{"precision/fp32@v1"},
		Connector:           "custom.edge.v1",
	}
	runtimeOnly := ArchitectureDriverDescriptor{
		ID:                "architecture/model-runtime-only@v1",
		Family:            "model-runtime-only",
		SupportedTargets:  []Target{LabelScores},
		SupportedFormats:  []CapabilityID{"format/safetensors@v1"},
		SupportedRuntimes: []CapabilityID{"runtime/model-runtime@v1"},
	}
	trainer := TrainerDescriptor{
		ID:                     "trainer/constrained@v1",
		Component:              Component{Name: "constrained", Version: "1"},
		SupportedTargets:       []Target{LabelScores},
		SupportedArchitectures: []CapabilityID{runtimeOnly.ID},
		SupportedExecutors:     []CapabilityID{"executor/train@v1"},
		SupportedHardware:      []CapabilityID{"hardware/cuda@v1"},
		SupportedPrecisions:    []CapabilityID{"precision/fp16@v1"},
		ProducedFormats:        []CapabilityID{"format/safetensors@v1"},
		Parameters: map[string]ParameterConstraint{
			"k":      {Type: "int", Minimum: &minK, Maximum: &maxK},
			"layers": {Type: "array"},
		},
	}
	if err := registry.RegisterRuntime(edge); err != nil {
		t.Fatal(err)
	}
	if err := registry.RegisterArchitecture(runtimeOnly); err != nil {
		t.Fatal(err)
	}
	if err := registry.RegisterTrainer(trainer); err != nil {
		t.Fatal(err)
	}
	planner := NewPlanner(registry)
	request := func() TrainingPlanRequest {
		return TrainingPlanRequest{
			SchemaVersion:     Version,
			TargetContract:    LabelScores,
			Trainer:           trainer.ID,
			TrainingHardware:  "hardware/cuda@v1",
			TrainingPrecision: "precision/fp16@v1",
			Parameters:        map[string]any{"k": 8.0, "layers": []any{256.0, 128.0}},
			QualificationTargets: []QualificationTargetRequest{{
				Key: "runtime-rocm", Runtime: "runtime/model-runtime@v1", Hardware: "hardware/rocm@v1", Precision: "precision/fp32@v1",
			}},
		}
	}
	if resp := planner.Plan(request()); !resp.Valid {
		t.Fatalf("expected constrained plan to be valid, got diagnostics: %+v", resp.Diagnostics)
	}

	cases := []struct {
		name   string
		modify func(*TrainingPlanRequest)
		code   DiagnosticCode
		field  string
	}{
		{"below minimum", func(r *TrainingPlanRequest) { r.Parameters["k"] = 0.0 }, CodeInvalidParameter, "parameters.k"},
		{"above maximum", func(r *TrainingPlanRequest) { r.Parameters["k"] = 65.0 }, CodeInvalidParameter, "parameters.k"},
		{"not an array", func(r *TrainingPlanRequest) { r.Parameters["layers"] = "256,128" }, CodeInvalidParameter, "parameters.layers"},
		{"runtime outside architecture", func(r *TrainingPlanRequest) {
			r.QualificationTargets[0] = QualificationTargetRequest{
				Key: "edge-cpu", Runtime: edge.ID, Hardware: "hardware/cpu@v1", Precision: "precision/fp32@v1",
			}
		}, CodeIncompatibleArchitecture, "qualification_targets.edge-cpu.runtime"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := request()
			tc.modify(&req)
			resp := planner.Plan(req)
			if resp.Valid || len(resp.Diagnostics) != 1 || resp.Diagnostics[0].Code != tc.code || resp.Diagnostics[0].Field != tc.field {
				t.Fatalf("expected only %s on %s, got valid=%v diagnostics=%+v", tc.code, tc.field, resp.Valid, resp.Diagnostics)
			}
		})
	}
}

func TestPlanParameterDiagnosticsFollowParameterNames(t *testing.T) {
	resp := NewPlanner(DefaultRegistry()).Plan(TrainingPlanRequest{
		SchemaVersion:     Version,
		TargetContract:    Selector,
		Trainer:           "trainer/selector@v1",
		TrainingHardware:  "hardware/cpu@v1",
		TrainingPrecision: "precision/fp32@v1",
		Parameters:        map[string]any{"seed": "abc", "kernel": "bogus", "normalize": "yes"},
	})
	fields := make([]string, 0, len(resp.Diagnostics))
	for _, d := range resp.Diagnostics {
		fields = append(fields, d.Field)
	}
	want := []string{"parameters.kernel", "parameters.normalize", "parameters.seed"}
	if !reflect.DeepEqual(fields, want) {
		t.Fatalf("expected diagnostics %v, got %v", want, fields)
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
	expected, err := json.MarshalIndent(DefaultRegistry().Catalog(), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	existing, err := os.ReadFile("testdata/capabilities.json")
	if err != nil {
		t.Fatal(err)
	}
	if string(existing) != string(expected)+"\n" {
		t.Fatal("testdata/capabilities.json is stale; run make training-contract-generate")
	}
}

func TestPlanRejectsDuplicateQualificationKeys(t *testing.T) {
	registry := DefaultRegistry()
	planner := NewPlanner(registry)
	architecture := CapabilityID("architecture/hf-modernbert@v1")
	request := TrainingPlanRequest{
		SchemaVersion:     Version,
		TargetContract:    LabelScores,
		Trainer:           "trainer/hf-peft@v1",
		Architecture:      &architecture,
		TrainingHardware:  "hardware/cuda@v1",
		TrainingPrecision: "precision/bf16@v1",
		Parameters:        map[string]any{"r": 16},
		QualificationTargets: []QualificationTargetRequest{
			{Key: "a", Runtime: "runtime/model-runtime@v1", Hardware: "hardware/cpu@v1", Precision: "precision/fp32@v1"},
			{Key: "a", Runtime: "runtime/model-runtime@v1", Hardware: "hardware/rocm@v1", Precision: "precision/fp32@v1"},
		},
	}

	response := planner.Plan(request)
	if response.Valid {
		if response.Plan == nil {
			t.Fatal("planner returned a valid response without a plan")
		}
		if err := ValidateRun(plannedSubmitRun(t, registry, response.Plan)); err != nil {
			t.Fatalf("planner marked a run valid but ValidateRun rejected it: %v", err)
		}
		t.Fatal("expected duplicate qualification keys to be rejected")
	}
	if response.Plan != nil {
		t.Fatal("expected an invalid response to omit the plan")
	}
	if len(response.Diagnostics) != 1 {
		t.Fatalf("expected one duplicate-key diagnostic, got %+v", response.Diagnostics)
	}
	diagnostic := response.Diagnostics[0]
	if diagnostic.Code != CodeInvalidParameter || diagnostic.Field != "qualification_targets.a.key" || diagnostic.Message == "" || diagnostic.Remediation == "" {
		t.Fatalf("unexpected duplicate-key diagnostic: %+v", diagnostic)
	}
}

// registerConvertedRuntime registers, out of tree, an ONNX format version, a
// runtime that accepts only that format, and the export rule from Safetensors,
// and lets the built-in ModernBERT architecture qualify on that runtime.
func registerConvertedRuntime(t *testing.T, registry *CapabilityRegistry, version string) CapabilityID {
	t.Helper()
	format := CapabilityID("format/onnx@v" + version)
	runtime := CapabilityID("runtime/onnx-edge-v" + version + "@v1")
	architecture, ok := registry.GetArchitecture("architecture/hf-modernbert@v1")
	if !ok {
		t.Fatal("expected built-in ModernBERT architecture")
	}
	architecture.SupportedFormats = append(append([]CapabilityID(nil), architecture.SupportedFormats...), format)
	architecture.SupportedRuntimes = append(append([]CapabilityID(nil), architecture.SupportedRuntimes...), runtime)
	if err := registry.RegisterArchitecture(architecture); err != nil {
		t.Fatal(err)
	}
	if err := registry.RegisterFormat(ArtifactFormatDescriptor{
		ID:             format,
		Component:      Component{Name: "onnx", Version: version},
		DisplayName:    "Open Neural Network Exchange v" + version,
		FileExtensions: []string{".onnx"},
		DirectRuntimes: []CapabilityID{runtime},
	}); err != nil {
		t.Fatal(err)
	}
	if err := registry.RegisterRuntime(RuntimeAdapterDescriptor{
		ID:                  runtime,
		Component:           Component{Name: "onnx-edge-v" + version, Version: "1"},
		DisplayName:         "ONNX edge runtime v" + version,
		SupportedTargets:    []Target{LabelScores},
		AcceptedFormats:     []CapabilityID{format},
		SupportedHardware:   []CapabilityID{"hardware/cuda@v1"},
		SupportedPrecisions: []CapabilityID{"precision/fp16@v1"},
		Connector:           "custom.onnx-edge.v" + version,
	}); err != nil {
		t.Fatal(err)
	}
	if err := registry.RegisterExecutor(ExecutorDescriptor{
		ID:                "executor/onnx-exporter@v1",
		Component:         Component{Name: "onnx-exporter", Version: "1"},
		DisplayName:       "Safetensors to ONNX Exporter",
		SupportedHardware: []CapabilityID{"hardware/cpu@v1", "hardware/cuda@v1"},
		IsolationLevel:    "container",
	}); err != nil {
		t.Fatal(err)
	}
	if err := registry.RegisterConversion(ConversionRule{
		SourceFormat: "format/safetensors@v1",
		TargetFormat: format,
		Executor:     "executor/onnx-exporter@v1",
		Hardware:     []CapabilityID{"hardware/cpu@v1", "hardware/cuda@v1"},
	}); err != nil {
		t.Fatal(err)
	}
	return runtime
}

func TestPlanPreservesFormatVersionsInConversionIdentities(t *testing.T) {
	registry := DefaultRegistry()
	runtimeV1 := registerConvertedRuntime(t, registry, "1")
	runtimeV2 := registerConvertedRuntime(t, registry, "2")
	architectureID := CapabilityID("architecture/hf-modernbert@v1")

	response := NewPlanner(registry).Plan(TrainingPlanRequest{
		SchemaVersion:     Version,
		TargetContract:    LabelScores,
		Trainer:           "trainer/hf-peft@v1",
		Architecture:      &architectureID,
		TrainingHardware:  "hardware/cuda@v1",
		TrainingPrecision: "precision/bf16@v1",
		Parameters:        map[string]any{"r": 16},
		QualificationTargets: []QualificationTargetRequest{
			{Key: "runtime-cpu", Runtime: "runtime/model-runtime@v1", Hardware: "hardware/cpu@v1", Precision: "precision/fp32@v1"},
			{Key: "onnx-v1", Runtime: runtimeV1, Hardware: "hardware/cuda@v1", Precision: "precision/fp16@v1"},
			{Key: "onnx-v2", Runtime: runtimeV2, Hardware: "hardware/cuda@v1", Precision: "precision/fp16@v1"},
		},
	})
	if !response.Valid || response.Plan == nil {
		t.Fatalf("expected versioned conversion plan to be valid, got diagnostics: %+v", response.Diagnostics)
	}
	if err := ValidateRun(plannedSubmitRun(t, registry, response.Plan)); err != nil {
		t.Fatalf("ValidateRun rejected the versioned conversion plan: %v", err)
	}

	taskKeys := make([]string, len(response.Plan.Tasks))
	dependencies := map[string][]string{}
	for i, task := range response.Plan.Tasks {
		taskKeys[i] = task.Key
		dependencies[task.Key] = task.DependsOn
	}
	wantTaskKeys := []string{"train", "evaluate", "export-format-onnx-v1", "export-format-onnx-v2", "qualify-runtime-cpu", "qualify-onnx-v1", "qualify-onnx-v2"}
	if !reflect.DeepEqual(taskKeys, wantTaskKeys) {
		t.Fatalf("expected distinct versioned tasks %v, got %v", wantTaskKeys, taskKeys)
	}
	for key, want := range map[string][]string{
		"export-format-onnx-v1": {"train"},
		"qualify-runtime-cpu":   {"evaluate"},
		"qualify-onnx-v1":       {"export-format-onnx-v1", "evaluate"},
		"qualify-onnx-v2":       {"export-format-onnx-v2", "evaluate"},
	} {
		if !reflect.DeepEqual(dependencies[key], want) {
			t.Errorf("%s depends on %v, want %v", key, dependencies[key], want)
		}
	}
	if len(response.Plan.ArtifactVariants) != 3 {
		t.Fatalf("expected primary and two converted variants, got %+v", response.Plan.ArtifactVariants)
	}
	wantVariants := []struct {
		key           string
		format        CapabilityID
		qualification string
	}{
		{key: "primary", format: "format/safetensors@v1", qualification: "runtime-cpu"},
		{key: "converted-format-onnx-v1", format: "format/onnx@v1", qualification: "onnx-v1"},
		{key: "converted-format-onnx-v2", format: "format/onnx@v2", qualification: "onnx-v2"},
	}
	for i, want := range wantVariants {
		variant := response.Plan.ArtifactVariants[i]
		if variant.Key != want.key || variant.Format != want.format {
			t.Errorf("variant %d = (%q, %q), want (%q, %q)", i, variant.Key, variant.Format, want.key, want.format)
		}
		if len(variant.Qualifications) != 1 || variant.Qualifications[0].Key != want.qualification {
			t.Errorf("variant %q has qualifications %+v, want only %q", variant.Key, variant.Qualifications, want.qualification)
		}
	}
}

func plannedSubmitRun(t *testing.T, registry *CapabilityRegistry, plan *ResolvedTrainingPlan) SubmitRunRequest {
	t.Helper()
	trainer, ok := registry.GetTrainer(plan.Trainer)
	if !ok {
		t.Fatalf("expected trainer %q in the capability registry", plan.Trainer)
	}
	return SubmitRunRequest{
		SchemaVersion:  Version,
		IdempotencyKey: "planner-regression",
		Spec: RunSpec{
			ExperimentID:   "experiment_planner",
			SnapshotID:     "snapshot_planner",
			TargetContract: plan.TargetContract,
			Trainer:        trainer.Component,
			Parameters:     plan.ResolvedParameters,
			Tasks:          plan.Tasks,
		},
	}
}
