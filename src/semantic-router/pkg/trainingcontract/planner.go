package trainingcontract

import (
	"fmt"
	"maps"
	"math"
	"reflect"
	"slices"
	"sort"
	"strings"

	"github.com/invopop/jsonschema"
)

type DiagnosticSeverity string

const (
	SeverityError   DiagnosticSeverity = "error"
	SeverityWarning DiagnosticSeverity = "warning"
)

func (DiagnosticSeverity) JSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type: "string",
		Enum: []any{SeverityError, SeverityWarning},
	}
}

type DiagnosticCode string

const (
	CodeUnsupportedTarget        DiagnosticCode = "UNSUPPORTED_TARGET"
	CodeUnknownCapability        DiagnosticCode = "UNKNOWN_CAPABILITY"
	CodeIncompatibleArchitecture DiagnosticCode = "INCOMPATIBLE_ARCHITECTURE"
	CodeIncompatibleExecutor     DiagnosticCode = "INCOMPATIBLE_EXECUTOR"
	CodeIncompatibleHardware     DiagnosticCode = "INCOMPATIBLE_HARDWARE"
	CodeIncompatiblePrecision    DiagnosticCode = "INCOMPATIBLE_PRECISION"
	CodeMissingFormatConversion  DiagnosticCode = "MISSING_FORMAT_CONVERSION"
	CodeUnsupportedQualification DiagnosticCode = "UNSUPPORTED_QUALIFICATION"
	CodeInvalidParameter         DiagnosticCode = "INVALID_PARAMETER"
)

func (DiagnosticCode) JSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type: "string",
		Enum: []any{
			CodeUnsupportedTarget,
			CodeUnknownCapability,
			CodeIncompatibleArchitecture,
			CodeIncompatibleExecutor,
			CodeIncompatibleHardware,
			CodeIncompatiblePrecision,
			CodeMissingFormatConversion,
			CodeUnsupportedQualification,
			CodeInvalidParameter,
		},
	}
}

// PlanDiagnostic provides machine-readable error codes and human actionable diagnostics.
type PlanDiagnostic struct {
	Code        DiagnosticCode     `json:"code"`
	Severity    DiagnosticSeverity `json:"severity"`
	Field       string             `json:"field,omitempty"`
	Message     string             `json:"message"`
	Remediation string             `json:"remediation,omitempty"`
}

// QualificationTargetRequest specifies an inference runtime qualification target.
type QualificationTargetRequest struct {
	Key       string       `json:"key"`
	Runtime   CapabilityID `json:"runtime"`
	Hardware  CapabilityID `json:"hardware"`
	Precision CapabilityID `json:"precision"`
}

// TrainingPlanRequest submits a proposed training configuration to be planned.
type TrainingPlanRequest struct {
	SchemaVersion        string                       `json:"schema_version" jsonschema:"enum=semantic-router.training/v2"`
	TargetContract       Target                       `json:"target_contract"`
	Architecture         *CapabilityID                `json:"architecture,omitempty"`
	Trainer              CapabilityID                 `json:"trainer"`
	Executor             *CapabilityID                `json:"executor,omitempty"`
	TrainingHardware     CapabilityID                 `json:"training_hardware"`
	TrainingPrecision    CapabilityID                 `json:"training_precision"`
	Parameters           map[string]any               `json:"parameters,omitempty"`
	QualificationTargets []QualificationTargetRequest `json:"qualification_targets,omitempty"`
}

// PlannedQualification describes a concrete qualification step against an artifact variant.
type PlannedQualification struct {
	Key       string       `json:"key"`
	TaskKey   string       `json:"task_key"`
	Runtime   CapabilityID `json:"runtime"`
	Hardware  CapabilityID `json:"hardware"`
	Precision CapabilityID `json:"precision"`
	Connector string       `json:"connector"`
}

// PlannedVariant describes an artifact variant to be produced and qualified.
type PlannedVariant struct {
	Key            string                 `json:"key"`
	Format         CapabilityID           `json:"format"`
	ProducingTask  string                 `json:"producing_task"`
	Qualifications []PlannedQualification `json:"qualifications"`
}

// ResolvedTrainingPlan is the executable, deterministic plan resulting from capability planning.
type ResolvedTrainingPlan struct {
	TargetContract     Target           `json:"target_contract"`
	Trainer            CapabilityID     `json:"trainer"`
	Architecture       CapabilityID     `json:"architecture,omitempty"`
	Executor           CapabilityID     `json:"executor"`
	TrainingHardware   CapabilityID     `json:"training_hardware"`
	TrainingPrecision  CapabilityID     `json:"training_precision"`
	Tasks              []TaskSpec       `json:"tasks"`
	ArtifactVariants   []PlannedVariant `json:"artifact_variants"`
	ResolvedParameters map[string]any   `json:"resolved_parameters"`
}

// TrainingPlanResponse returns either a fully resolved plan or structured diagnostics.
type TrainingPlanResponse struct {
	SchemaVersion string                `json:"schema_version" jsonschema:"enum=semantic-router.training/v2"`
	Valid         bool                  `json:"valid"`
	Plan          *ResolvedTrainingPlan `json:"plan,omitempty"`
	Diagnostics   []PlanDiagnostic      `json:"diagnostics,omitempty"`
}

// Planner resolves training requests against a CapabilityRegistry.
type Planner struct {
	registry *CapabilityRegistry
}

func NewPlanner(registry *CapabilityRegistry) *Planner {
	if registry == nil {
		registry = DefaultRegistry()
	}
	return &Planner{registry: registry}
}

func (p *Planner) Plan(req TrainingPlanRequest) TrainingPlanResponse {
	var diagnostics []PlanDiagnostic

	addError := func(code DiagnosticCode, field, msg, remediation string) {
		diagnostics = append(diagnostics, PlanDiagnostic{
			Code:        code,
			Severity:    SeverityError,
			Field:       field,
			Message:     msg,
			Remediation: remediation,
		})
	}

	if req.SchemaVersion != Version {
		addError(CodeInvalidParameter, "schema_version",
			fmt.Sprintf("unsupported schema_version %q, expected %q", req.SchemaVersion, Version),
			fmt.Sprintf("Use schema_version %q", Version))
	}

	if err := ValidateTarget(req.TargetContract); err != nil {
		addError(CodeUnsupportedTarget, "target_contract",
			fmt.Sprintf("invalid target contract: %v", err),
			"Choose one of: selector.model-choice/v1, signal.label-scores/v1, signal.spans/v1")
	}

	// 1. Resolve Trainer
	trainer, ok := p.registry.GetTrainer(req.Trainer)
	if !ok {
		addError(CodeUnknownCapability, "trainer",
			fmt.Sprintf("unknown trainer capability ID %q", req.Trainer),
			"Register the trainer descriptor or choose an existing trainer from the capability catalog")
	} else if !slices.Contains(trainer.SupportedTargets, req.TargetContract) {
		addError(CodeUnsupportedTarget, "trainer",
			fmt.Sprintf("trainer %q does not support target contract %q", req.Trainer, req.TargetContract),
			fmt.Sprintf("Trainer supports targets: %v", trainer.SupportedTargets))
	}

	// 2. Resolve Architecture (if provided)
	var resolvedArch CapabilityID
	if req.Architecture != nil {
		resolvedArch = *req.Architecture
		arch, archOk := p.registry.GetArchitecture(resolvedArch)
		if !archOk {
			addError(CodeUnknownCapability, "architecture",
				fmt.Sprintf("unknown architecture driver capability ID %q", resolvedArch),
				"Register the architecture driver or select from the capability catalog")
		} else {
			if !slices.Contains(arch.SupportedTargets, req.TargetContract) {
				addError(CodeIncompatibleArchitecture, "architecture",
					fmt.Sprintf("architecture %q does not support target contract %q", resolvedArch, req.TargetContract),
					fmt.Sprintf("Architecture supports targets: %v", arch.SupportedTargets))
			}
			if ok && len(trainer.SupportedArchitectures) > 0 && !slices.Contains(trainer.SupportedArchitectures, resolvedArch) {
				addError(CodeIncompatibleArchitecture, "architecture",
					fmt.Sprintf("trainer %q is not compatible with architecture %q", req.Trainer, resolvedArch),
					fmt.Sprintf("Trainer supports architectures: %v", trainer.SupportedArchitectures))
			}
		}
	} else if ok && len(trainer.SupportedArchitectures) > 0 {
		// Default to first supported architecture
		resolvedArch = trainer.SupportedArchitectures[0]
	}
	archDesc, archKnown := p.registry.GetArchitecture(resolvedArch)

	// 3. Resolve Training Hardware
	hw, hwOk := p.registry.GetHardware(req.TrainingHardware)
	if !hwOk {
		addError(CodeUnknownCapability, "training_hardware",
			fmt.Sprintf("unknown hardware capability ID %q", req.TrainingHardware),
			"Select an available hardware provider from the capability catalog")
	} else if ok && !slices.Contains(trainer.SupportedHardware, req.TrainingHardware) {
		addError(CodeIncompatibleHardware, "training_hardware",
			fmt.Sprintf("trainer %q does not support training hardware %q", req.Trainer, req.TrainingHardware),
			fmt.Sprintf("Trainer supports hardware: %v", trainer.SupportedHardware))
	}

	// 4. Resolve Training Precision
	_, precOk := p.registry.GetPrecision(req.TrainingPrecision)
	if !precOk {
		addError(CodeUnknownCapability, "training_precision",
			fmt.Sprintf("unknown precision capability ID %q", req.TrainingPrecision),
			"Select an available precision mode from the capability catalog")
	} else {
		if ok && !slices.Contains(trainer.SupportedPrecisions, req.TrainingPrecision) {
			addError(CodeIncompatiblePrecision, "training_precision",
				fmt.Sprintf("trainer %q does not support precision %q", req.Trainer, req.TrainingPrecision),
				fmt.Sprintf("Trainer supports precisions: %v", trainer.SupportedPrecisions))
		}
		if hwOk && !slices.Contains(hw.SupportedPrecisions, req.TrainingPrecision) {
			addError(CodeIncompatiblePrecision, "training_precision",
				fmt.Sprintf("hardware %q does not support precision %q", req.TrainingHardware, req.TrainingPrecision),
				fmt.Sprintf("Hardware supports precisions: %v", hw.SupportedPrecisions))
		}
	}

	// 5. Resolve Executor
	var resolvedExecutor CapabilityID
	if req.Executor != nil {
		resolvedExecutor = *req.Executor
		exec, execOk := p.registry.GetExecutor(resolvedExecutor)
		if !execOk {
			addError(CodeUnknownCapability, "executor",
				fmt.Sprintf("unknown executor capability ID %q", resolvedExecutor),
				"Register the executor or choose from the capability catalog")
		} else {
			if ok && !slices.Contains(trainer.SupportedExecutors, resolvedExecutor) {
				addError(CodeIncompatibleExecutor, "executor",
					fmt.Sprintf("trainer %q does not support executor %q", req.Trainer, resolvedExecutor),
					fmt.Sprintf("Trainer supports executors: %v", trainer.SupportedExecutors))
			}
			if hwOk && !slices.Contains(exec.SupportedHardware, req.TrainingHardware) {
				addError(CodeIncompatibleHardware, "executor",
					fmt.Sprintf("executor %q does not support training hardware %q", resolvedExecutor, req.TrainingHardware),
					fmt.Sprintf("Executor supports hardware: %v", exec.SupportedHardware))
			}
		}
	} else if ok {
		// Pick first supported executor that supports the requested training hardware
		found := false
		for _, execID := range trainer.SupportedExecutors {
			if exec, eOk := p.registry.GetExecutor(execID); eOk {
				if slices.Contains(exec.SupportedHardware, req.TrainingHardware) {
					resolvedExecutor = execID
					found = true
					break
				}
			}
		}
		if !found {
			addError(CodeIncompatibleHardware, "executor",
				fmt.Sprintf("no executor for trainer %q supports hardware %q", req.Trainer, req.TrainingHardware),
				"Select a compatible training hardware provider")
		}
	}

	// 6. Validate & Resolve Parameters
	resolvedParams := make(map[string]any)
	if ok {
		for _, paramName := range slices.Sorted(maps.Keys(trainer.Parameters)) {
			constraint := trainer.Parameters[paramName]
			val, present := req.Parameters[paramName]
			if !present {
				if constraint.Required {
					addError(CodeInvalidParameter, fmt.Sprintf("parameters.%s", paramName),
						fmt.Sprintf("required parameter %q is missing", paramName),
						fmt.Sprintf("Specify parameter %q of type %s", paramName, constraint.Type))
				} else if constraint.Default != nil {
					resolvedParams[paramName] = constraint.Default
				}
				continue
			}

			// Validate type and constraints
			field := fmt.Sprintf("parameters.%s", paramName)
			number, isNumber := numberValue(val)
			validType := true
			switch constraint.Type {
			case "int":
				validType = isNumber && math.Mod(number, 1.0) == 0
			case "float":
				validType = isNumber
			case "string":
				strVal, isStr := val.(string)
				if !isStr {
					validType = false
				} else if len(constraint.Enum) > 0 && !slices.Contains(constraint.Enum, strVal) {
					addError(CodeInvalidParameter, field,
						fmt.Sprintf("parameter %q value %q is not in allowed enum %v", paramName, strVal, constraint.Enum),
						fmt.Sprintf("Choose one of: %v", constraint.Enum))
				}
			case "bool":
				if _, isBool := val.(bool); !isBool {
					validType = false
				}
			case "array":
				validType = val != nil && reflect.TypeOf(val).Kind() == reflect.Slice
			}

			switch {
			case !validType:
				addError(CodeInvalidParameter, field,
					fmt.Sprintf("parameter %q has invalid type, expected %s", paramName, constraint.Type),
					fmt.Sprintf("Provide a valid %s value", constraint.Type))
			case isNumber && constraint.Minimum != nil && number < *constraint.Minimum:
				addError(CodeInvalidParameter, field,
					fmt.Sprintf("parameter %q value %v is below the minimum %v", paramName, val, *constraint.Minimum),
					fmt.Sprintf("Use a value of at least %v", *constraint.Minimum))
			case isNumber && constraint.Maximum != nil && number > *constraint.Maximum:
				addError(CodeInvalidParameter, field,
					fmt.Sprintf("parameter %q value %v is above the maximum %v", paramName, val, *constraint.Maximum),
					fmt.Sprintf("Use a value of at most %v", *constraint.Maximum))
			default:
				resolvedParams[paramName] = val
			}
		}
	}
	// Copy through additional parameters not explicitly in trainer schema
	for k, v := range req.Parameters {
		if _, exists := resolvedParams[k]; !exists {
			resolvedParams[k] = v
		}
	}

	// 7. Plan Tasks, Artifact Variants, and Qualification Targets
	var tasks []TaskSpec
	var variants []PlannedVariant

	if ok && len(trainer.ProducedFormats) > 0 {
		primaryFormat := trainer.ProducedFormats[0]
		primaryVariant := PlannedVariant{
			Key:            "primary",
			Format:         primaryFormat,
			ProducingTask:  "train",
			Qualifications: make([]PlannedQualification, 0),
		}

		coreTasks := []TaskSpec{
			{
				Key:      "train",
				Executor: resolvedExecutor.ToComponent(),
			},
			{
				Key:       "evaluate",
				DependsOn: []string{"train"},
				Executor:  Component{Name: "evaluate", Version: "1"},
			},
		}

		var exportTasks []TaskSpec
		var qualifyTasks []TaskSpec

		// Variants lookup
		variantsMap := map[CapabilityID]*PlannedVariant{
			primaryFormat: &primaryVariant,
		}

		// Qualification targets
		seenQualificationKeys := make(map[string]struct{}, len(req.QualificationTargets))
		for _, qTarget := range req.QualificationTargets {
			if _, exists := seenQualificationKeys[qTarget.Key]; exists {
				addError(CodeInvalidParameter, fmt.Sprintf("qualification_targets.%s.key", qTarget.Key),
					fmt.Sprintf("duplicate qualification target key %q", qTarget.Key),
					"Use a unique qualification target key")
				continue
			}
			seenQualificationKeys[qTarget.Key] = struct{}{}
			runtime, rOk := p.registry.GetRuntime(qTarget.Runtime)
			if !rOk {
				addError(CodeUnknownCapability, fmt.Sprintf("qualification_targets.%s.runtime", qTarget.Key),
					fmt.Sprintf("unknown runtime capability ID %q", qTarget.Runtime),
					"Select an available runtime adapter from the capability catalog")
				continue
			}
			if !slices.Contains(runtime.SupportedTargets, req.TargetContract) {
				addError(CodeUnsupportedQualification, fmt.Sprintf("qualification_targets.%s.runtime", qTarget.Key),
					fmt.Sprintf("runtime %q does not support target contract %q", qTarget.Runtime, req.TargetContract),
					fmt.Sprintf("Runtime supports targets: %v", runtime.SupportedTargets))
				continue
			}
			if archKnown && len(archDesc.SupportedRuntimes) > 0 && !slices.Contains(archDesc.SupportedRuntimes, qTarget.Runtime) {
				addError(CodeIncompatibleArchitecture, fmt.Sprintf("qualification_targets.%s.runtime", qTarget.Key),
					fmt.Sprintf("architecture %q does not support runtime %q", resolvedArch, qTarget.Runtime),
					fmt.Sprintf("Architecture supports runtimes: %v", archDesc.SupportedRuntimes))
				continue
			}
			qHw, qHwOk := p.registry.GetHardware(qTarget.Hardware)
			if !qHwOk {
				addError(CodeUnknownCapability, fmt.Sprintf("qualification_targets.%s.hardware", qTarget.Key),
					fmt.Sprintf("unknown qualification hardware capability ID %q", qTarget.Hardware),
					"Select an available hardware provider from the capability catalog")
				continue
			}
			if !slices.Contains(runtime.SupportedHardware, qTarget.Hardware) {
				addError(CodeIncompatibleHardware, fmt.Sprintf("qualification_targets.%s.hardware", qTarget.Key),
					fmt.Sprintf("runtime %q does not support qualification hardware %q", qTarget.Runtime, qTarget.Hardware),
					fmt.Sprintf("Runtime supports hardware: %v", runtime.SupportedHardware))
				continue
			}
			if !slices.Contains(runtime.SupportedPrecisions, qTarget.Precision) {
				addError(CodeIncompatiblePrecision, fmt.Sprintf("qualification_targets.%s.precision", qTarget.Key),
					fmt.Sprintf("runtime %q does not support qualification precision %q", qTarget.Runtime, qTarget.Precision),
					fmt.Sprintf("Runtime supports precisions: %v", runtime.SupportedPrecisions))
				continue
			}
			if !slices.Contains(qHw.SupportedPrecisions, qTarget.Precision) {
				addError(CodeIncompatiblePrecision, fmt.Sprintf("qualification_targets.%s.precision", qTarget.Key),
					fmt.Sprintf("qualification hardware %q does not support precision %q", qTarget.Hardware, qTarget.Precision),
					fmt.Sprintf("Hardware supports precisions: %v", qHw.SupportedPrecisions))
				continue
			}

			// Check format compatibility: can runtime consume primaryFormat directly?
			targetVariantKey := "primary"
			qualifyDependsOn := []string{"evaluate"}

			if slices.Contains(runtime.AcceptedFormats, primaryFormat) {
				// Direct compatibility
			} else {
				// Needs format conversion
				var foundConversion *ConversionRule
				for _, acceptedFmt := range runtime.AcceptedFormats {
					if rule, cOk := p.registry.FindConversion(primaryFormat, acceptedFmt); cOk {
						foundConversion = &rule
						break
					}
				}

				if foundConversion == nil {
					addError(CodeMissingFormatConversion, fmt.Sprintf("qualification_targets.%s", qTarget.Key),
						fmt.Sprintf("runtime %q accepts formats %v, but trainer produces format %q and no automated conversion rule exists",
							qTarget.Runtime, runtime.AcceptedFormats, primaryFormat),
						"Add a conversion rule or select a runtime that directly accepts the produced format")
					continue
				}

				formatIdentity := strings.NewReplacer("/", "-", "@", "-").Replace(string(foundConversion.TargetFormat))
				exportTaskKey := "export-" + formatIdentity
				targetVariantKey = "converted-" + formatIdentity

				if _, exists := variantsMap[foundConversion.TargetFormat]; !exists {
					exportTask := TaskSpec{
						Key:       exportTaskKey,
						DependsOn: []string{"train"},
						Executor:  foundConversion.Executor.ToComponent(),
					}
					exportTasks = append(exportTasks, exportTask)

					convertedVariant := PlannedVariant{
						Key:            targetVariantKey,
						Format:         foundConversion.TargetFormat,
						ProducingTask:  exportTaskKey,
						Qualifications: make([]PlannedQualification, 0),
					}
					variantsMap[foundConversion.TargetFormat] = &convertedVariant
				}

				qualifyDependsOn = []string{exportTaskKey, "evaluate"}
			}

			qualifyTaskKey := "qualify-" + qTarget.Key
			qualifyTasks = append(qualifyTasks, TaskSpec{
				Key:       qualifyTaskKey,
				DependsOn: qualifyDependsOn,
				Executor:  Component{Name: "qualify", Version: "1"},
			})

			plannedQ := PlannedQualification{
				Key:       qTarget.Key,
				TaskKey:   qualifyTaskKey,
				Runtime:   qTarget.Runtime,
				Hardware:  qTarget.Hardware,
				Precision: qTarget.Precision,
				Connector: runtime.Connector,
			}

			if targetVariantKey == "primary" {
				primaryVariant.Qualifications = append(primaryVariant.Qualifications, plannedQ)
			} else {
				for _, v := range variantsMap {
					if v.Key == targetVariantKey {
						v.Qualifications = append(v.Qualifications, plannedQ)
					}
				}
			}
		}

		tasks = append(append(coreTasks, exportTasks...), qualifyTasks...)

		// Collect variants deterministically
		variantFormats := make([]CapabilityID, 0, len(variantsMap))
		for fmtID := range variantsMap {
			variantFormats = append(variantFormats, fmtID)
		}
		sort.Slice(variantFormats, func(i, j int) bool {
			// primary first, then alphabetical
			if variantFormats[i] == primaryFormat {
				return true
			}
			if variantFormats[j] == primaryFormat {
				return false
			}
			return variantFormats[i] < variantFormats[j]
		})

		for _, fmtID := range variantFormats {
			variants = append(variants, *variantsMap[fmtID])
		}
	}

	// Check if any errors occurred
	hasErrors := false
	for _, d := range diagnostics {
		if d.Severity == SeverityError {
			hasErrors = true
			break
		}
	}

	if hasErrors {
		return TrainingPlanResponse{
			SchemaVersion: Version,
			Valid:         false,
			Diagnostics:   diagnostics,
		}
	}

	return TrainingPlanResponse{
		SchemaVersion: Version,
		Valid:         true,
		Plan: &ResolvedTrainingPlan{
			TargetContract:     req.TargetContract,
			Trainer:            req.Trainer,
			Architecture:       resolvedArch,
			Executor:           resolvedExecutor,
			TrainingHardware:   req.TrainingHardware,
			TrainingPrecision:  req.TrainingPrecision,
			Tasks:              tasks,
			ArtifactVariants:   variants,
			ResolvedParameters: resolvedParams,
		},
		Diagnostics: diagnostics,
	}
}

func numberValue(val any) (float64, bool) {
	switch n := val.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}
