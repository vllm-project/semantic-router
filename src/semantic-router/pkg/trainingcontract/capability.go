package trainingcontract

import (
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/invopop/jsonschema"
)

var capabilityIDPattern = regexp.MustCompile(`^[a-z]+/[a-z0-9._-]+@v[0-9]+$`)

// CapabilityID is a namespaced, versioned identifier (e.g. "trainer/hf-peft@v1").
// Format: <domain>/<name>@<version>
type CapabilityID string

func (c CapabilityID) JSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type:    "string",
		Pattern: capabilityIDPattern.String(),
	}
}

func (c CapabilityID) Valid() bool {
	return capabilityIDPattern.MatchString(string(c))
}

func (c CapabilityID) Domain() string {
	parts := strings.Split(string(c), "/")
	if len(parts) != 2 {
		return ""
	}
	return parts[0]
}

func (c CapabilityID) Name() string {
	parts := strings.Split(string(c), "/")
	if len(parts) != 2 {
		return ""
	}
	subparts := strings.Split(parts[1], "@")
	return subparts[0]
}

func (c CapabilityID) Version() string {
	parts := strings.Split(string(c), "@")
	if len(parts) != 2 {
		return ""
	}
	return parts[1]
}

func (c CapabilityID) ToComponent() Component {
	version := c.Version()
	return Component{
		Name:    c.Name(),
		Version: strings.TrimPrefix(version, "v"),
	}
}

func ValidateCapabilityID(id CapabilityID, expectedDomain string) error {
	if !id.Valid() {
		return fmt.Errorf("invalid capability ID %q (expected format <domain>/<name>@<version>)", id)
	}
	if expectedDomain != "" && id.Domain() != expectedDomain {
		return fmt.Errorf("capability ID %q has domain %q, expected %q", id, id.Domain(), expectedDomain)
	}
	return nil
}

// TargetDescriptor registers a versioned target contract.
type TargetDescriptor struct {
	ID             CapabilityID `json:"id"`
	TargetContract Target       `json:"target_contract"`
	DisplayName    string       `json:"display_name"`
	Description    string       `json:"description,omitempty"`
}

// ParameterConstraint defines allowed types, boundaries, and defaults for a parameter.
type ParameterConstraint struct {
	Type        string   `json:"type"` // "string", "int", "float", "bool", "array"
	Required    bool     `json:"required"`
	Default     any      `json:"default,omitempty"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	Minimum     *float64 `json:"minimum,omitempty"`
	Maximum     *float64 `json:"maximum,omitempty"`
}

// TrainerDescriptor registers an algorithm or training workflow.
type TrainerDescriptor struct {
	ID                     CapabilityID                   `json:"id"`
	Component              Component                      `json:"component"`
	DisplayName            string                         `json:"display_name"`
	SupportedTargets       []Target                       `json:"supported_targets"`
	SupportedArchitectures []CapabilityID                 `json:"supported_architectures,omitempty"`
	SupportedExecutors     []CapabilityID                 `json:"supported_executors"`
	SupportedHardware      []CapabilityID                 `json:"supported_hardware"`
	SupportedPrecisions    []CapabilityID                 `json:"supported_precisions"`
	ProducedFormats        []CapabilityID                 `json:"produced_formats"`
	Parameters             map[string]ParameterConstraint `json:"parameters,omitempty"`
}

// ArchitectureDriverDescriptor registers a model family and driver capabilities.
type ArchitectureDriverDescriptor struct {
	ID                 CapabilityID   `json:"id"`
	Family             string         `json:"family"`
	DisplayName        string         `json:"display_name"`
	SupportedTargets   []Target       `json:"supported_targets"`
	SupportedFormats   []CapabilityID `json:"supported_formats"`
	SupportedRuntimes  []CapabilityID `json:"supported_runtimes"`
	RequiredModelFiles []string       `json:"required_model_files,omitempty"`
}

// ExecutorDescriptor registers a worker runtime environment or task runner.
type ExecutorDescriptor struct {
	ID                CapabilityID   `json:"id"`
	Component         Component      `json:"component"`
	DisplayName       string         `json:"display_name"`
	SupportedHardware []CapabilityID `json:"supported_hardware"`
	MinMemoryBytes    int64          `json:"min_memory_bytes,omitempty"`
	IsolationLevel    string         `json:"isolation_level"` // "process", "container", "pod"
}

// ArtifactFormatDescriptor registers an artifact serialization representation.
type ArtifactFormatDescriptor struct {
	ID             CapabilityID   `json:"id"`
	Component      Component      `json:"component"`
	DisplayName    string         `json:"display_name"`
	FileExtensions []string       `json:"file_extensions"`
	DirectRuntimes []CapabilityID `json:"direct_runtimes"`
}

// RuntimeAdapterDescriptor registers an inference runtime for qualification.
type RuntimeAdapterDescriptor struct {
	ID                  CapabilityID   `json:"id"`
	Component           Component      `json:"component"`
	DisplayName         string         `json:"display_name"`
	SupportedTargets    []Target       `json:"supported_targets"`
	AcceptedFormats     []CapabilityID `json:"accepted_formats"`
	SupportedHardware   []CapabilityID `json:"supported_hardware"`
	SupportedPrecisions []CapabilityID `json:"supported_precisions"`
	Connector           string         `json:"connector"` // e.g. sr.model-runtime.openapi.v2
}

// PrecisionDescriptor registers numeric precision modes.
type PrecisionDescriptor struct {
	ID             CapabilityID `json:"id"`
	Name           string       `json:"name"`
	BitsPerElement int          `json:"bits_per_element"`
}

// HardwareProviderDescriptor registers compute accelerators and execution providers.
type HardwareProviderDescriptor struct {
	ID                  CapabilityID   `json:"id"`
	Provider            string         `json:"provider"`    // "cpu", "nvidia", "amd", "apple"
	DeviceType          string         `json:"device_type"` // "cpu", "gpu", "npu"
	DisplayName         string         `json:"display_name"`
	SupportedPrecisions []CapabilityID `json:"supported_precisions"`
}

// ConversionRule defines an automated path to convert from one format to another.
type ConversionRule struct {
	SourceFormat CapabilityID   `json:"source_format"`
	TargetFormat CapabilityID   `json:"target_format"`
	Executor     CapabilityID   `json:"executor"`
	Hardware     []CapabilityID `json:"hardware"`
}

// CapabilityCatalog is the complete snapshot of server capabilities returned by the API.
type CapabilityCatalog struct {
	SchemaVersion string                         `json:"schema_version" jsonschema:"enum=semantic-router.training/v2"`
	Targets       []TargetDescriptor             `json:"targets"`
	Trainers      []TrainerDescriptor            `json:"trainers"`
	Architectures []ArchitectureDriverDescriptor `json:"architectures"`
	Executors     []ExecutorDescriptor           `json:"executors"`
	Formats       []ArtifactFormatDescriptor     `json:"formats"`
	Runtimes      []RuntimeAdapterDescriptor     `json:"runtimes"`
	Precisions    []PrecisionDescriptor          `json:"precisions"`
	Hardware      []HardwareProviderDescriptor   `json:"hardware"`
	Conversions   []ConversionRule               `json:"conversions,omitempty"`
}

// CapabilityRegistry holds registered capability descriptors in memory.
// It is safely extensible by third-party drivers without editing central switch statements.
type CapabilityRegistry struct {
	mu            sync.RWMutex
	targets       map[CapabilityID]TargetDescriptor
	trainers      map[CapabilityID]TrainerDescriptor
	architectures map[CapabilityID]ArchitectureDriverDescriptor
	executors     map[CapabilityID]ExecutorDescriptor
	formats       map[CapabilityID]ArtifactFormatDescriptor
	runtimes      map[CapabilityID]RuntimeAdapterDescriptor
	precisions    map[CapabilityID]PrecisionDescriptor
	hardware      map[CapabilityID]HardwareProviderDescriptor
	conversions   []ConversionRule
}

func NewCapabilityRegistry() *CapabilityRegistry {
	return &CapabilityRegistry{
		targets:       make(map[CapabilityID]TargetDescriptor),
		trainers:      make(map[CapabilityID]TrainerDescriptor),
		architectures: make(map[CapabilityID]ArchitectureDriverDescriptor),
		executors:     make(map[CapabilityID]ExecutorDescriptor),
		formats:       make(map[CapabilityID]ArtifactFormatDescriptor),
		runtimes:      make(map[CapabilityID]RuntimeAdapterDescriptor),
		precisions:    make(map[CapabilityID]PrecisionDescriptor),
		hardware:      make(map[CapabilityID]HardwareProviderDescriptor),
	}
}

func (r *CapabilityRegistry) RegisterTarget(d TargetDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "target"); err != nil {
		return err
	}
	if err := ValidateTarget(d.TargetContract); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.targets[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterTrainer(d TrainerDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "trainer"); err != nil {
		return err
	}
	if err := ValidateComponent(d.Component); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.trainers[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterArchitecture(d ArchitectureDriverDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "architecture"); err != nil {
		return err
	}
	if d.Family == "" {
		return fmt.Errorf("architecture driver family is required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.architectures[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterExecutor(d ExecutorDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "executor"); err != nil {
		return err
	}
	if err := ValidateComponent(d.Component); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.executors[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterFormat(d ArtifactFormatDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "format"); err != nil {
		return err
	}
	if err := ValidateComponent(d.Component); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.formats[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterRuntime(d RuntimeAdapterDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "runtime"); err != nil {
		return err
	}
	if err := ValidateComponent(d.Component); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.runtimes[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterPrecision(d PrecisionDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "precision"); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.precisions[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterHardware(d HardwareProviderDescriptor) error {
	if err := ValidateCapabilityID(d.ID, "hardware"); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.hardware[d.ID] = d
	return nil
}

func (r *CapabilityRegistry) RegisterConversion(c ConversionRule) error {
	if err := ValidateCapabilityID(c.SourceFormat, "format"); err != nil {
		return err
	}
	if err := ValidateCapabilityID(c.TargetFormat, "format"); err != nil {
		return err
	}
	if err := ValidateCapabilityID(c.Executor, "executor"); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.conversions = append(r.conversions, c)
	return nil
}

func (r *CapabilityRegistry) GetTarget(id CapabilityID) (TargetDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.targets[id]
	return d, ok
}

func (r *CapabilityRegistry) GetTrainer(id CapabilityID) (TrainerDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.trainers[id]
	return d, ok
}

func (r *CapabilityRegistry) GetArchitecture(id CapabilityID) (ArchitectureDriverDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.architectures[id]
	return d, ok
}

func (r *CapabilityRegistry) GetExecutor(id CapabilityID) (ExecutorDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.executors[id]
	return d, ok
}

func (r *CapabilityRegistry) GetFormat(id CapabilityID) (ArtifactFormatDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.formats[id]
	return d, ok
}

func (r *CapabilityRegistry) GetRuntime(id CapabilityID) (RuntimeAdapterDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.runtimes[id]
	return d, ok
}

func (r *CapabilityRegistry) GetPrecision(id CapabilityID) (PrecisionDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.precisions[id]
	return d, ok
}

func (r *CapabilityRegistry) GetHardware(id CapabilityID) (HardwareProviderDescriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.hardware[id]
	return d, ok
}

func (r *CapabilityRegistry) FindConversion(source, target CapabilityID) (ConversionRule, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, c := range r.conversions {
		if c.SourceFormat == source && c.TargetFormat == target {
			return c, true
		}
	}
	return ConversionRule{}, false
}

func (r *CapabilityRegistry) Catalog() CapabilityCatalog {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return CapabilityCatalog{
		SchemaVersion: Version,
		Targets:       sortedByID(r.targets),
		Trainers:      sortedByID(r.trainers),
		Architectures: sortedByID(r.architectures),
		Executors:     sortedByID(r.executors),
		Formats:       sortedByID(r.formats),
		Runtimes:      sortedByID(r.runtimes),
		Precisions:    sortedByID(r.precisions),
		Hardware:      sortedByID(r.hardware),
		Conversions:   append([]ConversionRule(nil), r.conversions...),
	}
}

func sortedByID[T any](descriptors map[CapabilityID]T) []T {
	sorted := make([]T, 0, len(descriptors))
	for _, id := range slices.Sorted(maps.Keys(descriptors)) {
		sorted = append(sorted, descriptors[id])
	}
	return sorted
}

// DefaultRegistry returns the populated standard registry with built-in capabilities.
func DefaultRegistry() *CapabilityRegistry {
	r := NewCapabilityRegistry()

	// Targets
	_ = r.RegisterTarget(TargetDescriptor{
		ID:             "target/selector.model-choice@v1",
		TargetContract: Selector,
		DisplayName:    "Model Selector",
		Description:    "Query-level model selection based on quality, latency, and cost objectives",
	})
	_ = r.RegisterTarget(TargetDescriptor{
		ID:             "target/signal.label-scores@v1",
		TargetContract: LabelScores,
		DisplayName:    "Label Scores (Classifier)",
		Description:    "Classification signal with categorical label probability distribution",
	})
	_ = r.RegisterTarget(TargetDescriptor{
		ID:             "target/signal.spans@v1",
		TargetContract: Spans,
		DisplayName:    "Signal Spans (Token/Span Classifier)",
		Description:    "Entity and token-span level detection with unicode codepoint offsets",
	})

	// Precisions
	_ = r.RegisterPrecision(PrecisionDescriptor{ID: "precision/fp32@v1", Name: "float32", BitsPerElement: 32})
	_ = r.RegisterPrecision(PrecisionDescriptor{ID: "precision/fp16@v1", Name: "float16", BitsPerElement: 16})
	_ = r.RegisterPrecision(PrecisionDescriptor{ID: "precision/bf16@v1", Name: "bfloat16", BitsPerElement: 16})
	_ = r.RegisterPrecision(PrecisionDescriptor{ID: "precision/int8@v1", Name: "int8", BitsPerElement: 8})

	// Hardware
	_ = r.RegisterHardware(HardwareProviderDescriptor{
		ID:                  "hardware/cpu@v1",
		Provider:            "cpu",
		DeviceType:          "cpu",
		DisplayName:         "Host CPU",
		SupportedPrecisions: []CapabilityID{"precision/fp32@v1"},
	})
	_ = r.RegisterHardware(HardwareProviderDescriptor{
		ID:                  "hardware/cuda@v1",
		Provider:            "nvidia",
		DeviceType:          "gpu",
		DisplayName:         "NVIDIA CUDA GPU",
		SupportedPrecisions: []CapabilityID{"precision/fp32@v1", "precision/fp16@v1", "precision/bf16@v1", "precision/int8@v1"},
	})
	_ = r.RegisterHardware(HardwareProviderDescriptor{
		ID:                  "hardware/rocm@v1",
		Provider:            "amd",
		DeviceType:          "gpu",
		DisplayName:         "AMD ROCm GPU",
		SupportedPrecisions: []CapabilityID{"precision/fp32@v1", "precision/fp16@v1", "precision/bf16@v1"},
	})
	_ = r.RegisterHardware(HardwareProviderDescriptor{
		ID:                  "hardware/metal@v1",
		Provider:            "apple",
		DeviceType:          "gpu",
		DisplayName:         "Apple Silicon Metal GPU",
		SupportedPrecisions: []CapabilityID{"precision/fp32@v1", "precision/fp16@v1"},
	})

	// Formats
	_ = r.RegisterFormat(ArtifactFormatDescriptor{
		ID:             "format/selector-v2@v1",
		Component:      Component{Name: "selector-v2", Version: "1"},
		DisplayName:    "Selector Model Format v2",
		FileExtensions: []string{".json"},
		DirectRuntimes: []CapabilityID{"runtime/native@v1"},
	})
	_ = r.RegisterFormat(ArtifactFormatDescriptor{
		ID:             "format/safetensors@v1",
		Component:      Component{Name: "safetensors", Version: "1"},
		DisplayName:    "Hugging Face Safetensors",
		FileExtensions: []string{".safetensors", ".json"},
		DirectRuntimes: []CapabilityID{"runtime/model-runtime@v1"},
	})

	// Runtimes
	_ = r.RegisterRuntime(RuntimeAdapterDescriptor{
		ID:                  "runtime/native@v1",
		Component:           Component{Name: "native", Version: "1"},
		DisplayName:         "Native Router Runtime",
		SupportedTargets:    []Target{Selector},
		AcceptedFormats:     []CapabilityID{"format/selector-v2@v1"},
		SupportedHardware:   []CapabilityID{"hardware/cpu@v1"},
		SupportedPrecisions: []CapabilityID{"precision/fp32@v1"},
		Connector:           "sr.native.embedded.v1",
	})
	// The built-in model runtime (src/model-runtime) serves Router classifiers
	// over its OpenAPI 2.x contract: ModernBERT sequence and token checkpoints
	// loaded from safetensors, with FP32 weights and heads on every device.
	// Only the devices with readiness references qualify a classifier; the
	// runtime's CUDA and Metal paths are unvalidated (design section 11).
	_ = r.RegisterRuntime(RuntimeAdapterDescriptor{
		ID:                  "runtime/model-runtime@v1",
		Component:           Component{Name: "model-runtime", Version: "1"},
		DisplayName:         "vLLM Semantic Router Model Runtime",
		SupportedTargets:    []Target{LabelScores, Spans},
		AcceptedFormats:     []CapabilityID{"format/safetensors@v1"},
		SupportedHardware:   []CapabilityID{"hardware/cpu@v1", "hardware/rocm@v1"},
		SupportedPrecisions: []CapabilityID{"precision/fp32@v1"},
		Connector:           "sr.model-runtime.openapi.v2",
	})

	// Executors
	_ = r.RegisterExecutor(ExecutorDescriptor{
		ID:                "executor/train@v1",
		Component:         Component{Name: "train", Version: "1"},
		DisplayName:       "Standard Training Task Executor",
		SupportedHardware: []CapabilityID{"hardware/cpu@v1", "hardware/cuda@v1", "hardware/rocm@v1", "hardware/metal@v1"},
		IsolationLevel:    "container",
	})
	_ = r.RegisterExecutor(ExecutorDescriptor{
		ID:                "executor/evaluate@v1",
		Component:         Component{Name: "evaluate", Version: "1"},
		DisplayName:       "Standard Evaluation Task Executor",
		SupportedHardware: []CapabilityID{"hardware/cpu@v1", "hardware/cuda@v1"},
		IsolationLevel:    "container",
	})
	_ = r.RegisterExecutor(ExecutorDescriptor{
		ID:                "executor/qualify@v1",
		Component:         Component{Name: "qualify", Version: "1"},
		DisplayName:       "Standard Qualification Task Executor",
		SupportedHardware: []CapabilityID{"hardware/cpu@v1", "hardware/cuda@v1", "hardware/rocm@v1"},
		IsolationLevel:    "container",
	})

	// Architectures
	_ = r.RegisterArchitecture(ArchitectureDriverDescriptor{
		ID:                 "architecture/hf-modernbert@v1",
		Family:             "modernbert",
		DisplayName:        "Hugging Face ModernBERT",
		SupportedTargets:   []Target{LabelScores, Spans},
		SupportedFormats:   []CapabilityID{"format/safetensors@v1"},
		SupportedRuntimes:  []CapabilityID{"runtime/model-runtime@v1"},
		RequiredModelFiles: []string{"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"},
	})
	_ = r.RegisterArchitecture(ArchitectureDriverDescriptor{
		ID:                "architecture/selector-tabular@v1",
		Family:            "selector",
		DisplayName:       "Tabular Query-Outcome Selector",
		SupportedTargets:  []Target{Selector},
		SupportedFormats:  []CapabilityID{"format/selector-v2@v1"},
		SupportedRuntimes: []CapabilityID{"runtime/native@v1"},
	})

	// Trainers
	_ = r.RegisterTrainer(TrainerDescriptor{
		ID:                     "trainer/selector@v1",
		Component:              Component{Name: "selector", Version: "1"},
		DisplayName:            "Unified Model Selector Trainer",
		SupportedTargets:       []Target{Selector},
		SupportedArchitectures: []CapabilityID{"architecture/selector-tabular@v1"},
		SupportedExecutors:     []CapabilityID{"executor/train@v1"},
		SupportedHardware:      []CapabilityID{"hardware/cpu@v1"},
		SupportedPrecisions:    []CapabilityID{"precision/fp32@v1"},
		ProducedFormats:        []CapabilityID{"format/selector-v2@v1"},
		Parameters: map[string]ParameterConstraint{
			"seed":      {Type: "int", Required: false, Default: 42, Description: "Random seed for splitting and initialization"},
			"normalize": {Type: "bool", Required: false, Default: true, Description: "Normalize observation features"},
			"kernel":    {Type: "string", Required: false, Default: "rbf", Enum: []string{"linear", "rbf", "poly"}},
		},
	})
	_ = r.RegisterTrainer(TrainerDescriptor{
		ID:                     "trainer/selector-knn@v1",
		Component:              Component{Name: "selector-knn", Version: "1"},
		DisplayName:            "K-Nearest Neighbors Selector",
		SupportedTargets:       []Target{Selector},
		SupportedArchitectures: []CapabilityID{"architecture/selector-tabular@v1"},
		SupportedExecutors:     []CapabilityID{"executor/train@v1"},
		SupportedHardware:      []CapabilityID{"hardware/cpu@v1"},
		SupportedPrecisions:    []CapabilityID{"precision/fp32@v1"},
		ProducedFormats:        []CapabilityID{"format/selector-v2@v1"},
		Parameters: map[string]ParameterConstraint{
			"k":      {Type: "int", Required: false, Default: 5, Description: "Number of nearest neighbors"},
			"metric": {Type: "string", Required: false, Default: "cosine", Enum: []string{"cosine", "euclidean"}},
		},
	})
	_ = r.RegisterTrainer(TrainerDescriptor{
		ID:                     "trainer/selector-kmeans@v1",
		Component:              Component{Name: "selector-kmeans", Version: "1"},
		DisplayName:            "K-Means Selector",
		SupportedTargets:       []Target{Selector},
		SupportedArchitectures: []CapabilityID{"architecture/selector-tabular@v1"},
		SupportedExecutors:     []CapabilityID{"executor/train@v1"},
		SupportedHardware:      []CapabilityID{"hardware/cpu@v1"},
		SupportedPrecisions:    []CapabilityID{"precision/fp32@v1"},
		ProducedFormats:        []CapabilityID{"format/selector-v2@v1"},
		Parameters: map[string]ParameterConstraint{
			"k":        {Type: "int", Required: false, Default: 8, Description: "Requested cluster count"},
			"max_iter": {Type: "int", Required: false, Default: 300, Description: "Maximum Lloyd iterations"},
		},
	})
	_ = r.RegisterTrainer(TrainerDescriptor{
		ID:                     "trainer/neural@v1",
		Component:              Component{Name: "neural", Version: "1"},
		DisplayName:            "Neural Classifier Trainer",
		SupportedTargets:       []Target{LabelScores, Spans},
		SupportedArchitectures: []CapabilityID{"architecture/hf-modernbert@v1"},
		SupportedExecutors:     []CapabilityID{"executor/train@v1"},
		SupportedHardware:      []CapabilityID{"hardware/cuda@v1", "hardware/rocm@v1", "hardware/metal@v1", "hardware/cpu@v1"},
		SupportedPrecisions:    []CapabilityID{"precision/fp32@v1", "precision/fp16@v1", "precision/bf16@v1"},
		ProducedFormats:        []CapabilityID{"format/safetensors@v1"},
		Parameters: map[string]ParameterConstraint{
			"learning_rate": {Type: "float", Required: false, Default: 0.001, Description: "Optimizer learning rate"},
			"seed":          {Type: "int", Required: false, Default: 42, Description: "Random seed for reproducible training"},
		},
	})
	_ = r.RegisterTrainer(TrainerDescriptor{
		ID:                     "trainer/hf-peft@v1",
		Component:              Component{Name: "hf-peft", Version: "1"},
		DisplayName:            "Hugging Face Parameter-Efficient Fine-Tuning (LoRA)",
		SupportedTargets:       []Target{LabelScores, Spans},
		SupportedArchitectures: []CapabilityID{"architecture/hf-modernbert@v1"},
		SupportedExecutors:     []CapabilityID{"executor/train@v1"},
		SupportedHardware:      []CapabilityID{"hardware/cuda@v1", "hardware/rocm@v1"},
		SupportedPrecisions:    []CapabilityID{"precision/fp16@v1", "precision/bf16@v1"},
		ProducedFormats:        []CapabilityID{"format/safetensors@v1"},
		Parameters: map[string]ParameterConstraint{
			"r":             {Type: "int", Required: false, Default: 8, Description: "LoRA attention dimension"},
			"lora_alpha":    {Type: "int", Required: false, Default: 16, Description: "LoRA scaling alpha"},
			"learning_rate": {Type: "float", Required: false, Default: 0.0002, Description: "Learning rate"},
		},
	})

	return r
}
