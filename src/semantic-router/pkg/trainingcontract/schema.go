package trainingcontract

import "github.com/invopop/jsonschema"

//go:generate go run ../../../../tools/codegen/trainingcontract/main.go --root ../../../..

func (Target) JSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{Type: "string", Enum: []any{Selector, LabelScores, Spans}}
}

func (Status) JSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{Type: "string", Enum: []any{Pending, Running, Cancelling, Succeeded, Failed, Cancelled, Skipped}}
}

func (Profile) JSONSchemaExtend(schema *jsonschema.Schema) {
	for _, branch := range []struct {
		target Target
		field  string
	}{{Selector, "selector"}, {LabelScores, "classifier"}, {Spans, "spans"}} {
		constraint := &jsonschema.Schema{Required: []string{branch.field}, Properties: jsonschema.NewProperties()}
		constraint.Properties.Set("target_contract", &jsonschema.Schema{Const: branch.target})
		for _, field := range []string{"selector", "classifier", "spans"} {
			if field != branch.field {
				constraint.Properties.Set(field, jsonschema.FalseSchema)
			}
		}
		schema.OneOf = append(schema.OneOf, constraint)
	}
}

// Catalog makes every public request, response and worker message reachable by generators.
type Catalog struct {
	AssetRequest       DataAssetSpec       `json:"asset_request"`
	SnapshotRequest    SnapshotSpec        `json:"snapshot_request"`
	ExperimentRequest  ExperimentSpec      `json:"experiment_request"`
	ProposalRequest    BindingProposalSpec `json:"proposal_request"`
	Fixture            Fixture             `json:"fixture"`
	Asset              DataAsset           `json:"asset"`
	Snapshot           DataSnapshot        `json:"snapshot"`
	Upload             Upload              `json:"upload"`
	Experiment         Experiment          `json:"experiment"`
	Graph              RunGraph            `json:"graph"`
	Artifact           Artifact            `json:"artifact"`
	Variant            ArtifactVariant     `json:"variant"`
	Evaluation         Evaluation          `json:"evaluation"`
	Qualification      Qualification       `json:"qualification"`
	Proposal           BindingProposal     `json:"proposal"`
	Event              Event               `json:"event"`
	WorkerRequest      WorkerRequest       `json:"worker_request"`
	ComparisonRequest  ComparisonRequest   `json:"comparison_request"`
	ComparisonResponse ComparisonResponse  `json:"comparison_response"`
	ValidationRequest  ValidationRequest   `json:"validation_request"`
	ValidationResponse ValidationResponse  `json:"validation_response"`
	Error              APIError            `json:"error"`
	EventPage          EventPage           `json:"event_page"`
	WorkerSubmission   WorkerSubmission    `json:"worker_submission"`
}

// Workers report execution outcomes, not management-only pending/retry states.
func (WorkerResult) JSONSchemaExtend(schema *jsonschema.Schema) {
	status, _ := schema.Properties.Get("status")
	*status = jsonschema.Schema{Type: "string", Enum: []any{Running, Succeeded, Failed, Cancelled}}
	condition := jsonschema.NewProperties()
	condition.Set("status", &jsonschema.Schema{Const: Succeeded})
	schema.If = &jsonschema.Schema{Properties: condition}
	outputs := jsonschema.NewProperties()
	for _, field := range []string{"artifacts", "evaluations", "qualifications"} {
		zero := uint64(0)
		outputs.Set(field, &jsonschema.Schema{MaxItems: &zero})
	}
	schema.Else = &jsonschema.Schema{Properties: outputs}
}

// Skipped is a dependency outcome for tasks, never a run outcome.
func (TrainingRun) JSONSchemaExtend(schema *jsonschema.Schema) {
	status, _ := schema.Properties.Get("status")
	*status = jsonschema.Schema{Type: "string", Enum: []any{Pending, Running, Cancelling, Succeeded, Failed, Cancelled}}
}

func (ComparisonRequest) JSONSchemaExtend(schema *jsonschema.Schema) {
	ids, _ := schema.Properties.Get("run_ids")
	ids.Items.Pattern = handlePattern.String()
}

func (SelectorProfile) JSONSchemaExtend(schema *jsonschema.Schema) {
	one := uint64(1)
	for _, name := range []string{"candidate_models", "observation_fields"} {
		field, _ := schema.Properties.Get(name)
		field.Items.MinLength = &one
	}
}

func (SpanProfile) JSONSchemaExtend(schema *jsonschema.Schema) {
	one := uint64(1)
	labels, _ := schema.Properties.Get("labels")
	labels.Items.MinLength = &one
}

func (RunOutputs) JSONSchemaExtend(schema *jsonschema.Schema) {
	for _, name := range []string{"artifact_ids", "evaluation_ids", "qualification_ids"} {
		field, _ := schema.Properties.Get(name)
		field.Items.Pattern = handlePattern.String()
	}
}

func (ArtifactVariantSpec) JSONSchemaExtend(schema *jsonschema.Schema) {
	one := uint64(1)
	files, _ := schema.Properties.Get("files")
	files.MinProperties = &one
	files.PropertyNames = &jsonschema.Schema{
		MinLength: &one,
		Not:       &jsonschema.Schema{Pattern: invalidArtifactName.String()},
	}
}
