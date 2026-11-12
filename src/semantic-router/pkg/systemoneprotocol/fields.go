package systemoneprotocol

import "slices"

// Level is where a field sits in a message.
type Level string

const (
	LevelRequest      Level = "request"
	LevelQuestion     Level = "question"
	LevelQuestionType Level = "question_type"
	LevelAsset        Level = "asset"
	LevelAssetType    Level = "asset_type"
	LevelAnswer       Level = "answer"
	LevelResponse     Level = "response"
	LevelError        Level = "error"
	LevelListingEntry Level = "listing_entry"
	LevelEndpoint     Level = "endpoint"
)

// StandardTarget is what a standard (systemone/0.2) target does with an
// extension field.
type StandardTarget string

const (
	// TargetStrip sends the message without the field: its standard part
	// carries the meaning.
	TargetStrip StandardTarget = "strip"
	// TargetRefuse fails the rule that needs the field at load.
	TargetRefuse StandardTarget = "refuse"
	// TargetAbsent marks a field a standard server never sends.
	TargetAbsent StandardTarget = "absent"
)

// Field is one field a schema defines: by the standard (Origin
// systemone/0.2) or by the vllm-sr extension set (Origin vllm-sr/5.0).
type Field struct {
	Level  Level
	Name   string
	Origin string
	// Since is the extension set version the field appeared in.
	Since string
	// Families are the model families that honour the field; nil is every
	// family.
	Families []string
	// AppliesTo are the question types a question or answer field goes with;
	// nil is every type.
	AppliesTo []string
	// Endpoints limit the field to these endpoints; nil is every endpoint.
	Endpoints []string
	// StandardTarget is what a standard target does with an extension field.
	StandardTarget StandardTarget
	// Schema is the JSON Schema of the field's value.
	Schema  string
	Summary string
}

// RemovedField is a 4.0 form that vllm-sr/5.0 no longer accepts.
type RemovedField struct {
	Level      string
	Name       string
	ReplacedBy string
	Note       string
}

// Fields are every field of the standard and of the extension set, in the
// specs' order.
func Fields() []Field { return slices.Clone(generatedFields) }

// RemovedFields are the 4.0 forms the extension set replaced.
func RemovedFields() []RemovedField { return slices.Clone(generatedRemoved) }

// Families are the model families the extension set describes.
func Families() []string { return slices.Clone(generatedFamilies) }

// ExtensionField is the extension field at a level.
func ExtensionField(level Level, name string) (Field, bool) {
	for _, field := range generatedFields {
		if field.Level == level && field.Name == name && field.Origin == ExtensionSchemaID {
			return field, true
		}
	}
	return Field{}, false
}

// IsStandardField reports whether the standard defines a field at a level.
// A question type is a field of LevelQuestionType; a question field is
// standard when any standard question type has it.
func IsStandardField(level Level, name string) bool {
	for _, field := range generatedFields {
		if field.Level == level && field.Name == name && field.Origin == StandardSchemaID {
			return true
		}
	}
	return false
}

// AppliesToQuestionType reports whether a question or answer field goes with
// a question type.
func (f Field) AppliesToQuestionType(questionType string) bool {
	return len(f.AppliesTo) == 0 || slices.Contains(f.AppliesTo, questionType)
}

// HonouredBy reports whether a model family honours the field.
func (f Field) HonouredBy(family string) bool {
	return f.Families == nil || slices.Contains(f.Families, family)
}

// IsField reports whether the standard or the extension set defines a field
// at a level.
func IsField(level Level, name string) bool {
	for _, field := range generatedFields {
		if field.Level == level && field.Name == name {
			return true
		}
	}
	return false
}

// Removed is the 4.0 form a name was at a level ("request", "question",
// "options", ...), which vllm-sr/5.0 replaced.
func Removed(level, name string) (RemovedField, bool) {
	for _, removed := range generatedRemoved {
		if removed.Level == level && removed.Name == name {
			return removed, true
		}
	}
	return RemovedField{}, false
}
