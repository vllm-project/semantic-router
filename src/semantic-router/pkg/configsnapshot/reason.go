package configsnapshot

import (
	"errors"
	"strings"
)

// Stage is a step of the lifecycle every update goes through.
type Stage string

const (
	// StageParse reads the source document into the configuration model.
	StageParse Stage = "parse"
	// StageCompile builds the typed resources and resolves their references.
	StageCompile Stage = "compile"
	// StageValidate checks the candidate against the running Router without
	// building anything: hot-reload compatibility, artifacts and capabilities.
	StageValidate Stage = "validate"
	// StageWarm builds and warms what the candidate serves with: models,
	// classifiers, connection pools and precomputed state.
	StageWarm Stage = "warm"
	// StageActivate swaps the candidate in.
	StageActivate Stage = "activate"
)

// Code classifies why a stage rejected an update.
type Code string

const (
	// CodeInvalidDocument: the document does not parse or fails the schema.
	CodeInvalidDocument Code = "invalid_document"
	// CodeDuplicateName: two resources of one kind share a name.
	CodeDuplicateName Code = "duplicate_name"
	// CodeUnresolvedReference: a resource names one that does not exist.
	CodeUnresolvedReference Code = "unresolved_reference"
	// CodeInvalidResource: a resource is incomplete or malformed.
	CodeInvalidResource Code = "invalid_resource"
	// CodeRestartRequired: the change cannot be applied without a restart.
	CodeRestartRequired Code = "restart_required"
	// CodeArtifactUnavailable: a model artifact the candidate needs is missing.
	CodeArtifactUnavailable Code = "artifact_unavailable"
	// CodeUnsupported: the candidate needs something the running Router lacks.
	CodeUnsupported Code = "unsupported"
	// CodeModelUnavailable: a model could not be downloaded or prepared.
	CodeModelUnavailable Code = "model_unavailable"
	// CodeBuildFailed: building the candidate's runtime failed.
	CodeBuildFailed Code = "build_failed"
	// CodeWarmupFailed: warming the candidate's runtime failed.
	CodeWarmupFailed Code = "warmup_failed"
	// CodeActivationFailed: the warmed candidate could not be swapped in.
	CodeActivationFailed Code = "activation_failed"
	// CodeShuttingDown: the Router began shutting down.
	CodeShuttingDown Code = "shutting_down"
	// CodeCanceled: the caller gave up before the update finished.
	CodeCanceled Code = "canceled"
)

// Reason is one structured cause of a rejected update.
type Reason struct {
	Stage Stage `json:"stage"`
	Code  Code  `json:"code"`
	// Path locates the offending part of the canonical document, when known.
	Path    string `json:"path,omitempty"`
	Message string `json:"message"`
}

// Rejection is the error of a rejected (NACKed) update. Its message is the
// message of the error that caused it, so callers that report the cause read
// the same text as before the lifecycle classified it.
type Rejection struct {
	Stage   Stage
	Reasons []Reason
	cause   error
	// recorded is set once a Manager has recorded the rejection as an
	// attempt's result.
	recorded bool
}

// Reject classifies err as the rejection of an update at stage.
func Reject(stage Stage, code Code, err error) *Rejection {
	if err == nil {
		return nil
	}
	var rejection *Rejection
	if errors.As(err, &rejection) {
		return rejection
	}
	return &Rejection{
		Stage:   stage,
		Reasons: []Reason{{Stage: stage, Code: code, Message: err.Error()}},
		cause:   err,
	}
}

// RejectReasons is the rejection of reasons found at one stage, or nil when
// there are none.
func RejectReasons(stage Stage, reasons []Reason) *Rejection {
	if len(reasons) == 0 {
		return nil
	}
	return &Rejection{Stage: stage, Reasons: reasons}
}

func (r *Rejection) Error() string {
	if r.cause != nil {
		return r.cause.Error()
	}
	messages := make([]string, 0, len(r.Reasons))
	for _, reason := range r.Reasons {
		message := reason.Message
		if reason.Path != "" {
			message = reason.Path + ": " + message
		}
		messages = append(messages, message)
	}
	return strings.Join(messages, "; ")
}

func (r *Rejection) Unwrap() error { return r.cause }

// ReasonsOf returns the structured reasons of a rejection, or nil when err is
// not one.
func ReasonsOf(err error) []Reason {
	var rejection *Rejection
	if !errors.As(err, &rejection) {
		return nil
	}
	return append([]Reason(nil), rejection.Reasons...)
}
