/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package handoff defines the portable, protocol-neutral handoff envelope that
// an external agent runtime attaches when it moves delegated work across a
// Router-selected model boundary. Transport encodings such as HTTP headers
// belong to the protocol adapter, not this package.
package handoff

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	Version            = "1"
	MaxJSONBytes       = 4 * 1024
	MaxIDLength        = 128
	MaxNameLength      = 64
	MaxSummaryBytes    = 1024
	MaxListItems       = 16
	MaxRemainingTokens = 10_000_000
	MaxLifetime        = 15 * time.Minute
)

var (
	opaqueIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/@-]*$`)
	namePattern     = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
)

// State is the lifecycle state the external runtime asserts for a handoff.
type State string

const (
	StateActive    State = "active"
	StateCancelled State = "cancelled"
)

// Envelope is the version-1 handoff contract. Identity, lifecycle, and expiry
// live at the top level. Selection holds the only Router-readable facts;
// Runtime holds external-runtime state the Router never interprets.
type Envelope struct {
	Version            string     `json:"version"`
	HandoffID          string     `json:"handoff_id"`
	RootInvocationID   string     `json:"root_invocation_id"`
	ParentInvocationID string     `json:"parent_invocation_id,omitempty"`
	State              State      `json:"state,omitempty"`
	ExpiresAt          time.Time  `json:"expires_at"`
	Selection          *Selection `json:"selection,omitempty"`
	Runtime            *Runtime   `json:"runtime,omitempty"`
}

// Selection carries Router-readable selection facts. Version 1 validates them
// but does not consume them; projection into model selection belongs to the
// lineage contract in #3379.
type Selection struct {
	DelegatedRole        string   `json:"delegated_role,omitempty"`
	RequiredCapabilities []string `json:"required_capabilities,omitempty"`
	RemainingTokens      *int     `json:"remaining_tokens,omitempty"`
}

// Runtime carries opaque external-runtime state. The Router bounds it but never
// dereferences, logs, persists, or forwards it.
type Runtime struct {
	TaskSummary   string   `json:"task_summary,omitempty"`
	ResultSummary string   `json:"result_summary,omitempty"`
	ToolStateRefs []string `json:"tool_state_refs,omitempty"`
}

// String keeps accidental formatting of an envelope from leaking its contents.
func (e *Envelope) String() string {
	if e == nil {
		return "handoff(<nil>)"
	}
	return fmt.Sprintf("handoff(version=%s id=%s)", e.Version, e.HandoffID)
}

// Digest identifies the normalized envelope contents for idempotency checks.
func (e *Envelope) Digest() [sha256.Size]byte {
	encoded, _ := json.Marshal(e)
	return sha256.Sum256(encoded)
}

// ErrorCode is safe to expose as a machine-readable receipt reason. Errors
// never include envelope values.
type ErrorCode string

const (
	CodePayloadTooLarge    ErrorCode = "payload_too_large"
	CodeMalformedJSON      ErrorCode = "malformed_json"
	CodeUnsupportedVersion ErrorCode = "unsupported_version"
	CodeUnknownField       ErrorCode = "unknown_field"
	CodeMissingField       ErrorCode = "missing_field"
	CodeInvalidField       ErrorCode = "invalid_field"
	CodeExpired            ErrorCode = "expired"
)

// ContractError is a value-safe validation error.
type ContractError struct {
	Code  ErrorCode
	Field string
}

func (e *ContractError) Error() string {
	if e.Field == "" {
		return string(e.Code)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Field)
}

// CodeOf extracts a stable contract error code.
func CodeOf(err error) ErrorCode {
	var contractErr *ContractError
	if errors.As(err, &contractErr) {
		return contractErr.Code
	}
	return CodeMalformedJSON
}

func contractError(code ErrorCode, field string) error {
	return &ContractError{Code: code, Field: field}
}

// Parse decodes compact JSON and validates the envelope at now. A different
// version is reported before any field checks so a newer envelope degrades to
// unsupported_version rather than unknown_field.
func Parse(data []byte, now time.Time) (*Envelope, error) {
	if len(data) > MaxJSONBytes {
		return nil, contractError(CodePayloadTooLarge, "")
	}
	if err := checkStructure(data); err != nil {
		return nil, err
	}
	var header struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, contractError(CodeMalformedJSON, "")
	}
	if header.Version == "" {
		return nil, contractError(CodeMissingField, "version")
	}
	if header.Version != Version {
		return nil, contractError(CodeUnsupportedVersion, "version")
	}

	var envelope Envelope
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&envelope); err != nil {
		var timeErr *time.ParseError
		switch {
		case strings.HasPrefix(err.Error(), "json: unknown field"):
			return nil, contractError(CodeUnknownField, "")
		case errors.As(err, &timeErr):
			return nil, contractError(CodeInvalidField, "expires_at")
		}
		return nil, contractError(CodeMalformedJSON, "")
	}
	if envelope.State == "" {
		envelope.State = StateActive
	}
	envelope.ExpiresAt = envelope.ExpiresAt.UTC()
	if err := envelope.validate(now); err != nil {
		return nil, err
	}
	return &envelope, nil
}

// checkStructure rejects duplicate object keys and trailing values, which the
// standard decoder would otherwise resolve silently.
func checkStructure(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := checkValue(decoder); err != nil {
		return err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return contractError(CodeMalformedJSON, "")
	}
	return nil
}

func checkValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return contractError(CodeMalformedJSON, "")
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	seen := map[string]struct{}{}
	for decoder.More() {
		if delimiter == '{' {
			keyToken, err := decoder.Token()
			key, ok := keyToken.(string)
			if err != nil || !ok {
				return contractError(CodeMalformedJSON, "")
			}
			if _, duplicate := seen[key]; duplicate {
				return contractError(CodeMalformedJSON, key)
			}
			seen[key] = struct{}{}
		}
		if err := checkValue(decoder); err != nil {
			return err
		}
	}
	if _, err := decoder.Token(); err != nil {
		return contractError(CodeMalformedJSON, "")
	}
	return nil
}

func (e *Envelope) validate(now time.Time) error {
	if err := requireOpaqueID("handoff_id", e.HandoffID); err != nil {
		return err
	}
	if err := requireOpaqueID("root_invocation_id", e.RootInvocationID); err != nil {
		return err
	}
	if e.ParentInvocationID != "" {
		if err := requireOpaqueID("parent_invocation_id", e.ParentInvocationID); err != nil {
			return err
		}
	}
	if e.State != StateActive && e.State != StateCancelled {
		return contractError(CodeInvalidField, "state")
	}
	if err := e.Selection.validate(); err != nil {
		return err
	}
	if err := e.Runtime.validate(); err != nil {
		return err
	}
	return validateExpiry(e.ExpiresAt, now)
}

func (s *Selection) validate() error {
	if s == nil {
		return nil
	}
	if s.DelegatedRole != "" && !validName(s.DelegatedRole) {
		return contractError(CodeInvalidField, "selection.delegated_role")
	}
	if err := validateList("selection.required_capabilities", s.RequiredCapabilities, validName); err != nil {
		return err
	}
	if s.RemainingTokens != nil && (*s.RemainingTokens < 0 || *s.RemainingTokens > MaxRemainingTokens) {
		return contractError(CodeInvalidField, "selection.remaining_tokens")
	}
	return nil
}

func (r *Runtime) validate() error {
	if r == nil {
		return nil
	}
	if !validSummary(r.TaskSummary) {
		return contractError(CodeInvalidField, "runtime.task_summary")
	}
	if !validSummary(r.ResultSummary) {
		return contractError(CodeInvalidField, "runtime.result_summary")
	}
	return validateList("runtime.tool_state_refs", r.ToolStateRefs, validOpaqueID)
}

func validateExpiry(expiresAt, now time.Time) error {
	if expiresAt.IsZero() {
		return contractError(CodeMissingField, "expires_at")
	}
	if !now.Before(expiresAt) {
		return contractError(CodeExpired, "expires_at")
	}
	if expiresAt.After(now.Add(MaxLifetime)) {
		return contractError(CodeInvalidField, "expires_at")
	}
	return nil
}

func requireOpaqueID(field, value string) error {
	if value == "" {
		return contractError(CodeMissingField, field)
	}
	if !validOpaqueID(value) {
		return contractError(CodeInvalidField, field)
	}
	return nil
}

func validOpaqueID(value string) bool {
	return len(value) <= MaxIDLength && opaqueIDPattern.MatchString(value)
}

func validName(value string) bool {
	return len(value) <= MaxNameLength && namePattern.MatchString(value)
}

func validSummary(value string) bool {
	return len(value) <= MaxSummaryBytes && utf8.ValidString(value)
}

func validateList(field string, values []string, valid func(string) bool) error {
	if len(values) > MaxListItems {
		return contractError(CodeInvalidField, field)
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, duplicate := seen[value]; duplicate || !valid(value) {
			return contractError(CodeInvalidField, field)
		}
		seen[value] = struct{}{}
	}
	return nil
}
