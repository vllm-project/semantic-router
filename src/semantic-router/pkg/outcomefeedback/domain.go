// Package outcomefeedback owns authenticated, replay-bound inference outcome
// ingestion. It deliberately has no dependency on Dashboard or Management
// identities: callers are the logical inference identities already verified by
// the Router.
package outcomefeedback

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	MaximumBodyBytes       = 64 << 10
	MaximumIdempotencySize = 256
	MaximumReasonSize      = 2048
	MaximumMetadataEntries = 32
	MaximumMetadataKeySize = 128
	MaximumMetadataValue   = 1024
	MaximumReplayIDSize    = 256
	MaximumTargetRefSize   = 512
)

var (
	ErrInvalid             = errors.New("invalid inference outcome")
	ErrNotFound            = errors.New("inference replay not found")
	ErrIdempotencyConflict = errors.New("outcome idempotency conflict")
	ErrRateLimited         = errors.New("outcome feedback rate limited")
	ErrUnavailable         = errors.New("outcome feedback unavailable")

	canonicalIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$`)
	metadataKeyPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
)

type Source string

const (
	SourceAPIKey    Source = "api_key"
	SourceDelegated Source = "delegated_inference_session"
)

func (source Source) Valid() bool {
	return source == SourceAPIKey || source == SourceDelegated
}

type Target string

const (
	TargetModel     Target = "model"
	TargetRoute     Target = "route"
	TargetPolicy    Target = "policy"
	TargetStability Target = "stability"
	TargetProvider  Target = "provider"
	TargetRouter    Target = "router"
)

func (target Target) Valid() bool {
	switch target {
	case TargetModel, TargetRoute, TargetPolicy, TargetStability, TargetProvider, TargetRouter:
		return true
	default:
		return false
	}
}

type Verdict string

const (
	VerdictGoodFit         Verdict = "good_fit"
	VerdictUnderpowered    Verdict = "underpowered"
	VerdictOverprovisioned Verdict = "overprovisioned"
	VerdictFailed          Verdict = "failed"
)

func (verdict Verdict) Valid() bool {
	switch verdict {
	case VerdictGoodFit, VerdictUnderpowered, VerdictOverprovisioned, VerdictFailed:
		return true
	default:
		return false
	}
}

// Caller is derived exclusively from a successfully authenticated inference
// session. None of these fields are accepted in the public JSON body.
type Caller struct {
	NamespaceID string
	APIKeyID    string
	UserID      string
	TeamID      string
	Source      Source
}

func (caller Caller) Validate() error {
	if !canonicalIdentifier(caller.NamespaceID, MaximumReplayIDSize) ||
		!canonicalIdentifier(caller.APIKeyID, MaximumReplayIDSize) || !caller.Source.Valid() {
		return fmt.Errorf("%w: authenticated namespace, logical key, and source are required", ErrInvalid)
	}
	if caller.UserID != "" && !canonicalIdentifier(caller.UserID, MaximumReplayIDSize) {
		return fmt.Errorf("%w: authenticated user is invalid", ErrInvalid)
	}
	if caller.TeamID != "" && !canonicalIdentifier(caller.TeamID, MaximumReplayIDSize) {
		return fmt.Errorf("%w: authenticated team is invalid", ErrInvalid)
	}
	return nil
}

// Request is the complete bounded public outcome payload after strict JSON
// decoding. Model outcomes must identify both the Model and immutable revision
// that served the replay.
type Request struct {
	ReplayID       string            `json:"replay_id"`
	Target         Target            `json:"target"`
	TargetRef      string            `json:"target_ref,omitempty"`
	TargetRevision *int64            `json:"target_revision,omitempty"`
	Verdict        Verdict           `json:"verdict"`
	Reason         string            `json:"reason,omitempty"`
	Score          *float64          `json:"score,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

func (request Request) Validate() error {
	if !canonicalIdentifier(request.ReplayID, MaximumReplayIDSize) {
		return fmt.Errorf("%w: replay_id is required and must be canonical", ErrInvalid)
	}
	if !request.Target.Valid() {
		return fmt.Errorf("%w: target is invalid", ErrInvalid)
	}
	if !request.Verdict.Valid() {
		return fmt.Errorf("%w: verdict is invalid", ErrInvalid)
	}
	if request.TargetRef != strings.TrimSpace(request.TargetRef) || len(request.TargetRef) > MaximumTargetRefSize || strings.ContainsRune(request.TargetRef, '\x00') {
		return fmt.Errorf("%w: target_ref is not canonical", ErrInvalid)
	}
	if request.Target == TargetModel {
		if request.TargetRef == "" || request.TargetRevision == nil || *request.TargetRevision <= 0 {
			return fmt.Errorf("%w: model outcomes require target_ref and a positive target_revision", ErrInvalid)
		}
	} else if request.TargetRevision != nil {
		return fmt.Errorf("%w: target_revision is valid only for model outcomes", ErrInvalid)
	}
	if request.Reason != strings.TrimSpace(request.Reason) || len(request.Reason) > MaximumReasonSize || strings.ContainsRune(request.Reason, '\x00') {
		return fmt.Errorf("%w: reason is not canonical", ErrInvalid)
	}
	if request.Score != nil && (math.IsNaN(*request.Score) || math.IsInf(*request.Score, 0) || *request.Score < 0 || *request.Score > 1) {
		return fmt.Errorf("%w: score must be between zero and one", ErrInvalid)
	}
	if len(request.Metadata) > MaximumMetadataEntries {
		return fmt.Errorf("%w: metadata contains too many entries", ErrInvalid)
	}
	for key, value := range request.Metadata {
		if len(key) > MaximumMetadataKeySize || !metadataKeyPattern.MatchString(key) ||
			value != strings.TrimSpace(value) || len(value) > MaximumMetadataValue || strings.ContainsRune(value, '\x00') {
			return fmt.Errorf("%w: metadata is not canonical", ErrInvalid)
		}
	}
	return nil
}

func DecodeRequest(payload []byte) (Request, error) {
	if len(payload) == 0 || len(payload) > MaximumBodyBytes || bytes.IndexByte(payload, 0) >= 0 {
		return Request{}, fmt.Errorf("%w: body must be non-empty, NUL-free, and at most %d bytes", ErrInvalid, MaximumBodyBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var request Request
	if err := decoder.Decode(&request); err != nil {
		return Request{}, fmt.Errorf("%w: malformed body", ErrInvalid)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return Request{}, fmt.Errorf("%w: body has trailing data", ErrInvalid)
	}
	if err := request.Validate(); err != nil {
		return Request{}, err
	}
	return request, nil
}

func ValidateIdempotencyKey(value string) error {
	if value == "" || len(value) > MaximumIdempotencySize || value != strings.TrimSpace(value) || strings.ContainsRune(value, '\x00') {
		return fmt.Errorf("%w: Idempotency-Key must be between 1 and %d canonical bytes", ErrInvalid, MaximumIdempotencySize)
	}
	for _, character := range []byte(value) {
		if character < 0x21 || character > 0x7e {
			return fmt.Errorf("%w: Idempotency-Key contains unsupported characters", ErrInvalid)
		}
	}
	return nil
}

type Receipt struct {
	ID                 string    `json:"id"`
	ReplayID           string    `json:"replay_id"`
	ProjectionRevision int64     `json:"projection_revision"`
	Duplicate          bool      `json:"duplicate,omitempty"`
	CreatedAt          time.Time `json:"created_at"`
}

type ServedModel struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Revision int64  `json:"revision"`
}

type ReplayRoutingContext struct {
	RecipeID       string `json:"recipe_id,omitempty"`
	RecipeName     string `json:"recipe_name,omitempty"`
	RecipeRevision int64  `json:"recipe_revision,omitempty"`
	DecisionID     string `json:"decision_id,omitempty"`
	DecisionName   string `json:"decision_name,omitempty"`
	DecisionTier   int    `json:"decision_tier,omitempty"`
}

type ReplayRecord struct {
	NamespaceID string
	ReplayID    string
	APIKeyID    string
	UserID      string
	TeamID      string
	Routing     ReplayRoutingContext
	Models      []ServedModel
	CreatedAt   time.Time
}

func (record ReplayRecord) Owns(caller Caller) bool {
	return record.NamespaceID == caller.NamespaceID && record.APIKeyID == caller.APIKeyID
}

func (record ReplayRecord) Served(ref string, revision int64) bool {
	for _, model := range record.Models {
		if model.Revision == revision && (model.ID == ref || model.Name == ref) {
			return true
		}
	}
	return false
}

func IdempotencyDigest(caller Caller, replayID, key string) [sha256.Size]byte {
	return digestFields("vllm-sr/outcome-idempotency/v1", caller.NamespaceID, caller.APIKeyID, replayID, key)
}

func RequestDigest(request Request) ([sha256.Size]byte, error) {
	metadataKeys := make([]string, 0, len(request.Metadata))
	for key := range request.Metadata {
		metadataKeys = append(metadataKeys, key)
	}
	sort.Strings(metadataKeys)
	metadata := make([][2]string, 0, len(metadataKeys))
	for _, key := range metadataKeys {
		metadata = append(metadata, [2]string{key, request.Metadata[key]})
	}
	targetRevision := int64(0)
	if request.TargetRevision != nil {
		targetRevision = *request.TargetRevision
	}
	score := ""
	if request.Score != nil {
		score = strconv.FormatFloat(*request.Score, 'g', -1, 64)
	}
	payload := struct {
		ReplayID       string      `json:"replay_id"`
		Target         Target      `json:"target"`
		TargetRef      string      `json:"target_ref"`
		TargetRevision int64       `json:"target_revision"`
		Verdict        Verdict     `json:"verdict"`
		Reason         string      `json:"reason"`
		Score          string      `json:"score"`
		Metadata       [][2]string `json:"metadata"`
	}{request.ReplayID, request.Target, request.TargetRef, targetRevision, request.Verdict, request.Reason, score, metadata}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return [sha256.Size]byte{}, fmt.Errorf("encode outcome digest: %w", err)
	}
	return sha256.Sum256(encoded), nil
}

func DigestHex(value [sha256.Size]byte) string { return hex.EncodeToString(value[:]) }

func digestFields(domain string, fields ...string) [sha256.Size]byte {
	var payload bytes.Buffer
	payload.WriteString(domain)
	payload.WriteByte(0)
	for _, field := range fields {
		payload.WriteString(strconv.Itoa(len(field)))
		payload.WriteByte(':')
		payload.WriteString(field)
		payload.WriteByte(0)
	}
	return sha256.Sum256(payload.Bytes())
}

func canonicalIdentifier(value string, maximum int) bool {
	return value != "" && len(value) <= maximum && value == strings.TrimSpace(value) && canonicalIDPattern.MatchString(value)
}
