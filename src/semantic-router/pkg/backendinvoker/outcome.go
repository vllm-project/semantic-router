package backendinvoker

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const (
	DispatchOutcomeHeader = "X-VSR-Dispatch-Outcome"
	dispatchOutcomePrefix = "vdo"
)

// DispatchOutcomeCandidate is the bounded Router-owned response summary for
// one candidate that actually began a physical attempt. Full attempt evidence
// remains in the authoritative journal; this token identifies exactly which
// dispatch records the outer settlement layer must read.
type DispatchOutcomeCandidate struct {
	DispatchID         string          `json:"dispatchId"`
	DispatchType       string          `json:"dispatchType"`
	Ordinal            int             `json:"ordinal"`
	DispatchPlanDigest string          `json:"dispatchPlanDigest"`
	ModelID            string          `json:"modelId"`
	ModelRevision      int64           `json:"modelRevision"`
	Priority           int             `json:"priority"`
	State              AttemptState    `json:"state"`
	FallbackTrigger    FallbackTrigger `json:"fallbackTrigger,omitempty"`
	AttemptCount       int             `json:"attemptCount"`
}

// DispatchOutcome is signed by the private dispatch handler using the same
// rotated keyring as request capabilities. Provider headers cannot forge or
// retarget it.
type DispatchOutcome struct {
	NamespaceID        string                     `json:"namespaceId"`
	QuotaPartition     string                     `json:"quotaPartition"`
	PublicationID      string                     `json:"publicationId"`
	RuntimeEpoch       uint64                     `json:"runtimeEpoch"`
	RoutingRevision    int64                      `json:"routingRevision"`
	RoutingDigest      string                     `json:"routingDigest"`
	AdmissionID        string                     `json:"admissionId"`
	AdmissionDigest    string                     `json:"admissionDigest"`
	RequestID          string                     `json:"requestId"`
	RequestDigest      string                     `json:"requestDigest"`
	Attempted          []DispatchOutcomeCandidate `json:"attempted"`
	SelectedDispatchID string                     `json:"selectedDispatchId,omitempty"`
	Audience           string                     `json:"audience"`
	IssuedAt           int64                      `json:"issuedAt"`
	ExpiresAt          int64                      `json:"expiresAt"`
}

func (k SigningKeyring) SignOutcome(outcome DispatchOutcome, now time.Time) (string, error) {
	key, ok := k.Keys[k.ActiveVersion]
	if !ok || len(key) < sha256.Size {
		return "", fmt.Errorf("active dispatch signing key is unavailable")
	}
	if err := validateDispatchOutcome(outcome, now, k.MaxLifetime); err != nil {
		return "", err
	}
	payload, err := json.Marshal(outcome)
	if err != nil {
		return "", fmt.Errorf("marshal dispatch outcome: %w", err)
	}
	encoded := base64.RawURLEncoding.EncodeToString(payload)
	message := dispatchOutcomePrefix + "." + k.ActiveVersion + "." + encoded
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(message))
	return message + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func (k SigningKeyring) VerifyOutcome(token, audience string, now time.Time) (DispatchOutcome, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 4 || parts[0] != dispatchOutcomePrefix {
		return DispatchOutcome{}, fmt.Errorf("invalid dispatch outcome")
	}
	key, ok := k.Keys[parts[1]]
	if !ok || len(key) < sha256.Size {
		return DispatchOutcome{}, fmt.Errorf("unknown dispatch signing key")
	}
	provided, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(provided) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(provided) != parts[3] {
		return DispatchOutcome{}, fmt.Errorf("invalid dispatch outcome signature")
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(strings.Join(parts[:3], ".")))
	expected := mac.Sum(nil)
	if len(provided) != len(expected) || subtle.ConstantTimeCompare(provided, expected) != 1 {
		return DispatchOutcome{}, fmt.Errorf("invalid dispatch outcome signature")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return DispatchOutcome{}, fmt.Errorf("invalid dispatch outcome payload")
	}
	var outcome DispatchOutcome
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&outcome); err != nil {
		return DispatchOutcome{}, fmt.Errorf("invalid dispatch outcome payload")
	}
	if outcome.Audience != audience {
		return DispatchOutcome{}, fmt.Errorf("dispatch outcome audience mismatch")
	}
	if err := validateDispatchOutcome(outcome, now, k.MaxLifetime); err != nil {
		return DispatchOutcome{}, err
	}
	return outcome, nil
}

func outcomeForResult(
	capability DispatchCapability,
	result Result,
	now time.Time,
	maximumLifetime time.Duration,
) (DispatchOutcome, error) {
	if err := validateResultAgainstCapability(capability, result); err != nil {
		return DispatchOutcome{}, err
	}
	lifetime := maximumLifetime
	if lifetime <= 0 || lifetime > 30*time.Second {
		lifetime = 30 * time.Second
	}
	outcome := DispatchOutcome{
		NamespaceID: capability.NamespaceID, QuotaPartition: capability.QuotaPartition,
		PublicationID: capability.PublicationID, RuntimeEpoch: capability.RuntimeEpoch,
		RoutingRevision: capability.RoutingRevision, RoutingDigest: capability.RoutingDigest,
		AdmissionID: capability.AdmissionID, AdmissionDigest: capability.AdmissionDigest,
		RequestID: capability.RequestID, RequestDigest: capability.RequestDigest,
		Attempted: make([]DispatchOutcomeCandidate, 0, len(result.Outcomes)),
		Audience:  capability.Audience, IssuedAt: now.Unix(), ExpiresAt: now.Add(lifetime).Unix(),
	}
	for _, candidate := range result.Outcomes {
		outcome.Attempted = append(outcome.Attempted, DispatchOutcomeCandidate{
			DispatchID: candidate.DispatchID, DispatchType: candidate.DispatchType,
			Ordinal: candidate.Ordinal, DispatchPlanDigest: candidate.DispatchPlanDigest,
			ModelID: candidate.ModelID, ModelRevision: candidate.ModelRevision,
			Priority: candidate.Priority, State: candidate.State,
			FallbackTrigger: candidate.FallbackTrigger, AttemptCount: len(candidate.Attempts),
		})
	}
	if result.Selected != nil {
		outcome.SelectedDispatchID = result.Selected.DispatchID
	}
	return outcome, nil
}

func validateResultAgainstCapability(capability DispatchCapability, result Result) error {
	if len(result.Outcomes) > len(capability.Candidates) {
		return fmt.Errorf("dispatch result contains candidates outside its capability")
	}
	for index, outcome := range result.Outcomes {
		if !sameCandidate(capability.Candidates[index], DispatchCandidate{
			DispatchID: outcome.DispatchID, DispatchType: outcome.DispatchType,
			Ordinal: outcome.Ordinal, DispatchPlanDigest: outcome.DispatchPlanDigest,
			ModelID: outcome.ModelID, ModelRevision: outcome.ModelRevision, Priority: outcome.Priority,
		}) || len(outcome.Attempts) == 0 || len(outcome.Attempts) > 6 {
			return fmt.Errorf("dispatch result candidate %d does not match its capability", index)
		}
		for attemptIndex, attempt := range outcome.Attempts {
			if attempt.Number != attemptIndex+1 || attempt.ID == "" || attempt.BackendID == "" {
				return fmt.Errorf("dispatch result candidate %d has a non-contiguous attempt journal", index)
			}
		}
		terminal := outcome.Attempts[len(outcome.Attempts)-1]
		if terminal.State != outcome.State || terminal.FallbackTrigger != outcome.FallbackTrigger {
			return fmt.Errorf("dispatch result candidate %d terminal evidence differs", index)
		}
		if index+1 < len(result.Outcomes) &&
			(outcome.State != AttemptKnownZero || !fallbackEnabled(capability.Fallback, outcome.FallbackTrigger)) {
			return fmt.Errorf("dispatch result advanced without an authorized known-zero trigger")
		}
	}
	if result.Selected != nil {
		if len(result.Outcomes) == 0 || result.Selected.DispatchID != result.Outcomes[len(result.Outcomes)-1].DispatchID ||
			result.Outcomes[len(result.Outcomes)-1].State != AttemptResponseStarted {
			return fmt.Errorf("selected dispatch is not the terminal response-started candidate")
		}
	}
	return nil
}

func validateDispatchOutcome(outcome DispatchOutcome, now time.Time, maximumLifetime time.Duration) error {
	if !validBoundedIdentity(outcome.NamespaceID) || !validBoundedIdentity(outcome.QuotaPartition) ||
		!validBoundedIdentity(outcome.PublicationID) || outcome.RuntimeEpoch == 0 ||
		outcome.RoutingRevision <= 0 || !validSHA256Hex(outcome.RoutingDigest) ||
		!validBoundedIdentity(outcome.AdmissionID) || !validSHA256Hex(outcome.AdmissionDigest) ||
		!validBoundedIdentity(outcome.RequestID) || !validRequestDigest(outcome.RequestDigest) ||
		!validBoundedIdentity(outcome.Audience) || len(outcome.Attempted) > maximumDispatchCandidates {
		return fmt.Errorf("dispatch outcome is incomplete")
	}
	seen := make(map[string]struct{}, len(outcome.Attempted))
	candidates := make([]DispatchCandidate, 0, len(outcome.Attempted))
	for index, candidate := range outcome.Attempted {
		if !validBoundedIdentity(candidate.DispatchID) || !validBoundedIdentity(candidate.DispatchType) ||
			candidate.Ordinal < 0 || !validSHA256Hex(candidate.DispatchPlanDigest) ||
			!validBoundedIdentity(candidate.ModelID) || candidate.ModelRevision <= 0 ||
			candidate.Priority < 0 || candidate.Priority > 31 ||
			candidate.AttemptCount < 1 || candidate.AttemptCount > 6 {
			return fmt.Errorf("dispatch outcome candidate %d is invalid", index)
		}
		if _, duplicate := seen[candidate.DispatchID]; duplicate {
			return fmt.Errorf("dispatch outcome candidate is duplicated")
		}
		seen[candidate.DispatchID] = struct{}{}
		candidates = append(candidates, DispatchCandidate{
			DispatchID: candidate.DispatchID, DispatchType: candidate.DispatchType,
			Ordinal: candidate.Ordinal, DispatchPlanDigest: candidate.DispatchPlanDigest,
			ModelID: candidate.ModelID, ModelRevision: candidate.ModelRevision, Priority: candidate.Priority,
		})
		switch candidate.State {
		case AttemptKnownZero:
			if candidate.FallbackTrigger != "" && fallbackTriggerOrder(candidate.FallbackTrigger) < 0 {
				return fmt.Errorf("dispatch outcome has an invalid fallback trigger")
			}
		case AttemptResponseStarted, AttemptUnknown:
			if candidate.FallbackTrigger != "" {
				return fmt.Errorf("non-zero dispatch outcome cannot authorize fallback")
			}
		default:
			return fmt.Errorf("dispatch outcome has an invalid state")
		}
	}
	if len(candidates) > 0 {
		if err := validateCandidateChain(candidates, FallbackPolicy{}); err != nil {
			return fmt.Errorf("dispatch outcome candidate chain: %w", err)
		}
	}
	if outcome.SelectedDispatchID != "" {
		if len(outcome.Attempted) == 0 ||
			outcome.Attempted[len(outcome.Attempted)-1].DispatchID != outcome.SelectedDispatchID ||
			outcome.Attempted[len(outcome.Attempted)-1].State != AttemptResponseStarted {
			return fmt.Errorf("dispatch outcome selected candidate is invalid")
		}
	}
	issued := time.Unix(outcome.IssuedAt, 0)
	expires := time.Unix(outcome.ExpiresAt, 0)
	if maximumLifetime <= 0 || !expires.After(issued) || expires.Sub(issued) > maximumLifetime ||
		now.Before(issued.Add(-5*time.Second)) || !now.Before(expires) {
		return fmt.Errorf("dispatch outcome is outside its validity window")
	}
	return nil
}
