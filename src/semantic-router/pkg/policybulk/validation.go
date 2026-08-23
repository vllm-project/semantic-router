package policybulk

import (
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

var (
	ErrInvalidRequest   = errors.New("policy bulk request is invalid")
	ErrNotFound         = errors.New("policy bulk operation not found")
	ErrConflict         = errors.New("policy bulk operation conflict")
	ErrRevisionConflict = errors.New("policy bulk operation revision conflict")
	ErrLeaseLost        = errors.New("policy bulk item lease is no longer owned")
	ErrExecutionDenied  = errors.New("policy bulk execution authorization denied")
	ErrUnavailable      = errors.New("policy bulk service is unavailable")
)

const validationNamespaceID = "00000000-0000-4000-8000-000000000001"

func ValidateAccessItems(items []AccessBindingItem) error {
	if len(items) == 0 || len(items) > MaximumItems {
		return ErrInvalidRequest
	}
	seen := make(map[string]struct{}, len(items))
	for _, item := range items {
		if !canonicalUUID(item.ItemID) || !canonicalUUID(item.PolicyID) || !validSubject(item.Subject) {
			return ErrInvalidRequest
		}
		if _, duplicate := seen[item.ItemID]; duplicate {
			return ErrInvalidRequest
		}
		seen[item.ItemID] = struct{}{}
	}
	return nil
}

func ValidateRateItems(items []RateBindingItem) error {
	if len(items) == 0 || len(items) > MaximumItems {
		return ErrInvalidRequest
	}
	seen := make(map[string]struct{}, len(items))
	for _, item := range items {
		if !canonicalUUID(item.ItemID) || !validSubject(item.Subject) || !item.Mode.Valid() {
			return ErrInvalidRequest
		}
		if _, duplicate := seen[item.ItemID]; duplicate {
			return ErrInvalidRequest
		}
		seen[item.ItemID] = struct{}{}
		hasPolicy := item.PolicyID != ""
		hasInline := item.InlinePolicy != nil
		if hasPolicy == hasInline || (hasPolicy && !canonicalUUID(item.PolicyID)) {
			return ErrInvalidRequest
		}
		if hasInline && validateInline(item.ItemID, *item.InlinePolicy) != nil {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validateInline(itemID string, inline InlineRateLimitPolicy) error {
	index := 0
	_, err := policymanagement.CompileInlineRateLimitPolicy(policymanagement.InlineRateLimitPolicySpec{
		NamespaceID: validationNamespaceID, PolicyID: itemID,
		Name: inline.Name, Description: inline.Description, Rules: inline.Rules,
		Now: time.Unix(1, 0), NewRuleID: func() string {
			index++
			return uuid.NewSHA1(uuid.NameSpaceOID, []byte(itemID+":"+strconv.Itoa(index))).String()
		},
	})
	return err
}

func validSubject(subject policymanagement.Subject) bool {
	return subject.Type.Valid() && canonicalUUID(subject.ID)
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value && strings.ToLower(value) == value && parsed.Version() >= 1 && parsed.Version() <= 5
}

func validActor(namespaceID string, actor policymanagement.Actor) bool {
	if !canonicalUUID(namespaceID) || !canonicalUUID(actor.PrincipalID) || strings.TrimSpace(actor.RequestID) == "" {
		return false
	}
	for _, principalID := range actor.ActorChain {
		if !canonicalUUID(principalID) {
			return false
		}
	}
	return true
}

func validOperationKind(kind string) bool {
	return kind == AccessBindingOperationKind || kind == RateBindingOperationKind
}

func (state OperationState) Valid() bool {
	switch state {
	case OperationPending, OperationRunning, OperationSucceeded,
		OperationPartiallySucceeded, OperationFailed, OperationCancelled:
		return true
	default:
		return false
	}
}
