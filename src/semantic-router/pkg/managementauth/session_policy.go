package managementauth

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

// SessionPolicyLoader is authoritative for every token issue and validation.
// Implementations must not return a positive cached policy after durable state
// becomes unavailable.
type SessionPolicyLoader interface {
	LoadSessionPolicy(context.Context) (SessionPolicy, error)
}

const SupportedSessionPolicySeedVersion uint64 = 1

type AuthenticationRequirementKind string

const (
	RequirementHuman    AuthenticationRequirementKind = "human"
	RequirementWorkload AuthenticationRequirementKind = "workload"
)

type HumanRequirement struct {
	MinimumAAL                  string   `json:"minimum_aal"`
	AcceptedAMR                 []string `json:"accepted_amr"`
	MaxAuthenticationAgeSeconds int64    `json:"max_authentication_age_seconds"`
}

type WorkloadRequirement struct {
	MinimumWorkloadClass string `json:"minimum_workload_class"`
	MaxSourceAgeSeconds  int64  `json:"max_source_age_seconds"`
}

// AuthenticationRequirement is one branch in an explicit OR-set. Exactly one
// typed evidence predicate is present; human and workload assurance are never
// compared or coerced.
type AuthenticationRequirement struct {
	Kind     AuthenticationRequirementKind `json:"kind"`
	Human    *HumanRequirement             `json:"human,omitempty"`
	Workload *WorkloadRequirement          `json:"workload,omitempty"`
}

type ActionRequirement struct {
	AnyOf []AuthenticationRequirement `json:"any_of"`
}

type SessionPolicy struct {
	AccessTokenTTL     time.Duration
	SessionTTL         time.Duration
	MaxActiveSessions  int
	ActionRequirements map[string]ActionRequirement
	SeedVersion        uint64
	Revision           uint64
	UpdatedAt          time.Time
}

func (policy SessionPolicy) Validate() error {
	if policy.AccessTokenTTL < time.Second || policy.AccessTokenTTL > 24*time.Hour ||
		policy.SessionTTL < policy.AccessTokenTTL || policy.SessionTTL > 30*24*time.Hour ||
		policy.MaxActiveSessions < 1 || policy.MaxActiveSessions > 100 ||
		policy.SeedVersion != SupportedSessionPolicySeedVersion || policy.Revision == 0 || policy.UpdatedAt.IsZero() {
		return errors.New("management session policy scalar values are invalid")
	}
	if len(policy.ActionRequirements) == 0 || len(policy.ActionRequirements) > 64 {
		return errors.New("management session policy action requirements are invalid")
	}
	for action, requirement := range policy.ActionRequirements {
		if !canonicalAction(action) || len(requirement.AnyOf) == 0 || len(requirement.AnyOf) > 2 {
			return fmt.Errorf("management session policy action %q is invalid", action)
		}
		seen := make(map[AuthenticationRequirementKind]struct{}, len(requirement.AnyOf))
		for _, branch := range requirement.AnyOf {
			if _, duplicate := seen[branch.Kind]; duplicate {
				return fmt.Errorf("management session policy action %q repeats an evidence kind", action)
			}
			seen[branch.Kind] = struct{}{}
			if err := branch.Validate(); err != nil {
				return fmt.Errorf("management session policy action %q: %w", action, err)
			}
		}
	}
	return nil
}

func (requirement AuthenticationRequirement) Validate() error {
	switch requirement.Kind {
	case RequirementHuman:
		if requirement.Human == nil || requirement.Workload != nil || !validAAL(requirement.Human.MinimumAAL) ||
			requirement.Human.MaxAuthenticationAgeSeconds < 1 || requirement.Human.MaxAuthenticationAgeSeconds > 30*24*60*60 {
			return errors.New("human authentication requirement is invalid")
		}
		methods := slices.Clone(requirement.Human.AcceptedAMR)
		slices.Sort(methods)
		for index, method := range methods {
			if !canonicalText(method, 1, 64) || (index > 0 && methods[index-1] == method) {
				return errors.New("human accepted AMR set is invalid")
			}
		}
	case RequirementWorkload:
		if requirement.Workload == nil || requirement.Human != nil || !validWorkloadClass(requirement.Workload.MinimumWorkloadClass) ||
			requirement.Workload.MaxSourceAgeSeconds < 1 || requirement.Workload.MaxSourceAgeSeconds > 365*24*60*60 {
			return errors.New("workload authentication requirement is invalid")
		}
	default:
		return errors.New("authentication requirement kind is invalid")
	}
	return nil
}

func (requirement ActionRequirement) Allows(session LiveSession, now time.Time) bool {
	for _, branch := range requirement.AnyOf {
		switch branch.Kind {
		case RequirementHuman:
			if session.Human != nil && branch.Human != nil &&
				aalRank(session.Human.AAL) >= aalRank(branch.Human.MinimumAAL) &&
				now.Sub(session.AuthenticatedAt) <= time.Duration(branch.Human.MaxAuthenticationAgeSeconds)*time.Second &&
				acceptedAMR(session.Human.AMR, branch.Human.AcceptedAMR) {
				return true
			}
		case RequirementWorkload:
			if session.Workload != nil && branch.Workload != nil &&
				workloadRank(session.Workload.Class) >= workloadRank(branch.Workload.MinimumWorkloadClass) &&
				now.Sub(time.Unix(session.Workload.SourceAssuredAt, 0)) <= time.Duration(branch.Workload.MaxSourceAgeSeconds)*time.Second {
				return true
			}
		}
	}
	return false
}

func canonicalAction(value string) bool {
	if len(value) < 1 || len(value) > 128 || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if (character < 'a' || character > 'z') && (character < '0' || character > '9') && character != '.' && character != '_' {
			return false
		}
	}
	return true
}

func aalRank(value string) int {
	switch value {
	case "aal1":
		return 1
	case "aal2":
		return 2
	case "aal3":
		return 3
	default:
		return 0
	}
}

func workloadRank(value string) int {
	switch value {
	case "workload_standard":
		return 1
	case "workload_strong":
		return 2
	default:
		return 0
	}
}

func acceptedAMR(actual, required []string) bool {
	if len(required) == 0 {
		return true
	}
	for _, candidate := range required {
		if slices.Contains(actual, candidate) {
			return true
		}
	}
	return false
}
