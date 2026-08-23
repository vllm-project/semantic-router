package accessruntime

import (
	"encoding/base64"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// CompileRuleBindings converts one immutable access projection into the exact
// binding-owned counter plan used by both admission and Management live-meter
// reads. Keeping this conversion shared prevents the read API from drifting
// from data-plane quota semantics.
func CompileRuleBindings(projection accessprojection.Projection) []quotaruntime.RuleBinding {
	count := 0
	for _, binding := range projection.RateBindings {
		count += len(binding.Rules)
	}
	rules := make([]quotaruntime.RuleBinding, 0, count)
	for _, binding := range projection.RateBindings {
		for _, projected := range binding.Rules {
			rule := projected.Rule
			currency := ""
			if string(rule.Metric) == "cost" {
				currency = projection.BillingCurrency
			}
			schedule := make([]quotaruntime.CalendarInterval, len(projected.CalendarSchedule))
			for index, interval := range projected.CalendarSchedule {
				schedule[index] = quotaruntime.CalendarInterval{Start: interval.Start, End: interval.End}
			}
			rules = append(rules, quotaruntime.RuleBinding{
				BindingID: binding.BindingID, Rule: rule, Currency: currency, CalendarSchedule: schedule,
			})
		}
	}
	return rules
}

func compilePreconditions(
	keyPrefix string,
	location CredentialLocation,
	kind accesscredential.Kind,
	publicID string,
	credential accessprojection.CredentialProjection,
	active ActivePolicy,
	projection accessprojection.Projection,
) ([]quotaruntime.AdmissionPrecondition, error) {
	keys, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(keyPrefix, location.QuotaPartition)
	if err != nil {
		return nil, err
	}
	publicationKeys, err := accesspublisher.NewKeyspace(keyPrefix, location.NamespaceID, location.QuotaPartition)
	if err != nil {
		return nil, err
	}
	credentialKey := keys.Credential(string(kind), publicID)
	logicalKey := keys.LogicalKey(credential.KeyID)
	activeKey := keys.Active(credential.KeyID)
	policyKey := keys.Policy(credential.KeyID, strconv.FormatUint(active.Revision, 10))
	checks := []quotaruntime.AdmissionPrecondition{
		hashEqual(publicationKeys.AccessGate(), "publication_id", active.PublicationID, quotaruntime.AdmissionUnavailable, "access_publication_changed"),
		hashEqual(publicationKeys.AccessGate(), "runtime_epoch", strconv.FormatUint(active.RuntimeEpoch, 10), quotaruntime.AdmissionUnavailable, "runtime_epoch_changed"),
		hashEqual(publicationKeys.RoutingGate(), "publication_id", active.PublicationID, quotaruntime.AdmissionUnavailable, "routing_publication_changed"),
		hashEqual(publicationKeys.RoutingGate(), "runtime_epoch", strconv.FormatUint(active.RuntimeEpoch, 10), quotaruntime.AdmissionUnavailable, "runtime_epoch_changed"),
		hashEqual(publicationKeys.RoutingGate(), "revision", strconv.FormatInt(active.RoutingRevision, 10), quotaruntime.AdmissionUnavailable, "routing_revision_changed"),
		hashEqual(publicationKeys.RoutingGate(), "snapshot_digest", active.RoutingSnapshotHash, quotaruntime.AdmissionUnavailable, "routing_snapshot_changed"),
		hashEqual(credentialKey, "publication_id", active.PublicationID, quotaruntime.AdmissionUnauthenticated, "credential_changed"),
		hashEqual(credentialKey, "kind", string(kind), quotaruntime.AdmissionUnauthenticated, "credential_changed"),
		hashEqual(credentialKey, "key_id", credential.KeyID, quotaruntime.AdmissionUnauthenticated, "credential_changed"),
		hashEqual(credentialKey, "secret_hmac", base64.RawURLEncoding.EncodeToString(credential.SecretHMAC), quotaruntime.AdmissionUnauthenticated, "credential_changed"),
		hashEqual(credentialKey, "pepper_version", credential.PepperVersion, quotaruntime.AdmissionUnavailable, "credential_pepper_changed"),
		hashEqual(credentialKey, "status", credential.Status, quotaruntime.AdmissionUnauthenticated, "credential_inactive"),
		{Key: credentialKey, Kind: quotaruntime.AdmissionCheckNotBefore, Field: "not_before_ms", Failure: quotaruntime.AdmissionUnauthenticated, Reason: "credential_not_yet_valid"},
		hashEqual(logicalKey, "publication_id", active.PublicationID, quotaruntime.AdmissionUnauthenticated, "key_changed"),
		hashEqual(logicalKey, "status", string(projection.KeyStatus), quotaruntime.AdmissionUnauthenticated, "key_inactive"),
		hashEqual(logicalKey, "policy_epoch", strconv.FormatUint(projection.PolicyEpoch, 10), quotaruntime.AdmissionForbidden, "key_policy_changed"),
		hashEqual(logicalKey, "delegation_epoch", strconv.FormatUint(projection.DelegationEpoch, 10), quotaruntime.AdmissionForbidden, "key_delegation_changed"),
		hashEqual(activeKey, "publication_id", active.PublicationID, quotaruntime.AdmissionForbidden, "policy_changed"),
		hashEqual(activeKey, "revision", strconv.FormatUint(active.Revision, 10), quotaruntime.AdmissionForbidden, "policy_changed"),
		hashEqual(activeKey, "digest", active.Digest, quotaruntime.AdmissionForbidden, "policy_changed"),
		hashEqual(policyKey, "publication_id", active.PublicationID, quotaruntime.AdmissionUnavailable, "policy_projection_changed"),
		hashEqual(policyKey, "digest", active.Digest, quotaruntime.AdmissionUnavailable, "policy_projection_changed"),
	}
	if kind == accesscredential.KindDelegation {
		checks = append(checks,
			hashEqual(credentialKey, "management_session_id", credential.ManagementSessionID, quotaruntime.AdmissionForbidden, "management_session_changed"),
			hashEqual(credentialKey, "principal_id", credential.PrincipalID, quotaruntime.AdmissionForbidden, "management_principal_changed"),
			hashEqual(credentialKey, "delegation_epoch", strconv.FormatUint(credential.DelegationEpoch, 10), quotaruntime.AdmissionForbidden, "key_delegation_changed"),
			hashEqual(credentialKey, "user_id", credential.UserID, quotaruntime.AdmissionForbidden, "delegation_user_changed"),
			hashEqual(credentialKey, "team_id", credential.TeamID, quotaruntime.AdmissionForbidden, "delegation_team_changed"),
			hashEqual(credentialKey, "audience", credential.Audience, quotaruntime.AdmissionUnauthenticated, "delegation_audience_changed"),
		)
	}
	if credential.ExpiresAt != nil {
		checks = append(checks, quotaruntime.AdmissionPrecondition{
			Key: credentialKey, Kind: quotaruntime.AdmissionCheckExpiresAfter, Field: "expires_at_ms",
			Failure: quotaruntime.AdmissionUnauthenticated, Reason: "credential_expired",
		})
	}
	if projection.KeyExpiresAt != nil {
		checks = append(checks, quotaruntime.AdmissionPrecondition{
			Key: logicalKey, Kind: quotaruntime.AdmissionCheckExpiresAfter, Field: "expires_at_ms",
			Failure: quotaruntime.AdmissionUnauthenticated, Reason: "key_expired",
		})
	}
	for _, denied := range denyReferences(keys, kind, publicID, credential, projection) {
		checks = append(checks, quotaruntime.AdmissionPrecondition{
			Key: denied.key, Kind: quotaruntime.AdmissionCheckKeyAbsent,
			Failure: quotaruntime.AdmissionForbidden, Reason: denied.reason,
		})
	}
	return checks, nil
}

type denyReference struct{ key, reason string }

func denyReferences(
	keys quotaruntime.AccessProjectionKeyspace,
	kind accesscredential.Kind,
	publicID string,
	credential accessprojection.CredentialProjection,
	projection accessprojection.Projection,
) []denyReference {
	references := []denyReference{
		{keys.Deny("namespace", projection.NamespaceID), "namespace_denied"},
		{keys.Deny("credential", string(kind)+":"+publicID), "credential_denied"},
		{keys.Deny("api_key", projection.KeyID), "key_denied"},
	}
	userID, teamID := projection.UserID, projection.TeamID
	if kind == accesscredential.KindDelegation {
		userID, teamID = credential.UserID, credential.TeamID
		references = append(references,
			denyReference{keys.Deny("management_session", credential.ManagementSessionID), "management_session_denied"},
			denyReference{keys.Deny("management_principal", credential.PrincipalID), "management_principal_denied"},
		)
	}
	if userID != "" {
		references = append(references, denyReference{keys.Deny("user", userID), "user_denied"})
	}
	if teamID != "" {
		references = append(references, denyReference{keys.Deny("team", teamID), "team_denied"})
	}
	if userID != "" && teamID != "" {
		references = append(references, denyReference{keys.Deny("membership", teamID+":"+userID), "membership_denied"})
	}
	return references
}

func hashEqual(key, field, expected string, failure quotaruntime.AdmissionDisposition, reason string) quotaruntime.AdmissionPrecondition {
	return quotaruntime.AdmissionPrecondition{Key: key, Kind: quotaruntime.AdmissionCheckHashEqual, Field: field, Expected: expected, Failure: failure, Reason: reason}
}

func tenantContext(
	projection accessprojection.Projection,
	credential accessprojection.CredentialProjection,
	active ActivePolicy,
) TenantContext {
	claims := make(map[string]routingsnapshot.ClaimValue, len(projection.RoutingClaims))
	for key, value := range projection.RoutingClaims {
		claims[key] = value
	}
	userID, teamID := projection.UserID, projection.TeamID
	if credential.Kind == string(accesscredential.KindDelegation) {
		userID, teamID = credential.UserID, credential.TeamID
	}
	return TenantContext{
		NamespaceID:    projection.NamespaceID,
		QuotaPartition: projection.QuotaPartition, APIKeyID: projection.KeyID,
		UserID: userID, TeamID: teamID,
		PolicyRevision: projection.Revision, PolicyDigest: projection.Digest,
		PublicationID: active.PublicationID, RuntimeEpoch: active.RuntimeEpoch,
		RoutingRevision: active.RoutingRevision, RoutingDigest: active.RoutingSnapshotHash,
		BillingCurrency: projection.BillingCurrency,
		RoutingClaims:   claims,
	}
}
