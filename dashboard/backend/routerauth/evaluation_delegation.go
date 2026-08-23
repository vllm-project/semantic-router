package routerauth

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

var errEvaluationInferenceUnavailable = errors.New("router evaluation inference authorization is unavailable")

// IssueEvaluationInferenceToken selects a key from Router Management's narrow
// self-service eligibility view and creates a short-lived delegated session for
// this Dashboard principal. Raw API-key credentials are never read or revealed.
func (provider *managementSessionProvider) IssueEvaluationInferenceToken(
	ctx context.Context,
	principal dashboardauth.AuthContext,
	ownerUserID string,
	ownerTeamID string,
) (string, error) {
	managementToken, identity, err := provider.evaluationIdentity(ctx, principal)
	if err != nil {
		return "", errEvaluationInferenceUnavailable
	}
	namespaceID, userID, ok := evaluationNamespace(identity, ownerUserID, ownerTeamID)
	if !ok {
		return "", errEvaluationInferenceUnavailable
	}
	keyID, err := provider.evaluationInferenceKey(
		ctx,
		managementToken,
		namespaceID,
		userID,
		ownerTeamID,
	)
	if err != nil {
		return "", errEvaluationInferenceUnavailable
	}

	var envelope managementapi.SecretEnvelope
	err = provider.authorizedManagementRequest(
		ctx,
		managementToken,
		namespaceID,
		http.MethodPost,
		managementBasePath+"/self/inference-sessions",
		managementapi.DelegatedInferenceSessionCreateRequest{KeyID: keyID},
		map[string]string{managementapi.HeaderIdempotencyKey: uuid.NewString()},
		[]int{http.StatusCreated},
		&envelope,
	)
	if err != nil || envelope.Kind != managementapi.SecretKindDelegatedCredential ||
		strings.TrimSpace(envelope.Secret) == "" || strings.ContainsAny(envelope.Secret, "\r\n\t ") ||
		envelope.ExpiresAt == nil || !provider.now().UTC().Before(envelope.ExpiresAt.UTC()) {
		return "", errEvaluationInferenceUnavailable
	}
	return envelope.Secret, nil
}

func evaluationNamespace(
	identity managementapi.Me,
	ownerUserID string,
	ownerTeamID string,
) (string, string, bool) {
	eligible := make([]managementapi.MeNamespaceScope, 0, len(identity.Namespaces))
	for _, namespace := range identity.Namespaces {
		if namespace.Namespace.Status != "active" || namespace.User == nil ||
			namespace.User.Status != "active" || strings.TrimSpace(namespace.User.UserID) == "" {
			continue
		}
		if ownerTeamID == "" {
			if namespace.User.UserID == ownerUserID && ownerUserID != "" {
				eligible = append(eligible, namespace)
			}
			continue
		}
		for _, membership := range namespace.Teams {
			if membership.TeamID == ownerTeamID && membership.Status == "active" {
				eligible = append(eligible, namespace)
				break
			}
		}
	}
	if len(eligible) == 0 {
		return "", "", false
	}
	sort.Slice(eligible, func(i, j int) bool {
		return eligible[i].Namespace.NamespaceID < eligible[j].Namespace.NamespaceID
	})
	return eligible[0].Namespace.NamespaceID, eligible[0].User.UserID, true
}

func (provider *managementSessionProvider) evaluationInferenceKey(
	ctx context.Context,
	managementToken string,
	namespaceID string,
	userID string,
	teamID string,
) (string, error) {
	cursor := ""
	seenCursors := make(map[string]struct{})
	for {
		query := url.Values{"pageSize": []string{"200"}}
		if cursor != "" {
			query.Set("cursor", cursor)
		}
		var page managementapi.EligibleInferenceKeyPage
		if err := provider.authorizedManagementRequest(
			ctx,
			managementToken,
			namespaceID,
			http.MethodGet,
			managementBasePath+"/self/inference-keys?"+query.Encode(),
			nil,
			nil,
			[]int{http.StatusOK},
			&page,
		); err != nil {
			return "", errEvaluationInferenceUnavailable
		}
		for _, key := range page.Data {
			if evaluationKeyMatches(key, userID, teamID, provider.now().UTC()) {
				return key.KeyID, nil
			}
		}
		if !page.Page.HasMore || page.Page.NextCursor == "" {
			return "", errEvaluationInferenceUnavailable
		}
		if _, duplicate := seenCursors[page.Page.NextCursor]; duplicate {
			return "", errEvaluationInferenceUnavailable
		}
		seenCursors[page.Page.NextCursor] = struct{}{}
		cursor = page.Page.NextCursor
	}
}

func evaluationKeyMatches(
	key managementapi.EligibleInferenceKey,
	userID string,
	teamID string,
	now time.Time,
) bool {
	if key.KeyID == "" || (key.ExpiresAt != nil && !now.Before(key.ExpiresAt.UTC())) {
		return false
	}
	if teamID != "" {
		return key.ContextTeamID == teamID || (key.Owner.Type == "team" && key.Owner.ID == teamID)
	}
	return key.ContextTeamID == "" && key.Owner.Type == "user" && key.Owner.ID == userID
}
