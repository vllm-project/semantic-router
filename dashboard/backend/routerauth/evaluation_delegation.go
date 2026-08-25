package routerauth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

const maxEvaluationDelegationCacheEntries = 4096

type evaluationDelegationCacheKey struct {
	managementSessionID string
	namespaceID         string
	keyID               string
}

type cachedEvaluationDelegation struct {
	secret    []byte
	expiresAt time.Time
}

// IssueEvaluationInferenceToken selects a key from Router Management's narrow
// self-service eligibility view and creates a short-lived delegated session for
// this Dashboard principal. Raw API-key credentials are never read or revealed.
func (provider *managementSessionProvider) IssueEvaluationInferenceToken(
	ctx context.Context,
	principal dashboardauth.AuthContext,
	ownerUserID string,
	ownerTeamID string,
) (string, error) {
	managementCredential, identity, err := provider.evaluationIdentity(ctx, principal)
	if err != nil {
		return "", errEvaluationInferenceUnavailable
	}
	namespaceID, userID, ok := evaluationNamespace(identity, ownerUserID, ownerTeamID)
	if !ok {
		return "", errEvaluationInferenceUnavailable
	}
	keyID, err := provider.evaluationInferenceKey(
		ctx,
		managementCredential.accessToken,
		namespaceID,
		userID,
		ownerTeamID,
	)
	if err != nil {
		return "", errEvaluationInferenceUnavailable
	}
	cacheKey := evaluationDelegationCacheKey{
		managementSessionID: managementCredential.managementSessionID,
		namespaceID:         namespaceID,
		keyID:               keyID,
	}
	if secret, ok := provider.cachedEvaluationDelegation(cacheKey); ok {
		return secret, nil
	}
	idempotencyKey, err := evaluationDelegationIdempotencyKey(
		managementCredential.managementSessionID,
		namespaceID,
		keyID,
	)
	if err != nil {
		return "", errEvaluationInferenceUnavailable
	}

	var envelope managementapi.SecretEnvelope
	err = provider.authorizedManagementRequest(
		ctx,
		managementCredential.accessToken,
		namespaceID,
		http.MethodPost,
		managementBasePath+"/self/inference-sessions",
		managementapi.DelegatedInferenceSessionCreateRequest{KeyID: keyID},
		map[string]string{managementapi.HeaderIdempotencyKey: idempotencyKey},
		[]int{http.StatusCreated},
		&envelope,
	)
	if err != nil || envelope.Kind != managementapi.SecretKindDelegatedCredential ||
		strings.TrimSpace(envelope.Secret) == "" || strings.ContainsAny(envelope.Secret, "\r\n\t ") ||
		envelope.ExpiresAt == nil || !provider.now().UTC().Before(envelope.ExpiresAt.UTC()) {
		return "", errEvaluationInferenceUnavailable
	}
	provider.cacheEvaluationDelegation(cacheKey, envelope.Secret, envelope.ExpiresAt.UTC())
	return envelope.Secret, nil
}

func (provider *managementSessionProvider) cachedEvaluationDelegation(
	key evaluationDelegationCacheKey,
) (string, bool) {
	if provider == nil {
		return "", false
	}
	now := provider.now().UTC()
	provider.mu.Lock()
	defer provider.mu.Unlock()
	provider.pruneEvaluationDelegationsLocked(now)
	cached, ok := provider.delegations[key]
	if !ok || len(cached.secret) == 0 || !now.Add(tokenRefreshSkew).Before(cached.expiresAt) {
		return "", false
	}
	return string(cached.secret), true
}

func (provider *managementSessionProvider) cacheEvaluationDelegation(
	key evaluationDelegationCacheKey,
	secret string,
	expiresAt time.Time,
) {
	if provider == nil || secret == "" || expiresAt.IsZero() {
		return
	}
	now := provider.now().UTC()
	provider.mu.Lock()
	defer provider.mu.Unlock()
	provider.pruneEvaluationDelegationsLocked(now)
	if provider.delegations == nil {
		provider.delegations = make(map[evaluationDelegationCacheKey]cachedEvaluationDelegation)
	}
	if previous, ok := provider.delegations[key]; ok {
		clear(previous.secret)
	}
	for len(provider.delegations) >= maxEvaluationDelegationCacheEntries {
		var oldestKey evaluationDelegationCacheKey
		var oldestExpiry time.Time
		found := false
		for candidateKey, candidate := range provider.delegations {
			if !found || candidate.expiresAt.Before(oldestExpiry) {
				oldestKey, oldestExpiry, found = candidateKey, candidate.expiresAt, true
			}
		}
		if !found {
			break
		}
		oldest := provider.delegations[oldestKey]
		clear(oldest.secret)
		delete(provider.delegations, oldestKey)
	}
	provider.delegations[key] = cachedEvaluationDelegation{
		secret: append([]byte(nil), secret...), expiresAt: expiresAt,
	}
}

func (provider *managementSessionProvider) pruneEvaluationDelegationsLocked(now time.Time) {
	for key, cached := range provider.delegations {
		if !now.Add(tokenRefreshSkew).Before(cached.expiresAt) {
			clear(cached.secret)
			delete(provider.delegations, key)
		}
	}
}

func (provider *managementSessionProvider) clearEvaluationDelegationsLocked(managementSessionID string) {
	if managementSessionID == "" {
		return
	}
	for key, cached := range provider.delegations {
		if key.managementSessionID == managementSessionID {
			clear(cached.secret)
			delete(provider.delegations, key)
		}
	}
}

func evaluationDelegationIdempotencyKey(
	managementSessionID string,
	namespaceID string,
	keyID string,
) (string, error) {
	for _, value := range []string{managementSessionID, namespaceID, keyID} {
		parsed, err := uuid.Parse(value)
		if err != nil || parsed.String() != value {
			return "", errEvaluationInferenceUnavailable
		}
	}
	payload := strings.Join([]string{
		"vllm-sr/dashboard/playground-delegation/v1",
		managementSessionID,
		namespaceID,
		keyID,
	}, "\x00")
	digest := sha256.Sum256([]byte(payload))
	return "playground-delegation-" + hex.EncodeToString(digest[:]), nil
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
