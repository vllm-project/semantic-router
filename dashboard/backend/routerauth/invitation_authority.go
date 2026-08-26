package routerauth

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	maximumInvitationErrorBodyBytes = 64 << 10
)

type invitationErrorDiagnostic struct {
	Error struct {
		Code      string `json:"code"`
		RequestID string `json:"requestId"`
	} `json:"error"`
}

func (provider *managementSessionProvider) ListInvitations(
	ctx context.Context,
	actor dashboardauth.AuthContext,
	namespaceID string,
) ([]managementapi.Invitation, error) {
	var page managementapi.InvitationPage
	err := provider.authorizedInvitationRequest(ctx, actor, namespaceID, http.MethodGet,
		managementBasePath+"/invitations?pageSize=200", nil, nil, http.StatusOK, &page)
	return page.Data, err
}

func (provider *managementSessionProvider) CreateInvitation(
	ctx context.Context,
	actor dashboardauth.AuthContext,
	namespaceID string,
	idempotencyKey string,
	request managementapi.InvitationCreateRequest,
) (managementapi.InvitationIssuedSecret, error) {
	var issued managementapi.InvitationIssuedSecret
	err := provider.authorizedInvitationRequest(ctx, actor, namespaceID, http.MethodPost,
		managementBasePath+"/invitations", request,
		map[string]string{managementapi.HeaderIdempotencyKey: idempotencyKey}, http.StatusCreated, &issued)
	return issued, err
}

func (provider *managementSessionProvider) RotateInvitation(
	ctx context.Context,
	actor dashboardauth.AuthContext,
	namespaceID, invitationID string,
	expectedRevision uint64,
	idempotencyKey string,
	expiresAt *time.Time,
) (managementapi.InvitationIssuedSecret, error) {
	var issued managementapi.InvitationIssuedSecret
	err := provider.authorizedInvitationRequest(ctx, actor, namespaceID, http.MethodPost,
		managementBasePath+"/invitations/"+url.PathEscape(invitationID)+":rotate-token",
		managementapi.InvitationRotateTokenRequest{ExpiresAt: expiresAt}, map[string]string{
			managementapi.HeaderIfMatch:        invitationETag(expectedRevision),
			managementapi.HeaderIdempotencyKey: idempotencyKey,
		}, http.StatusOK, &issued)
	return issued, err
}

func (provider *managementSessionProvider) RevokeInvitation(
	ctx context.Context,
	actor dashboardauth.AuthContext,
	namespaceID, invitationID string,
	expectedRevision uint64,
) (uint64, error) {
	responseHeaders, err := provider.authorizedInvitationRequestWithHeaders(ctx, actor, namespaceID, http.MethodDelete,
		managementBasePath+"/invitations/"+url.PathEscape(invitationID), nil,
		map[string]string{managementapi.HeaderIfMatch: invitationETag(expectedRevision)},
		http.StatusNoContent, nil)
	if err != nil {
		return 0, err
	}
	return parseInvitationETag(responseHeaders.Get(managementapi.HeaderETag))
}

func (provider *managementSessionProvider) AcceptInvitation(
	ctx context.Context,
	request dashboardauth.RouterInvitationAcceptance,
) (dashboardauth.RouterInvitationAcceptanceResult, error) {
	if provider == nil || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.PlannedSubject) ||
		strings.TrimSpace(request.InvitationToken) == "" || strings.TrimSpace(request.Email) == "" ||
		strings.TrimSpace(request.DisplayName) == "" ||
		!validSourceSessionExpiry(provider.now().UTC(), request.SessionExpiresAt) {
		return dashboardauth.RouterInvitationAcceptanceResult{}, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	now := provider.now().UTC()
	challenge, err := provider.challenge(ctx)
	if err != nil || challenge.Nonce == "" || challenge.ExchangeChallengeID == "" || !now.Before(challenge.ExpiresAt) {
		return dashboardauth.RouterInvitationAcceptanceResult{}, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	assertion, err := provider.invitationAssertion(request, challenge.Nonce, now)
	if err != nil {
		return dashboardauth.RouterInvitationAcceptanceResult{}, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	token := request.InvitationToken
	var exchanged managementapi.TokenExchangeResponse
	err = provider.request(ctx, http.MethodPost, managementBasePath+"/auth/token-exchange",
		managementapi.TokenExchangeRequest{
			IssuerID:            provider.issuerID,
			ExchangeChallengeID: challenge.ExchangeChallengeID, SubjectToken: assertion,
			SubjectTokenType: "router_local_assertion", InvitationToken: &token,
		}, http.StatusOK, &exchanged)
	if err != nil || exchanged.Onboarding == nil || exchanged.AccessToken == "" ||
		exchanged.TokenType != "Bearer" || exchanged.ExpiresIn <= 0 {
		return dashboardauth.RouterInvitationAcceptanceResult{}, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	onboarding := *exchanged.Onboarding
	if !canonicalUUID(onboarding.InvitationID) || !canonicalUUID(onboarding.PrincipalID) ||
		!canonicalUUID(onboarding.UserID) || !validOptionalInvitationKey(onboarding, now) {
		return dashboardauth.RouterInvitationAcceptanceResult{}, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	identity, err := provider.invitedIdentity(ctx, exchanged.AccessToken)
	if err != nil || identity.Principal.PrincipalID != onboarding.PrincipalID {
		return dashboardauth.RouterInvitationAcceptanceResult{}, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	role, err := invitedDashboardRole(identity, request.NamespaceID, onboarding)
	if err != nil {
		return dashboardauth.RouterInvitationAcceptanceResult{}, err
	}
	return dashboardauth.RouterInvitationAcceptanceResult{Onboarding: onboarding, DashboardRole: role}, nil
}

func validOptionalInvitationKey(onboarding managementapi.OnboardingResult, now time.Time) bool {
	hasID, hasSecret := onboarding.APIKeyID != "", onboarding.APIKey != ""
	if hasID != hasSecret {
		return false
	}
	return !hasID || (canonicalUUID(onboarding.APIKeyID) && now.Before(onboarding.DeliveryExpiresAt))
}

func (provider *managementSessionProvider) invitationAssertion(
	request dashboardauth.RouterInvitationAcceptance,
	nonce string,
	now time.Time,
) (string, error) {
	assertionExpiresAt := now.Add(assertionLifetime)
	if !validSourceSessionExpiry(now, request.SessionExpiresAt) ||
		request.SessionExpiresAt.UTC().Before(assertionExpiresAt) {
		return "", dashboardauth.ErrInvitationAuthorityUnavailable
	}
	return provider.signer.Sign(jwt.MapClaims{
		"iss": provider.issuerURL, "sub": request.PlannedSubject, "aud": managementAudience,
		"iat": now.Unix(), "exp": assertionExpiresAt.Unix(), "jti": uuid.NewString(),
		routerSourceExpiryClaim: request.SessionExpiresAt.UTC().Unix(),
		"nonce":                 nonce, "auth_time": now.Unix(), "aal": "aal1", "amr": []string{"pwd"},
		"email": request.Email, "email_verified": true, "name": request.DisplayName,
	})
}

func (provider *managementSessionProvider) invitedIdentity(ctx context.Context, accessToken string) (managementapi.Me, error) {
	var identity managementapi.Me
	err := provider.routerRequest(ctx, http.MethodGet, managementBasePath+"/me", nil,
		map[string]string{"Authorization": "Bearer " + accessToken}, http.StatusOK, &identity)
	return identity, err
}

func invitedDashboardRole(
	identity managementapi.Me,
	namespaceID string,
	onboarding managementapi.OnboardingResult,
) (string, error) {
	for _, scope := range identity.Namespaces {
		if scope.Namespace.NamespaceID != namespaceID || scope.User == nil || scope.User.UserID != onboarding.UserID {
			continue
		}
		if onboarding.TeamID != "" {
			found := false
			for _, team := range scope.Teams {
				if team.TeamID == onboarding.TeamID {
					found = true
					break
				}
			}
			if !found {
				return "", dashboardauth.ErrInvitationAuthorityUnavailable
			}
		}
		return dashboardauth.DashboardRoleFromManagementBindings(
			scope.RoleBindings, namespaceID, onboarding.UserID,
		)
	}
	return "", dashboardauth.ErrInvitationAuthorityUnavailable
}

func (provider *managementSessionProvider) authorizedInvitationRequest(
	ctx context.Context,
	actor dashboardauth.AuthContext,
	namespaceID, method, path string,
	body any,
	headers map[string]string,
	wantStatus int,
	response any,
) error {
	_, err := provider.authorizedInvitationRequestWithHeaders(
		ctx, actor, namespaceID, method, path, body, headers, wantStatus, response,
	)
	return err
}

func (provider *managementSessionProvider) authorizedInvitationRequestWithHeaders(
	ctx context.Context,
	actor dashboardauth.AuthContext,
	namespaceID, method, path string,
	body any,
	headers map[string]string,
	wantStatus int,
	response any,
) (http.Header, error) {
	if !canonicalUUID(namespaceID) {
		return nil, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	token, err := provider.ManagementAccessToken(ctx, actor)
	if err != nil {
		return nil, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	requestHeaders := map[string]string{
		"Authorization":                 "Bearer " + token,
		managementapi.HeaderNamespaceID: namespaceID,
	}
	for name, value := range headers {
		requestHeaders[name] = value
	}
	return provider.routerRequestWithHeaders(ctx, method, path, body, requestHeaders, wantStatus, response)
}

func (provider *managementSessionProvider) routerRequest(
	ctx context.Context,
	method, path string,
	body any,
	headers map[string]string,
	wantStatus int,
	response any,
) error {
	_, err := provider.routerRequestWithHeaders(ctx, method, path, body, headers, wantStatus, response)
	return err
}

func (provider *managementSessionProvider) routerRequestWithHeaders(
	ctx context.Context,
	method, path string,
	body any,
	headers map[string]string,
	wantStatus int,
	response any,
) (http.Header, error) {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, provider.routerURL+path, reader)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", managementMediaType)
	if body != nil {
		request.Header.Set("Content-Type", managementMediaType)
	}
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	result, err := provider.client.Do(request)
	if err != nil {
		return nil, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	defer result.Body.Close()
	if result.StatusCode != wantStatus {
		return nil, invitationAuthorityResponseError(result)
	}
	if response == nil {
		_, _ = io.Copy(io.Discard, io.LimitReader(result.Body, 64<<10))
		return result.Header.Clone(), nil
	}
	mediaType, _, mediaTypeErr := mime.ParseMediaType(result.Header.Get("Content-Type"))
	if mediaTypeErr != nil || mediaType != managementMediaType {
		return nil, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	decoder := json.NewDecoder(io.LimitReader(result.Body, 256<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(response); err != nil {
		return nil, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	return result.Header.Clone(), nil
}

func invitationAuthorityResponseError(result *http.Response) *dashboardauth.InvitationAuthorityError {
	diagnostic := &dashboardauth.InvitationAuthorityError{Status: result.StatusCode}
	if requestID := safeInvitationRequestID(result.Header.Get(managementapi.HeaderRequestID)); requestID != "" {
		diagnostic.RequestID = requestID
	}
	body, err := io.ReadAll(io.LimitReader(result.Body, maximumInvitationErrorBodyBytes+1))
	if err != nil || len(body) > maximumInvitationErrorBodyBytes {
		return diagnostic
	}
	var envelope invitationErrorDiagnostic
	if json.Unmarshal(body, &envelope) != nil {
		return diagnostic
	}
	diagnostic.Code = safeInvitationErrorCode(envelope.Error.Code)
	if diagnostic.RequestID == "" {
		diagnostic.RequestID = safeInvitationRequestID(envelope.Error.RequestID)
	}
	return diagnostic
}

func safeInvitationErrorCode(value string) string {
	switch value {
	case "authentication_unavailable", "authorization_unavailable", "conflict", "forbidden",
		"identity_already_onboarded", "invalid_cursor", "invalid_idempotency_key", "invalid_namespace",
		"invalid_precondition", "invalid_request",
		"invitation_expired", "invitation_service_unavailable", "not_found", "onboarding_defaults_changed",
		"precondition_required", "revision_conflict", "secret_result_expired", "unauthenticated",
		"unsupported_media_type":
		return value
	default:
		return ""
	}
}

func safeInvitationRequestID(value string) string {
	if canonicalUUID(value) {
		return value
	}
	return ""
}

func invitationETag(revision uint64) string {
	return `"invitation:` + strconv.FormatUint(revision, 10) + `"`
}

func parseInvitationETag(value string) (uint64, error) {
	if !strings.HasPrefix(value, `"invitation:`) || !strings.HasSuffix(value, `"`) {
		return 0, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	revision, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(value, `"invitation:`), `"`), 10, 64)
	if err != nil || revision == 0 {
		return 0, dashboardauth.ErrInvitationAuthorityUnavailable
	}
	return revision, nil
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

var _ dashboardauth.InvitationAuthority = (*managementSessionProvider)(nil)
