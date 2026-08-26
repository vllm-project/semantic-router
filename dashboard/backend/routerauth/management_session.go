package routerauth

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const (
	managementAudience               = "vllm-sr-management"
	managementMediaType              = "application/vnd.vllm-semantic-router.management.v1+json"
	managementBasePath               = "/management/v1"
	backchannelLogoutEvent           = "http://schemas.openid.net/event/backchannel-logout"
	assertionLifetime                = time.Minute
	maximumSourceSessionLifetime     = 30 * 24 * time.Hour
	routerSourceExpiryClaim          = "source_session_exp"
	tokenRefreshSkew                 = 30 * time.Second
	maxManagementSessionCacheEntries = 4096
)

var ErrManagementSessionUnavailable = errors.New("router Management session is unavailable")

// ManagementSessionProvider exchanges one authenticated Dashboard browser
// session for a Router-owned, short-lived Management session. Its Management
// cache holds only the returned token and opaque session ID; it never receives
// inference API keys, rate-limit state, or access-policy data.
type ManagementSessionProvider interface {
	ManagementAccessToken(context.Context, dashboardauth.AuthContext) (string, error)
}

type ManagementIdentityProvider interface {
	ManagementSessionProvider
	dashboardauth.DashboardSessionRetirer
	dashboardauth.InvitationAuthority
	dashboardauth.FirstAdminProvisioner
	ResolveEvaluationScope(
		context.Context,
		dashboardauth.AuthContext,
	) (userIDs []string, teamUsers map[string]string, err error)
	IssueEvaluationInferenceToken(
		context.Context,
		dashboardauth.AuthContext,
		string,
		string,
	) (string, error)
}

type ManagementSessionOptions struct {
	RouterURL          string
	IssuerURL          string
	IssuerID           string
	Signer             AssertionSigner
	Client             *http.Client
	Now                func() time.Time
	BootstrapTokenFile string
}

type managementSessionProvider struct {
	routerURL          string
	issuerURL          string
	issuerID           string
	signer             AssertionSigner
	client             *http.Client
	now                func() time.Time
	bootstrapTokenFile string

	mu          sync.Mutex
	cache       map[string]cachedManagementToken
	inflight    map[string]*managementTokenExchange
	delegations map[evaluationDelegationCacheKey]cachedEvaluationDelegation
}

type cachedManagementToken struct {
	accessToken         string
	managementSessionID string
	expiresAt           time.Time
}

type managementTokenExchange struct {
	done       chan struct{}
	credential cachedManagementToken
	err        error
}

type exchangeChallenge struct {
	ExchangeChallengeID string    `json:"exchangeChallengeId"`
	Nonce               string    `json:"nonce"`
	ExpiresAt           time.Time `json:"expiresAt"`
}

type managementTokenEnvelope struct {
	AccessToken         string `json:"accessToken"`
	TokenType           string `json:"tokenType"`
	ExpiresIn           int64  `json:"expiresIn"`
	ManagementSessionID string `json:"managementSessionId"`
}

// NewManagementSessionProvider validates the trust material eagerly. Missing
// or partial issuer configuration is an error so production never falls back
// to a broad Dashboard service identity.
func NewManagementSessionProvider(options ManagementSessionOptions) (ManagementIdentityProvider, error) {
	routerURL, err := canonicalBaseURL(options.RouterURL, false)
	if err != nil {
		return nil, fmt.Errorf("router Management URL: %w", err)
	}
	issuerURL, err := CanonicalIssuerURL(options.IssuerURL)
	if err != nil {
		return nil, fmt.Errorf("dashboard issuer: %w", err)
	}
	issuerID, err := uuid.Parse(options.IssuerID)
	if err != nil || issuerID.String() != options.IssuerID {
		return nil, errors.New("dashboard issuer ID must be a canonical UUID")
	}
	if options.Signer == nil {
		return nil, errors.New("dashboard assertion signer is required")
	}
	client := options.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &managementSessionProvider{
		routerURL: routerURL, issuerURL: issuerURL, issuerID: options.IssuerID,
		signer: options.Signer, client: client, now: now,
		bootstrapTokenFile: strings.TrimSpace(options.BootstrapTokenFile),
		cache:              make(map[string]cachedManagementToken), inflight: make(map[string]*managementTokenExchange),
		delegations: make(map[evaluationDelegationCacheKey]cachedEvaluationDelegation),
	}, nil
}

// CanonicalIssuerURL returns the one issuer origin used in discovery metadata
// and signed assertions so those two trust surfaces cannot drift.
func CanonicalIssuerURL(raw string) (string, error) {
	return canonicalBaseURL(raw, true)
}

func canonicalBaseURL(raw string, requireHTTPS bool) (string, error) {
	trimmed := strings.TrimSpace(raw)
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("a canonical origin is required")
	}
	if parsed.Path != "" && parsed.Path != "/" {
		return "", errors.New("origin must not contain a path")
	}
	if (requireHTTPS && parsed.Scheme != "https") || (!requireHTTPS && parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", errors.New("invalid URL scheme")
	}
	canonical := parsed.Scheme + "://" + strings.ToLower(parsed.Host)
	if trimmed != canonical {
		return "", errors.New("URL must be a canonical origin")
	}
	return canonical, nil
}

func (provider *managementSessionProvider) ManagementAccessToken(
	ctx context.Context,
	principal dashboardauth.AuthContext,
) (string, error) {
	credential, err := provider.managementCredential(ctx, principal)
	return credential.accessToken, err
}

func (provider *managementSessionProvider) managementCredential(
	ctx context.Context,
	principal dashboardauth.AuthContext,
) (cachedManagementToken, error) {
	if provider == nil || principal.UserID == "" || principal.SessionID == "" {
		return cachedManagementToken{}, ErrManagementSessionUnavailable
	}
	now := provider.now().UTC()
	if !validSourceSessionExpiry(now, principal.ExpiresAt) {
		return cachedManagementToken{}, ErrManagementSessionUnavailable
	}
	provider.mu.Lock()
	cached, ok := provider.cache[principal.SessionID]
	if ok && now.Add(tokenRefreshSkew).Before(cached.expiresAt) {
		provider.mu.Unlock()
		return cached, nil
	}
	delete(provider.cache, principal.SessionID)
	if pending := provider.inflight[principal.SessionID]; pending != nil {
		provider.mu.Unlock()
		select {
		case <-pending.done:
			return pending.credential, pending.err
		case <-ctx.Done():
			return cachedManagementToken{}, ErrManagementSessionUnavailable
		}
	}
	pending := &managementTokenExchange{done: make(chan struct{})}
	provider.inflight[principal.SessionID] = pending
	provider.mu.Unlock()

	token, managementSessionID, expiresAt, err := provider.issueManagementToken(ctx, principal, now)
	credential := cachedManagementToken{
		accessToken: token, managementSessionID: managementSessionID, expiresAt: expiresAt,
	}
	provider.mu.Lock()
	if err == nil {
		provider.pruneCacheLocked(now)
		provider.cache[principal.SessionID] = credential
	}
	pending.credential, pending.err = credential, err
	delete(provider.inflight, principal.SessionID)
	close(pending.done)
	provider.mu.Unlock()
	return credential, err
}

func (provider *managementSessionProvider) issueManagementToken(
	ctx context.Context,
	principal dashboardauth.AuthContext,
	now time.Time,
) (string, string, time.Time, error) {
	challenge, err := provider.challenge(ctx)
	if err != nil {
		return "", "", time.Time{}, fmt.Errorf("%w: create exchange challenge: %w", ErrManagementSessionUnavailable, err)
	}
	if challenge.Nonce == "" || challenge.ExchangeChallengeID == "" || !now.Before(challenge.ExpiresAt) {
		return "", "", time.Time{}, fmt.Errorf("%w: invalid exchange challenge", ErrManagementSessionUnavailable)
	}
	assertion, err := provider.assertion(principal, challenge.Nonce, now)
	if err != nil {
		return "", "", time.Time{}, fmt.Errorf("%w: sign source assertion", ErrManagementSessionUnavailable)
	}
	envelope, err := provider.exchange(ctx, challenge.ExchangeChallengeID, assertion)
	if err != nil {
		return "", "", time.Time{}, fmt.Errorf("%w: exchange source assertion: %w", ErrManagementSessionUnavailable, err)
	}
	managementSessionID, sessionIDErr := uuid.Parse(envelope.ManagementSessionID)
	if envelope.AccessToken == "" || envelope.TokenType != "Bearer" || envelope.ExpiresIn <= 0 ||
		sessionIDErr != nil || managementSessionID.String() != envelope.ManagementSessionID {
		return "", "", time.Time{}, fmt.Errorf("%w: invalid token exchange response", ErrManagementSessionUnavailable)
	}
	expiresAt := now.Add(time.Duration(envelope.ExpiresIn) * time.Second)
	return envelope.AccessToken, envelope.ManagementSessionID, expiresAt, nil
}

// RetireDashboardSession performs issuer-authenticated back-channel logout.
// The Router selects the derived Management session by the original Dashboard
// session ID, so logout works even after the short-lived Management bearer has
// expired or the Dashboard process has restarted.
func (provider *managementSessionProvider) RetireDashboardSession(
	ctx context.Context,
	dashboardSessionID string,
) error {
	if provider == nil || strings.TrimSpace(dashboardSessionID) == "" {
		return ErrManagementSessionUnavailable
	}
	now := provider.now().UTC()
	claims := jwt.MapClaims{
		"iss": provider.issuerURL,
		"aud": managementAudience,
		"iat": now.Unix(),
		"exp": now.Add(assertionLifetime).Unix(),
		"jti": uuid.NewString(),
		"sid": dashboardSessionID,
		"events": map[string]any{
			backchannelLogoutEvent: map[string]any{},
		},
	}
	logoutToken, err := provider.signer.Sign(claims)
	if err != nil {
		return ErrManagementSessionUnavailable
	}
	var response struct {
		Applied  bool `json:"applied"`
		Replayed bool `json:"replayed"`
	}
	if err := provider.request(
		ctx,
		http.MethodPost,
		managementBasePath+"/auth/backchannel-logout",
		map[string]string{"issuerId": provider.issuerID, "logoutToken": logoutToken},
		http.StatusOK,
		&response,
	); err != nil || (!response.Applied && !response.Replayed) {
		return ErrManagementSessionUnavailable
	}
	provider.mu.Lock()
	cached := provider.cache[dashboardSessionID]
	delete(provider.cache, dashboardSessionID)
	provider.clearEvaluationDelegationsLocked(cached.managementSessionID)
	provider.mu.Unlock()
	return nil
}

func (provider *managementSessionProvider) pruneCacheLocked(now time.Time) {
	for sessionID, cached := range provider.cache {
		if !now.Add(tokenRefreshSkew).Before(cached.expiresAt) {
			delete(provider.cache, sessionID)
		}
	}
	for len(provider.cache) >= maxManagementSessionCacheEntries {
		var oldestSession string
		var oldestExpiry time.Time
		for sessionID, cached := range provider.cache {
			if oldestSession == "" || cached.expiresAt.Before(oldestExpiry) {
				oldestSession, oldestExpiry = sessionID, cached.expiresAt
			}
		}
		if oldestSession == "" {
			return
		}
		delete(provider.cache, oldestSession)
	}
}

func (provider *managementSessionProvider) challenge(ctx context.Context) (exchangeChallenge, error) {
	var response exchangeChallenge
	err := provider.request(ctx, http.MethodPost, managementBasePath+"/auth/exchange-challenges", map[string]string{
		"issuerId": provider.issuerID,
	}, http.StatusCreated, &response)
	return response, err
}

func (provider *managementSessionProvider) assertion(
	principal dashboardauth.AuthContext,
	nonce string,
	now time.Time,
) (string, error) {
	assertionExpiresAt := now.Add(assertionLifetime)
	if !validSourceSessionExpiry(now, principal.ExpiresAt) ||
		principal.ExpiresAt.UTC().Before(assertionExpiresAt) {
		return "", ErrManagementSessionUnavailable
	}
	claims := jwt.MapClaims{
		"iss":                   provider.issuerURL,
		"sub":                   principal.UserID,
		"aud":                   managementAudience,
		"iat":                   now.Unix(),
		"exp":                   assertionExpiresAt.Unix(),
		routerSourceExpiryClaim: principal.ExpiresAt.UTC().Unix(),
		"jti":                   uuid.NewString(),
		"nonce":                 nonce,
		"sid":                   principal.SessionID,
		"auth_time":             principal.AuthenticatedAt.UTC().Unix(),
		"aal":                   "aal1",
		"amr":                   []string{"pwd"},
	}
	if principal.AuthenticatedAt.IsZero() {
		claims["auth_time"] = now.Unix()
	}
	if principal.Email != "" {
		claims["email"] = principal.Email
		claims["email_verified"] = true
	}
	if principal.Name != "" {
		claims["name"] = principal.Name
	}
	return provider.signer.Sign(claims)
}

func validSourceSessionExpiry(now time.Time, expiresAt time.Time) bool {
	expiresAt = expiresAt.UTC()
	return !expiresAt.IsZero() && now.Before(expiresAt) &&
		!expiresAt.After(now.Add(maximumSourceSessionLifetime))
}

func (provider *managementSessionProvider) exchange(
	ctx context.Context,
	challengeID string,
	assertion string,
) (managementTokenEnvelope, error) {
	var response managementTokenEnvelope
	err := provider.request(ctx, http.MethodPost, managementBasePath+"/auth/token-exchange", map[string]string{
		"issuerId":            provider.issuerID,
		"exchangeChallengeId": challengeID,
		"subjectToken":        assertion,
		"subjectTokenType":    "router_local_assertion",
	}, http.StatusOK, &response)
	return response, err
}

func (provider *managementSessionProvider) request(
	ctx context.Context,
	method string,
	path string,
	body any,
	expectedStatus int,
	response any,
) error {
	encoded, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, method, provider.routerURL+path, bytes.NewReader(encoded))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", managementMediaType)
	request.Header.Set("Accept", managementMediaType)
	result, err := provider.client.Do(request)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer result.Body.Close()
	mediaType, _, mediaTypeErr := mime.ParseMediaType(result.Header.Get("Content-Type"))
	if result.StatusCode != expectedStatus || mediaTypeErr != nil || mediaType != managementMediaType {
		_, _ = io.Copy(io.Discard, io.LimitReader(result.Body, 64<<10))
		return fmt.Errorf(
			"unexpected response status=%d content_type=%q",
			result.StatusCode,
			mediaType,
		)
	}
	decoder := json.NewDecoder(io.LimitReader(result.Body, 64<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(response); err != nil {
		return err
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return errors.New("router Management response contains trailing data")
	}
	return nil
}

// RewriteManagementAuthorization installs only the Router token exchanged for
// the current Dashboard principal. Browser cookies, local JWTs, proxy headers,
// and query credentials are removed before the upstream request is sent.
func RewriteManagementAuthorization(request *http.Request, provider ManagementSessionProvider) error {
	if request == nil {
		return errors.New("router Management request is nil")
	}
	principal, ok := dashboardauth.AuthFromContext(request)
	stripBrowserCredentials(request)
	if !ok || provider == nil {
		return ErrManagementSessionUnavailable
	}
	token, err := provider.ManagementAccessToken(request.Context(), principal)
	if err != nil || strings.TrimSpace(token) == "" || strings.ContainsAny(token, "\r\n\t ") {
		if err != nil {
			return fmt.Errorf("%w: %w", ErrManagementSessionUnavailable, err)
		}
		return ErrManagementSessionUnavailable
	}
	request.Header.Set("Authorization", "Bearer "+token)
	return nil
}

// stripBrowserCredentials prevents a Dashboard session, inference credential,
// or caller-supplied identity hint from crossing the Management BFF boundary.
// The exchanged Router Management token is the only outbound authority.
func stripBrowserCredentials(request *http.Request) {
	if request == nil {
		return
	}
	request.Header.Del("Authorization")
	request.Header.Del("Proxy-Authorization")
	request.Header.Del("Cookie")
	request.Header.Del("X-API-Key")
	request.Header.Del("X-VLLM-SR-Principal")
	request.Header.Del("X-VLLM-SR-User")
	request.Header.Del("X-VLLM-SR-Team")
	if request.URL == nil {
		return
	}
	query := request.URL.Query()
	query.Del("authToken")
	query.Del("access_token")
	query.Del("api_key")
	request.URL.RawQuery = query.Encode()
}
