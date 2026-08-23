package managementserver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

type AuthenticationService interface {
	Ready(context.Context) error
	CreateChallenge(context.Context, string, string) (managementauth.ExchangeChallenge, error)
	Exchange(context.Context, string, string, managementauth.SubjectTokenType, string, string) (managementauth.IdentityExchangeResult, error)
	ServiceToken(context.Context, string) (managementauth.IssuedToken, error)
	MTLSToken(context.Context, managementauth.VerifiedMTLSEvidence) (managementauth.IssuedToken, error)
}

type BootstrapService interface {
	Ready(context.Context) error
	Bootstrap(context.Context, managementidentity.BootstrapRequest, string) (managementidentity.BootstrapResult, error)
}

type RecoveryService interface {
	Ready(context.Context) error
	Recover(context.Context, managementidentity.RecoveryRequest, string) (managementidentity.RecoveryResult, error)
}

type IdentityAuthRoutesOptions struct {
	Service                AuthenticationService
	Bootstrap              BootstrapService
	Recovery               RecoveryService
	AllowPlaintextForTests bool
}

type IdentityAuthRoutes struct {
	service        AuthenticationService
	bootstrap      BootstrapService
	recovery       RecoveryService
	allowPlaintext bool
}

func NewIdentityAuthRoutes(options IdentityAuthRoutesOptions) (*IdentityAuthRoutes, error) {
	if options.Service == nil || options.Bootstrap == nil {
		return nil, errors.New("management authentication and bootstrap services are required")
	}
	return &IdentityAuthRoutes{
		service: options.Service, bootstrap: options.Bootstrap, recovery: options.Recovery,
		allowPlaintext: options.AllowPlaintextForTests,
	}, nil
}

func (routes *IdentityAuthRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Management authentication routes and mux are required")
	}
	mux.Handle("POST "+managementapi.BasePath+"/auth/bootstrap", routes)
	if routes.recovery != nil {
		mux.Handle("POST "+managementapi.BasePath+"/auth/recovery", routes)
	}
	mux.Handle("POST "+managementapi.BasePath+"/auth/exchange-challenges", routes)
	mux.Handle("POST "+managementapi.BasePath+"/auth/token-exchange", routes)
	mux.Handle("POST "+managementapi.BasePath+"/auth/service-token", routes)
}

func (routes *IdentityAuthRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil || routes.bootstrap == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	if err := routes.bootstrap.Ready(ctx); err != nil {
		return err
	}
	if routes.recovery != nil {
		if err := routes.recovery.Ready(ctx); err != nil {
			return err
		}
	}
	return routes.service.Ready(ctx)
}

func (routes *IdentityAuthRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path ||
		request.Method != http.MethodPost || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.TLS == nil && !routes.allowPlaintext {
		writeProviderError(response, http.StatusBadRequest, "tls_required", "TLS is required.", requestID)
		return
	}
	switch request.URL.Path {
	case managementapi.BasePath + "/auth/bootstrap":
		routes.bootstrapInstallation(response, request, requestID)
	case managementapi.BasePath + "/auth/recovery":
		if routes.recovery == nil {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
			return
		}
		routes.recoverAdministrator(response, request, requestID)
	case managementapi.BasePath + "/auth/exchange-challenges":
		routes.challenge(response, request, requestID)
	case managementapi.BasePath + "/auth/token-exchange":
		routes.exchange(response, request, requestID)
	case managementapi.BasePath + "/auth/service-token":
		routes.serviceToken(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *IdentityAuthRoutes) recoverAdministrator(response http.ResponseWriter, request *http.Request, requestID string) {
	address := directRequestIP(request)
	if !address.IsValid() || !address.IsLoopback() {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	credential, ok := recoveryCredential(response, request, requestID)
	if !ok {
		return
	}
	defer zeroString(&credential)
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.RecoveryRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	canonical, err := json.Marshal(body)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Recovery request is invalid.", requestID)
		return
	}
	defer zeroIdentityBytes(canonical)
	result, err := routes.recovery.Recover(request.Context(), managementidentity.RecoveryRequest{
		PrincipalID: body.PrincipalID, Reason: body.Reason, RequestID: requestID,
		IdempotencyKey: string(idempotencyKey), CanonicalRequest: canonical,
	}, credential)
	if err != nil {
		writeRecoveryError(response, err, requestID)
		return
	}
	if result.Replayed {
		response.Header().Set(managementapi.HeaderIdempotencyReplayed, "true")
	}
	writeProviderJSON(response, result.ResponseStatus, managementapi.RecoveryResponse{
		PrincipalID: result.PrincipalID, RoleBindingID: result.RoleBindingID,
		RecoveryDisableRequired: true,
	}, requestID)
}

func recoveryCredential(response http.ResponseWriter, request *http.Request, requestID string) (string, bool) {
	values := request.Header.Values("Authorization")
	const prefix = "VSR-Recovery "
	if len(values) != 1 || !strings.HasPrefix(values[0], prefix) {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Recovery authentication failed.", requestID)
		return "", false
	}
	credential := strings.TrimPrefix(values[0], prefix)
	if credential == "" || strings.TrimSpace(credential) != credential || strings.ContainsAny(credential, "\r\n\t ") {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Recovery authentication failed.", requestID)
		return "", false
	}
	return credential, true
}

func writeRecoveryError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, managementidentity.ErrInvalidRecoveryRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Recovery request is invalid.", requestID)
	case errors.Is(err, managementidentity.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Management principal not found.", requestID)
	case errors.Is(err, managementidentity.ErrRecoveryConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency key was used with a different recovery request.", requestID)
	case errors.Is(err, managementidentity.ErrRecoveryConsumed):
		writeProviderError(response, http.StatusConflict, "recovery_consumed", "Recovery credential has already been used.", requestID)
	case errors.Is(err, managementidentity.ErrRecoveryUnavailable):
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Recovery authentication failed.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "recovery_unavailable", "Recovery is unavailable.", requestID)
	}
}

func (routes *IdentityAuthRoutes) bootstrapInstallation(response http.ResponseWriter, request *http.Request, requestID string) {
	credential, ok := bootstrapCredential(response, request, requestID)
	if !ok {
		return
	}
	defer zeroString(&credential)
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.BootstrapRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	domain, canonical, ok := bootstrapDomainRequest(response, body, string(idempotencyKey), requestID)
	if !ok {
		return
	}
	defer zeroIdentityBytes(canonical)
	result, err := routes.bootstrap.Bootstrap(request.Context(), domain, credential)
	if err != nil {
		writeBootstrapError(response, err, requestID)
		return
	}
	if result.Replayed {
		response.Header().Set(managementapi.HeaderIdempotencyReplayed, "true")
	}
	payload := managementapi.BootstrapResponse{
		PrincipalID: result.PrincipalID, RoleBindingID: result.RoleBindingID,
		ServiceAccountID:     result.ServiceAccountID,
		FinalizationRequired: result.FinalizationRequired,
	}
	if result.ServiceCredential != "" {
		expiresAt := result.ServiceCredentialExpiresAt
		payload.ServiceCredential = &managementapi.SecretEnvelope{
			ResourceID: result.ServiceCredentialID,
			Kind:       managementapi.SecretKindServiceCredential,
			Secret:     result.ServiceCredential,
			ExpiresAt:  &expiresAt,
		}
		defer zeroString(&payload.ServiceCredential.Secret)
	}
	writeProviderJSON(response, result.ResponseStatus, payload, requestID)
}

func bootstrapCredential(response http.ResponseWriter, request *http.Request, requestID string) (string, bool) {
	values := request.Header.Values("Authorization")
	const prefix = "VSR-Bootstrap "
	if len(values) != 1 || !strings.HasPrefix(values[0], prefix) {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Bootstrap authentication failed.", requestID)
		return "", false
	}
	credential := strings.TrimPrefix(values[0], prefix)
	if credential == "" || strings.TrimSpace(credential) != credential || strings.ContainsAny(credential, "\r\n\t ") {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Bootstrap authentication failed.", requestID)
		return "", false
	}
	return credential, true
}

func bootstrapDomainRequest(response http.ResponseWriter, body managementapi.BootstrapRequest, idempotencyKey, requestID string) (managementidentity.BootstrapRequest, []byte, bool) {
	domain := managementidentity.BootstrapRequest{
		Kind:           managementidentity.BootstrapKind(body.Kind),
		DisplayName:    body.DisplayName,
		IdempotencyKey: idempotencyKey,
	}
	switch domain.Kind {
	case managementidentity.BootstrapExternalPrincipal:
		if body.External == nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "External identity details are required.", requestID)
			return managementidentity.BootstrapRequest{}, nil, false
		}
		domain.IssuerID = body.External.IssuerID
		domain.Issuer = body.External.Issuer
		domain.Subject = body.External.Subject
		domain.DiscoveryURL = body.External.DiscoveryURL
		domain.Audience = body.External.Audience
	case managementidentity.BootstrapServiceAccount:
		if body.External != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "External identity details are not accepted for a service account.", requestID)
			return managementidentity.BootstrapRequest{}, nil, false
		}
	default:
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bootstrap kind is invalid.", requestID)
		return managementidentity.BootstrapRequest{}, nil, false
	}
	canonical, err := json.Marshal(body)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bootstrap request is invalid.", requestID)
		return managementidentity.BootstrapRequest{}, nil, false
	}
	domain.CanonicalRequest = canonical
	return domain, canonical, true
}

func writeBootstrapError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, managementidentity.ErrInvalidBootstrapRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Bootstrap request is invalid.", requestID)
	case errors.Is(err, managementidentity.ErrBootstrapConflict), errors.Is(err, managementidentity.ErrBootstrapConsumed):
		writeProviderError(response, http.StatusConflict, "bootstrap_conflict", "Bootstrap has already been completed.", requestID)
	case errors.Is(err, managementidentity.ErrBootstrapResultExpired):
		writeProviderError(response, http.StatusGone, "bootstrap_result_expired", "The one-time bootstrap result has expired.", requestID)
	case errors.Is(err, managementidentity.ErrBootstrapUnavailable):
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Bootstrap authentication failed.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "bootstrap_unavailable", "Bootstrap is unavailable.", requestID)
	}
}

func (routes *IdentityAuthRoutes) challenge(response http.ResponseWriter, request *http.Request, requestID string) {
	var body managementapi.ExchangeChallengeRequest
	if !decodeIdentityBody(response, request, requestID, &body) || !canonicalUUID(body.IssuerID) {
		return
	}
	address := directRequestIP(request)
	if !address.IsValid() {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "A direct client address is required.", requestID)
		return
	}
	challenge, err := routes.service.CreateChallenge(request.Context(), body.IssuerID, address.String())
	if err != nil {
		writeAuthenticationError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusCreated, managementapi.ExchangeChallengeResponse{
		ExchangeChallengeID: challenge.ID, Nonce: challenge.Nonce, ExpiresAt: challenge.ExpiresAt,
	}, requestID)
}

func (routes *IdentityAuthRoutes) exchange(response http.ResponseWriter, request *http.Request, requestID string) {
	var body managementapi.TokenExchangeRequest
	if !decodeIdentityBody(response, request, requestID, &body) || !canonicalUUID(body.IssuerID) ||
		!canonicalUUID(body.ExchangeChallengeID) {
		return
	}
	tokenType := managementauth.SubjectTokenType(body.SubjectTokenType)
	if tokenType != managementauth.SubjectTokenOIDCIDToken && tokenType != managementauth.SubjectTokenRouterAssertion {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Subject token type is invalid.", requestID)
		return
	}
	token := body.SubjectToken
	body.SubjectToken = ""
	defer zeroString(&token)
	invitationToken := ""
	if body.InvitationToken != nil {
		invitationToken = *body.InvitationToken
		*body.InvitationToken = ""
		if invitationToken == "" || len(invitationToken) > 512 || strings.TrimSpace(invitationToken) != invitationToken ||
			strings.ContainsAny(invitationToken, "\r\n\t ") {
			zeroString(&invitationToken)
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Invitation token is invalid.", requestID)
			return
		}
		defer zeroString(&invitationToken)
	}
	result, err := routes.service.Exchange(request.Context(), body.IssuerID, body.ExchangeChallengeID,
		tokenType, token, invitationToken)
	if err != nil {
		writeAuthenticationError(response, err, requestID)
		return
	}
	writeExchangeResult(response, result, requestID)
}

func (routes *IdentityAuthRoutes) serviceToken(response http.ResponseWriter, request *http.Request, requestID string) {
	if request.ContentLength != 0 || len(request.TransferEncoding) != 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Service token does not accept a request body.", requestID)
		return
	}
	values := request.Header.Values("Authorization")
	const prefix = "VSR-Service "
	var (
		issued managementauth.IssuedToken
		err    error
	)
	switch len(values) {
	case 0:
		evidence, evidenceErr := managementauth.VerifiedMTLSEvidenceFromConnection(request.TLS)
		if evidenceErr != nil {
			writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
			return
		}
		issued, err = routes.service.MTLSToken(request.Context(), evidence)
	case 1:
		if !strings.HasPrefix(values[0], prefix) {
			writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
			return
		}
		credential := strings.TrimPrefix(values[0], prefix)
		if credential == "" || strings.TrimSpace(credential) != credential || strings.ContainsAny(credential, "\r\n\t ") {
			writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
			return
		}
		defer zeroString(&credential)
		issued, err = routes.service.ServiceToken(request.Context(), credential)
	default:
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return
	}
	if err != nil {
		writeAuthenticationError(response, err, requestID)
		return
	}
	writeIssuedToken(response, issued, requestID)
}

func writeIssuedToken(response http.ResponseWriter, issued managementauth.IssuedToken, requestID string) {
	writeProviderJSON(response, http.StatusOK, managementapi.ManagementTokenEnvelope{
		AccessToken: issued.AccessToken, TokenType: issued.TokenType,
		ExpiresIn: int64(issued.ExpiresIn / time.Second), ManagementSessionID: issued.ManagementSessionID,
	}, requestID)
}

func writeExchangeResult(response http.ResponseWriter, result managementauth.IdentityExchangeResult, requestID string) {
	payload := managementapi.TokenExchangeResponse{ManagementTokenEnvelope: managementapi.ManagementTokenEnvelope{
		AccessToken: result.Issued.AccessToken, TokenType: result.Issued.TokenType,
		ExpiresIn: int64(result.Issued.ExpiresIn / time.Second), ManagementSessionID: result.Issued.ManagementSessionID,
	}}
	if result.Onboarding != nil {
		payload.Onboarding = &managementapi.OnboardingResult{
			InvitationID: result.Onboarding.InvitationID, PrincipalID: result.Onboarding.PrincipalID,
			UserID: result.Onboarding.UserID, TeamID: result.Onboarding.TeamID,
			APIKeyID: result.Onboarding.APIKeyID, APIKey: result.Onboarding.APIKey,
			DeliveryExpiresAt: result.Onboarding.DeliveryExpiresAt,
		}
		defer zeroString(&payload.Onboarding.APIKey)
	}
	if result.Replayed {
		response.Header().Set(managementapi.HeaderIdempotencyReplayed, "true")
	}
	writeProviderJSON(response, http.StatusOK, payload, requestID)
}

func writeAuthenticationError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, managementauth.ErrAuthenticationDenied), errors.Is(err, managementauth.ErrSessionLimitExceeded):
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication failed.", requestID)
	case errors.Is(err, managementauth.ErrInvitationExpired):
		writeProviderError(response, http.StatusGone, "invitation_expired", "The invitation expired.", requestID)
	case errors.Is(err, managementauth.ErrInvitationResultExpired):
		writeProviderError(response, http.StatusGone, "invitation_result_expired", "The one-time onboarding result expired.", requestID)
	case errors.Is(err, managementauth.ErrInvitationConflict):
		writeProviderError(response, http.StatusConflict, "invitation_conflict", "The invitation no longer matches current onboarding state.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
	}
}

func decodeIdentityBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumCredentialBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumCredentialBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		} else {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		}
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		return false
	}
	return true
}

func zeroIdentityBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

var _ RouteRegistrar = (*IdentityAuthRoutes)(nil)
