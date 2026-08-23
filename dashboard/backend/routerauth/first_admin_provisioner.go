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
	"os"
	"path/filepath"
	"strings"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	defaultNamespaceName     = "Default"
	defaultBillingCurrency   = "USD"
	platformAdminRoleName    = "platform_admin"
	maximumBootstrapTokenLen = 8 << 10
	maximumBootstrapPages    = 64
)

var errFirstAdminProvisioning = dashboardauth.ErrFirstAdminProvisioningUnavailable

type namespaceView struct {
	NamespaceID      string    `json:"namespaceId"`
	Name             string    `json:"name"`
	QuotaPartitionID string    `json:"quotaPartitionId"`
	BillingCurrency  string    `json:"billingCurrency"`
	Status           string    `json:"status"`
	Revision         uint64    `json:"revision"`
	RuntimeEpoch     uint64    `json:"runtimeEpoch"`
	CreatedAt        time.Time `json:"createdAt"`
	UpdatedAt        time.Time `json:"updatedAt"`
}

type namespacePage struct {
	Data []namespaceView        `json:"data"`
	Page managementapi.PageInfo `json:"page"`
}

func (provider *managementSessionProvider) ProvisionFirstAdmin(
	ctx context.Context,
	identity dashboardauth.FirstAdminIdentity,
) error {
	if provider == nil || !canonicalUUID(identity.UserID) || !canonicalUUID(identity.SessionID) ||
		strings.TrimSpace(identity.Email) == "" || strings.TrimSpace(identity.DisplayName) == "" {
		return errFirstAdminProvisioning
	}
	actor := dashboardauth.AuthContext{
		UserID: identity.UserID, SessionID: identity.SessionID,
		Email: identity.Email, Name: identity.DisplayName,
		Role: dashboardauth.RoleAdmin, AuthenticatedAt: identity.AuthenticatedAt.UTC(),
		ExpiresAt: identity.ExpiresAt.UTC(),
	}
	accessToken, err := provider.ManagementAccessToken(ctx, actor)
	var bootstrapFile os.FileInfo
	if err != nil {
		bootstrapFile, err = observeBootstrapToken(provider.bootstrapTokenFile)
		if err != nil {
			return err
		}
		if bootstrapErr := provider.bootstrapExternalPrincipal(ctx, identity); bootstrapErr != nil {
			return bootstrapErr
		}
		accessToken, err = provider.ManagementAccessToken(ctx, actor)
		if err != nil {
			return errFirstAdminProvisioning
		}
	} else if bootstrapFile, err = observeBootstrapTokenIfPresent(provider.bootstrapTokenFile); err != nil {
		return err
	}

	principalID, err := provider.currentPrincipalID(ctx, accessToken)
	if err != nil {
		return err
	}
	namespaceID, err := provider.ensureDefaultNamespace(ctx, accessToken, identity.UserID)
	if err != nil {
		return err
	}
	role, err := provider.platformAdminRole(ctx, accessToken, namespaceID)
	if err != nil {
		return err
	}
	if bindingErr := provider.ensurePlatformAdminBinding(ctx, accessToken, namespaceID, principalID, role, identity.UserID); bindingErr != nil {
		return bindingErr
	}
	userID, err := provider.ensureRouterUser(ctx, accessToken, namespaceID, identity)
	if err != nil {
		return err
	}
	if err := provider.ensurePrincipalUserLink(ctx, accessToken, namespaceID, principalID, userID, identity.UserID); err != nil {
		return err
	}
	if err := provider.verifyFirstAdmin(ctx, accessToken, namespaceID, principalID, userID, role.RoleID); err != nil {
		return err
	}
	if bootstrapFile == nil {
		// An already trusted principal proves that Router bootstrap completed.
		// This is the normal retry path after an external Kubernetes operator
		// removed the projected one-time credential and rolled the Router.
		return nil
	}
	return finalizeBootstrapToken(provider.bootstrapTokenFile, bootstrapFile)
}

func (provider *managementSessionProvider) bootstrapExternalPrincipal(
	ctx context.Context,
	identity dashboardauth.FirstAdminIdentity,
) error {
	token, err := readBootstrapToken(provider.bootstrapTokenFile)
	if err != nil {
		return errFirstAdminProvisioning
	}
	defer zeroStringValue(&token)
	request := managementapi.BootstrapRequest{
		Kind: "external_principal", DisplayName: identity.DisplayName,
		External: &managementapi.BootstrapExternalIdentity{
			IssuerID: provider.issuerID, Issuer: provider.issuerURL, Subject: identity.UserID,
			DiscoveryURL: provider.issuerURL + "/.well-known/openid-configuration",
			Audience:     managementAudience,
		},
	}
	var result managementapi.BootstrapResponse
	err = provider.managementRequest(ctx, http.MethodPost, managementBasePath+"/auth/bootstrap", request,
		map[string]string{
			"Authorization":                    "VSR-Bootstrap " + token,
			managementapi.HeaderIdempotencyKey: installationKey("bootstrap", identity.UserID),
		}, []int{http.StatusCreated}, &result)
	if err != nil || !canonicalUUID(result.PrincipalID) || !result.FinalizationRequired {
		return errFirstAdminProvisioning
	}
	return nil
}

func (provider *managementSessionProvider) currentPrincipalID(ctx context.Context, token string) (string, error) {
	var identity managementapi.Me
	if err := provider.authorizedManagementRequest(ctx, token, "", http.MethodGet,
		managementBasePath+"/me", nil, nil, []int{http.StatusOK}, &identity); err != nil ||
		!canonicalUUID(identity.Principal.PrincipalID) {
		return "", errFirstAdminProvisioning
	}
	return identity.Principal.PrincipalID, nil
}

func (provider *managementSessionProvider) ensureDefaultNamespace(
	ctx context.Context,
	token string,
	seed string,
) (string, error) {
	var page namespacePage
	if err := provider.authorizedManagementRequest(ctx, token, "", http.MethodGet,
		managementBasePath+"/namespaces?status=active&pageSize=200", nil, nil,
		[]int{http.StatusOK}, &page); err != nil {
		return "", err
	}
	var found string
	for _, namespace := range page.Data {
		if namespace.Name != defaultNamespaceName || namespace.Status != "active" || !canonicalUUID(namespace.NamespaceID) {
			continue
		}
		if found != "" && found != namespace.NamespaceID {
			return "", errFirstAdminProvisioning
		}
		found = namespace.NamespaceID
	}
	if found != "" {
		return found, nil
	}
	var receipt managementapi.MutationReceipt
	request := struct {
		Name            string `json:"name"`
		BillingCurrency string `json:"billingCurrency"`
		Reason          string `json:"reason"`
	}{defaultNamespaceName, defaultBillingCurrency, "Create the initial Dashboard workspace"}
	if err := provider.authorizedManagementRequest(ctx, token, "", http.MethodPost,
		managementBasePath+"/namespaces", request,
		map[string]string{managementapi.HeaderIdempotencyKey: installationKey("namespace", seed)},
		[]int{http.StatusCreated}, &receipt); err != nil {
		return "", err
	}
	if receipt.Resource == nil || receipt.Resource.Kind != "namespace" || !canonicalUUID(receipt.Resource.ID) {
		return "", errFirstAdminProvisioning
	}
	return receipt.Resource.ID, nil
}

func (provider *managementSessionProvider) ensureRouterUser(
	ctx context.Context,
	token string,
	namespaceID string,
	identity dashboardauth.FirstAdminIdentity,
) (string, error) {
	path := managementBasePath + "/users?pageSize=200"
	for pageNumber := 0; pageNumber < maximumBootstrapPages; pageNumber++ {
		var page managementapi.UserPage
		if err := provider.authorizedManagementRequest(ctx, token, namespaceID, http.MethodGet,
			path, nil, nil, []int{http.StatusOK}, &page); err != nil {
			return "", err
		}
		for _, user := range page.Data {
			if strings.EqualFold(user.Email, identity.Email) && user.Status == "active" {
				if !canonicalUUID(user.UserID) {
					return "", errFirstAdminProvisioning
				}
				return user.UserID, nil
			}
		}
		if !page.Page.HasMore || page.Page.NextCursor == "" {
			break
		}
		path = managementBasePath + "/users?pageSize=200&cursor=" + url.QueryEscape(page.Page.NextCursor)
	}
	var receipt managementapi.MutationReceipt
	if err := provider.authorizedManagementRequest(ctx, token, namespaceID, http.MethodPost,
		managementBasePath+"/users", managementapi.UserCreateRequest{
			Email: identity.Email, DisplayName: identity.DisplayName,
		}, map[string]string{managementapi.HeaderIdempotencyKey: installationKey("user", identity.UserID)},
		[]int{http.StatusCreated}, &receipt); err != nil {
		return "", err
	}
	if receipt.Resource == nil || receipt.Resource.Kind != "user" || !canonicalUUID(receipt.Resource.ID) {
		return "", errFirstAdminProvisioning
	}
	return receipt.Resource.ID, nil
}

func (provider *managementSessionProvider) platformAdminRole(
	ctx context.Context,
	token string,
	namespaceID string,
) (managementapi.ManagementRole, error) {
	var page managementapi.Page[managementapi.ManagementRole]
	path := managementBasePath + "/management-roles?namespaceId=" + url.QueryEscape(namespaceID) + "&pageSize=200"
	if err := provider.authorizedManagementRequest(ctx, token, "", http.MethodGet, path,
		nil, nil, []int{http.StatusOK}, &page); err != nil {
		return managementapi.ManagementRole{}, err
	}
	for _, role := range page.Data {
		if role.Name == platformAdminRoleName && role.BuiltIn && role.Status == "active" && canonicalUUID(role.RoleID) {
			return role, nil
		}
	}
	return managementapi.ManagementRole{}, errFirstAdminProvisioning
}

func (provider *managementSessionProvider) ensurePrincipalUserLink(
	ctx context.Context,
	token, namespaceID, principalID, userID, seed string,
) error {
	path := managementBasePath + "/namespaces/" + url.PathEscape(namespaceID) +
		"/principal-user-links/" + url.PathEscape(principalID)
	var detail managementapi.PrincipalUserLinkDetail
	err := provider.authorizedManagementRequest(ctx, token, namespaceID, http.MethodGet,
		path, nil, nil, []int{http.StatusOK}, &detail)
	if err == nil {
		if detail.Data.UserID == userID && detail.Data.PrincipalID == principalID {
			return nil
		}
		return errFirstAdminProvisioning
	}
	var response managementapi.PrincipalUserLinkDetail
	return provider.authorizedManagementRequest(ctx, token, namespaceID, http.MethodPut,
		path, managementapi.PrincipalUserLinkPutRequest{UserID: userID},
		map[string]string{managementapi.HeaderIdempotencyKey: installationKey("principal-link", seed)},
		[]int{http.StatusOK, http.StatusCreated}, &response)
}

func (provider *managementSessionProvider) ensurePlatformAdminBinding(
	ctx context.Context,
	token, namespaceID, principalID string,
	role managementapi.ManagementRole,
	seed string,
) error {
	var page managementapi.Page[managementapi.ManagementRoleBinding]
	path := managementBasePath + "/role-bindings?principalId=" + url.QueryEscape(principalID) + "&pageSize=200"
	if err := provider.authorizedManagementRequest(ctx, token, "", http.MethodGet,
		path, nil, nil, []int{http.StatusOK}, &page); err != nil {
		return err
	}
	for _, binding := range page.Data {
		if binding.RoleID == role.RoleID && binding.Status == "active" &&
			binding.Scope.Kind == "namespace" && binding.Scope.NamespaceID == namespaceID {
			return nil
		}
	}
	var receipt managementapi.MutationReceipt
	return provider.authorizedManagementRequest(ctx, token, namespaceID, http.MethodPost,
		managementBasePath+"/role-bindings", managementapi.ManagementRoleBindingCreateRequest{
			PrincipalID: principalID, RoleID: role.RoleID,
			Scope:             managementapi.ManagementScope{Kind: "namespace", NamespaceID: namespaceID},
			DelegationCeiling: append([]string(nil), role.Permissions...),
		}, map[string]string{managementapi.HeaderIdempotencyKey: installationKey("platform-admin", seed)},
		[]int{http.StatusCreated}, &receipt)
}

func (provider *managementSessionProvider) verifyFirstAdmin(
	ctx context.Context,
	token, namespaceID, principalID, userID, roleID string,
) error {
	var identity managementapi.Me
	if err := provider.authorizedManagementRequest(ctx, token, "", http.MethodGet,
		managementBasePath+"/me", nil, nil, []int{http.StatusOK}, &identity); err != nil ||
		identity.Principal.PrincipalID != principalID {
		return errFirstAdminProvisioning
	}
	for _, scope := range identity.Namespaces {
		if scope.Namespace.NamespaceID != namespaceID || scope.User == nil || scope.User.UserID != userID {
			continue
		}
		for _, binding := range scope.RoleBindings {
			if binding.RoleID == roleID && binding.Status == "active" &&
				binding.Scope.Kind == "namespace" && binding.Scope.NamespaceID == namespaceID {
				return nil
			}
		}
	}
	return errFirstAdminProvisioning
}

func (provider *managementSessionProvider) authorizedManagementRequest(
	ctx context.Context,
	token, namespaceID, method, path string,
	body any,
	headers map[string]string,
	wantStatuses []int,
	response any,
) error {
	requestHeaders := map[string]string{"Authorization": "Bearer " + token}
	if namespaceID != "" {
		requestHeaders[managementapi.HeaderNamespaceID] = namespaceID
	}
	for name, value := range headers {
		requestHeaders[name] = value
	}
	return provider.managementRequest(ctx, method, path, body, requestHeaders, wantStatuses, response)
}

func (provider *managementSessionProvider) managementRequest(
	ctx context.Context,
	method, path string,
	body any,
	headers map[string]string,
	wantStatuses []int,
	response any,
) error {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return errFirstAdminProvisioning
		}
		reader = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, provider.routerURL+path, reader)
	if err != nil {
		return errFirstAdminProvisioning
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
		return errFirstAdminProvisioning
	}
	defer result.Body.Close()
	accepted := false
	for _, status := range wantStatuses {
		accepted = accepted || result.StatusCode == status
	}
	if !accepted {
		_, _ = io.Copy(io.Discard, io.LimitReader(result.Body, 64<<10))
		return errFirstAdminProvisioning
	}
	if response == nil {
		_, _ = io.Copy(io.Discard, io.LimitReader(result.Body, 64<<10))
		return nil
	}
	mediaType, _, err := mime.ParseMediaType(result.Header.Get("Content-Type"))
	if err != nil || mediaType != managementMediaType {
		return errFirstAdminProvisioning
	}
	decoder := json.NewDecoder(io.LimitReader(result.Body, 256<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(response); err != nil {
		return errFirstAdminProvisioning
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errFirstAdminProvisioning
	}
	return nil
}

func observeBootstrapToken(path string) (os.FileInfo, error) {
	if path == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, errFirstAdminProvisioning
	}
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0o077 != 0 {
		return nil, errFirstAdminProvisioning
	}
	return info, nil
}

func readBootstrapToken(path string) (string, error) {
	info, err := observeBootstrapToken(path)
	if err != nil || info == nil {
		return "", errFirstAdminProvisioning
	}
	file, err := os.Open(path)
	if err != nil {
		return "", errFirstAdminProvisioning
	}
	defer file.Close()
	payload, err := io.ReadAll(io.LimitReader(file, maximumBootstrapTokenLen+1))
	if err != nil || len(payload) == 0 || len(payload) > maximumBootstrapTokenLen {
		return "", errFirstAdminProvisioning
	}
	defer zeroByteSlice(payload)
	token := strings.TrimSpace(string(payload))
	if len(token) < 32 || strings.ContainsAny(token, "\r\n\t ") {
		return "", errFirstAdminProvisioning
	}
	openedInfo, err := file.Stat()
	if err != nil || !os.SameFile(info, openedInfo) {
		return "", errFirstAdminProvisioning
	}
	return token, nil
}

func finalizeBootstrapToken(path string, observed os.FileInfo) error {
	if path == "" {
		return errFirstAdminProvisioning
	}
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		// Provisioning has already been verified through /me. A concurrent
		// finalizer removing the same one-time credential is therefore the
		// desired terminal state, regardless of whether this process observed
		// the file at the start of the saga.
		return nil
	}
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0o077 != 0 {
		return errFirstAdminProvisioning
	}
	if observed == nil || !os.SameFile(observed, info) {
		return errFirstAdminProvisioning
	}
	if removeErr := os.Remove(path); removeErr != nil {
		return errFirstAdminProvisioning
	}
	directory, err := os.Open(filepath.Dir(path))
	if err != nil {
		return errFirstAdminProvisioning
	}
	defer directory.Close()
	if err := directory.Sync(); err != nil {
		return errFirstAdminProvisioning
	}
	return nil
}

func observeBootstrapTokenIfPresent(path string) (os.FileInfo, error) {
	if path == "" {
		return nil, errFirstAdminProvisioning
	}
	if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, errFirstAdminProvisioning
	}
	return observeBootstrapToken(path)
}

func installationKey(kind, seed string) string {
	return "dashboard-first-admin:" + kind + ":" + seed
}

func zeroStringValue(value *string) {
	if value != nil {
		*value = ""
	}
}

func zeroByteSlice(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

var _ dashboardauth.FirstAdminProvisioner = (*managementSessionProvider)(nil)
