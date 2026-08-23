package routerauth

import (
	"context"
	"errors"
	"net/http"
	"sort"
	"strings"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

var errEvaluationScopeUnavailable = errors.New("router management evaluation scope is unavailable")

// ResolveEvaluationScope derives Evaluation ownership from the same Router
// Management session used by the browser. Dashboard-local user identifiers are
// deliberately never treated as model-serving User identifiers.
func (provider *managementSessionProvider) ResolveEvaluationScope(
	ctx context.Context,
	principal dashboardauth.AuthContext,
) ([]string, map[string]string, error) {
	_, identity, err := provider.evaluationIdentity(ctx, principal)
	if err != nil {
		return nil, nil, errEvaluationScopeUnavailable
	}

	userSet := make(map[string]struct{})
	teamUsers := make(map[string]string)
	for _, namespace := range identity.Namespaces {
		if namespace.User == nil || strings.TrimSpace(namespace.User.UserID) == "" || namespace.User.Status != "active" {
			continue
		}
		userID := namespace.User.UserID
		userSet[userID] = struct{}{}
		for _, membership := range namespace.Teams {
			if membership.Status == "active" && strings.TrimSpace(membership.TeamID) != "" {
				teamUsers[membership.TeamID] = userID
			}
		}
	}
	if len(userSet) == 0 {
		return nil, nil, errEvaluationScopeUnavailable
	}
	userIDs := make([]string, 0, len(userSet))
	for userID := range userSet {
		userIDs = append(userIDs, userID)
	}
	sort.Strings(userIDs)
	return userIDs, teamUsers, nil
}

func (provider *managementSessionProvider) evaluationIdentity(
	ctx context.Context,
	principal dashboardauth.AuthContext,
) (string, managementapi.Me, error) {
	token, err := provider.ManagementAccessToken(ctx, principal)
	if err != nil {
		return "", managementapi.Me{}, errEvaluationScopeUnavailable
	}
	var identity managementapi.Me
	if err = provider.authorizedManagementRequest(
		ctx,
		token,
		"",
		http.MethodGet,
		managementBasePath+"/me",
		nil,
		nil,
		[]int{http.StatusOK},
		&identity,
	); err != nil {
		return "", managementapi.Me{}, errEvaluationScopeUnavailable
	}
	return token, identity, nil
}
