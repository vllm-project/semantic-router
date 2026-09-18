package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func evaluationLifecycleRequest(
	method string,
	path string,
	body string,
	authContext *dashboardauth.AuthContext,
) *http.Request {
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	if authContext != nil {
		request = request.WithContext(dashboardauth.WithAuthContext(request.Context(), *authContext))
	}
	return request
}

func TestEvaluationLifecycleUsesAuthenticatedPrincipalWithoutIdentityLeaks(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	t.Cleanup(func() { _ = service.Close() })
	handler := NewEvaluationPlaneHandler(service, false)
	owner := dashboardauth.AuthContext{
		UserID: "owner-private-id", Email: "owner-secret@example.test", Role: dashboardauth.RoleWrite,
	}
	other := dashboardauth.AuthContext{UserID: "other-private-id", Role: dashboardauth.RoleWrite}
	admin := dashboardauth.AuthContext{UserID: "admin-private-id", Role: dashboardauth.RoleAdmin}

	unauthenticated := httptest.NewRecorder()
	handler.Runs(unauthenticated, evaluationLifecycleRequest(
		http.MethodPost, evaluationAPIBase+"/runs", validCreateRunJSON(), nil,
	))
	if unauthenticated.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated create status=%d body=%s", unauthenticated.Code, unauthenticated.Body.String())
	}

	created := httptest.NewRecorder()
	handler.Runs(created, evaluationLifecycleRequest(
		http.MethodPost, evaluationAPIBase+"/runs", validCreateRunJSON(), &owner,
	))
	if created.Code != http.StatusCreated {
		t.Fatalf("owner create status=%d body=%s", created.Code, created.Body.String())
	}
	var run evaluationplane.Run
	if err := json.NewDecoder(created.Body).Decode(&run); err != nil {
		t.Fatalf("decode created run: %v", err)
	}

	otherRead := httptest.NewRecorder()
	handler.RunRoute(otherRead, evaluationLifecycleRequest(
		http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/lifecycle", "", &other,
	))
	if otherRead.Code != http.StatusForbidden {
		t.Fatalf("cross-owner lifecycle read status=%d body=%s", otherRead.Code, otherRead.Body.String())
	}

	ownerHold := httptest.NewRecorder()
	handler.RunRoute(ownerHold, evaluationLifecycleRequest(
		http.MethodPost,
		evaluationAPIBase+"/runs/"+run.ID+"/lifecycle",
		`{"evidence_hold":true}`,
		&owner,
	))
	if ownerHold.Code != http.StatusOK || !strings.Contains(ownerHold.Body.String(), `"evidence_hold":true`) {
		t.Fatalf("owner hold status=%d body=%s", ownerHold.Code, ownerHold.Body.String())
	}
	for _, privateValue := range []string{owner.UserID, owner.Email} {
		if strings.Contains(ownerHold.Body.String(), privateValue) {
			t.Fatalf("lifecycle response leaked authenticated identity %q: %s", privateValue, ownerHold.Body.String())
		}
	}

	otherDelete := httptest.NewRecorder()
	handler.RunRoute(otherDelete, evaluationLifecycleRequest(
		http.MethodDelete, evaluationAPIBase+"/runs/"+run.ID, "", &other,
	))
	if otherDelete.Code != http.StatusForbidden {
		t.Fatalf("cross-owner delete status=%d body=%s", otherDelete.Code, otherDelete.Body.String())
	}

	usage := httptest.NewRecorder()
	handler.LifecycleUsage(usage, evaluationLifecycleRequest(
		http.MethodGet, evaluationAPIBase+"/lifecycle/usage", "", &owner,
	))
	if usage.Code != http.StatusOK {
		t.Fatalf("owner usage status=%d body=%s", usage.Code, usage.Body.String())
	}
	ownerActor, err := evaluationplane.NewActor(owner.UserID, false)
	if err != nil {
		t.Fatalf("derive expected actor: %v", err)
	}
	if !strings.Contains(usage.Body.String(), ownerActor.PrincipalDigest()) {
		t.Fatalf("usage response does not contain the server-owned principal digest: %s", usage.Body.String())
	}
	for _, privateValue := range []string{owner.UserID, owner.Email, other.UserID, admin.UserID} {
		if strings.Contains(usage.Body.String(), privateValue) {
			t.Fatalf("usage response leaked authenticated identity %q: %s", privateValue, usage.Body.String())
		}
	}

	nonAdminCollection := httptest.NewRecorder()
	handler.LifecycleCollection(nonAdminCollection, evaluationLifecycleRequest(
		http.MethodPost, evaluationAPIBase+"/lifecycle/collection", `{"apply":false}`, &owner,
	))
	if nonAdminCollection.Code != http.StatusForbidden {
		t.Fatalf("non-admin collection status=%d body=%s", nonAdminCollection.Code, nonAdminCollection.Body.String())
	}

	adminCollection := httptest.NewRecorder()
	handler.LifecycleCollection(adminCollection, evaluationLifecycleRequest(
		http.MethodPost, evaluationAPIBase+"/lifecycle/collection", `{"apply":false}`, &admin,
	))
	if adminCollection.Code != http.StatusOK || !strings.Contains(adminCollection.Body.String(), `"applied":false`) {
		t.Fatalf("admin collection status=%d body=%s", adminCollection.Code, adminCollection.Body.String())
	}
}
