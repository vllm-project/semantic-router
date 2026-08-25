package managementapi

import "testing"

func TestInvitationOpenAPIUsesSynchronousOneTimeContracts(t *testing.T) {
	document := GenerateOpenAPI()
	for _, name := range []string{
		"InvitationCreateRequest", "InvitationRotateTokenRequest", "InvitationIssuedSecret",
		"InvitationPage", "OnboardingCreateRequest", "OnboardingResult",
	} {
		if _, found := document.Components.Schemas[name]; !found {
			t.Fatalf("schema %q is missing", name)
		}
	}
	snapshot := document.Components.Schemas["InvitationOnboardingSnapshot"]
	if len(snapshot.OneOf) != 2 {
		t.Fatalf("InvitationOnboardingSnapshot variants = %d, want default and Team inheritance", len(snapshot.OneOf))
	}
	if _, found := snapshot.OneOf[0].Properties["team"]; found {
		t.Fatal("default onboarding snapshot unexpectedly accepts Team")
	}
	if _, found := snapshot.OneOf[1].Properties["accessPolicyId"]; found {
		t.Fatal("Team onboarding snapshot unexpectedly materializes a User policy override")
	}
	rotate, found := LookupOperation(MethodPOST, BasePath+"/invitations/{invitationId}:rotate-token")
	if !found || rotate.Revision != RevisionCAS || rotate.Secret.Output != SecretOutputOneTime || !rotate.Secret.NoStore {
		t.Fatalf("rotate contract = %#v", rotate)
	}
	revoke, found := LookupOperation(MethodDELETE, BasePath+"/invitations/{invitationId}")
	if !found || revoke.Revision != RevisionCAS {
		t.Fatalf("revoke contract = %#v", revoke)
	}
	onboarding, found := LookupOperation(MethodPOST, BasePath+"/onboarding")
	if !found || onboarding.Async != AsyncSynchronous || onboarding.Revision != RevisionNone ||
		onboarding.Secret.Output != SecretOutputOneTime || !onboarding.Secret.NoStore {
		t.Fatalf("onboarding contract = %#v", onboarding)
	}
	response := document.Paths[BasePath+"/onboarding"]["post"].Responses["201"]
	if response.Content[JSONMediaType].Schema.Ref != "#/components/schemas/OnboardingResult" {
		t.Fatalf("onboarding response schema = %#v", response.Content)
	}
}
