package accessmanagement

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestRoutingCatalogUsesAppliedKeyProjectionAndHidesCrossScopeResources(t *testing.T) {
	desiredProjection := testProjection()
	desiredProjection.Grants = []accessprojection.Grant{
		discoveryGrant("desired-hidden-model", accesscontrol.GrantResourceModel, "model-hidden", accesscontrol.GrantEffectAllow),
		discoveryGrant("desired-hidden-entrypoint", accesscontrol.GrantResourceEntrypoint, "entrypoint-hidden", accesscontrol.GrantEffectAllow),
	}
	appliedProjection := testProjection()
	appliedProjection.Grants = []accessprojection.Grant{
		discoveryGrant("visible-model", accesscontrol.GrantResourceModel, "model-visible", accesscontrol.GrantEffectAllow),
		discoveryGrant("hidden-model-allow", accesscontrol.GrantResourceModel, "model-hidden", accesscontrol.GrantEffectAllow),
		discoveryGrant("hidden-model-deny", accesscontrol.GrantResourceModel, "model-hidden", accesscontrol.GrantEffectDeny),
		discoveryGrant("visible-entrypoint", accesscontrol.GrantResourceEntrypoint, "entrypoint-visible", accesscontrol.GrantEffectAllow),
	}
	snapshot := routingCatalogSnapshot()
	applied := accessruntime.AppliedPolicy{
		Active: accessruntime.ActivePolicy{
			KeyID: testKeySubject.ID, Revision: appliedProjection.Revision, Digest: appliedProjection.Digest,
			PublicationID: "publication-9", RuntimeEpoch: 2,
			RoutingRevision: snapshot.Revision, RoutingDocumentDigest: strings.Repeat("d", 64),
		},
		Projection: appliedProjection,
	}
	service := newTestService(t,
		&repositoryStub{snapshot: testSnapshot(desiredProjection)},
		&appliedStub{policy: applied}, &meterStub{}, &waiterStub{},
	)
	reader := &routingPublicationStub{publication: &RoutingPublication{
		RoutingDocumentDigest: applied.Active.RoutingDocumentDigest, Snapshot: snapshot,
	}}
	service.routing = reader

	catalog, err := service.GetRoutingCatalog(context.Background(), testNamespaceID, testKeySubject)
	if err != nil {
		t.Fatal(err)
	}
	if catalog.Subject != testKeySubject || catalog.PolicyDigest != appliedProjection.Digest ||
		catalog.RoutingDigest != applied.Active.RoutingDocumentDigest || catalog.RoutingRevision != snapshot.Revision {
		t.Fatalf("catalog publication identity = %#v", catalog)
	}
	if reader.pin.NamespaceID != testNamespaceID || reader.pin.QuotaPartition != "partition-1" ||
		reader.pin.PublicationID != applied.Active.PublicationID || reader.pin.RuntimeEpoch != applied.Active.RuntimeEpoch ||
		reader.pin.RoutingRevision != applied.Active.RoutingRevision ||
		reader.pin.RoutingDocumentDigest != applied.Active.RoutingDocumentDigest {
		t.Fatalf("routing publication pin = %#v", reader.pin)
	}
	if len(catalog.Models) != 1 || catalog.Models[0].ID != "model-visible" ||
		len(catalog.Entrypoints) != 1 || catalog.Entrypoints[0].ID != "entrypoint-visible" ||
		len(catalog.Recipes) != 1 || catalog.Recipes[0].ID != "recipe-visible" {
		t.Fatalf("catalog was not filtered by the applied projection: %#v", catalog)
	}
	assignments := catalog.Entrypoints[0].Rules[0].Assignments["decision-visible"].Models
	if len(assignments) != 1 || assignments[0].ModelID != "model-visible" {
		t.Fatalf("topology exposed a hidden Model assignment: %#v", assignments)
	}
}

func TestRoutingCatalogFailsClosedWhenRoutingPinDoesNotMatchSnapshot(t *testing.T) {
	projection := testProjection()
	projection.Grants = []accessprojection.Grant{
		discoveryGrant("visible-model", accesscontrol.GrantResourceModel, "model-visible", accesscontrol.GrantEffectAllow),
	}
	snapshot := routingCatalogSnapshot()
	applied := testAppliedPolicy(projection)
	applied.Active.RoutingRevision = snapshot.Revision
	applied.Active.PublicationID = "publication-9"
	applied.Active.RuntimeEpoch = 2
	applied.Active.RoutingDocumentDigest = strings.Repeat("f", 64)
	service := newTestService(t,
		&repositoryStub{snapshot: testSnapshot(projection)},
		&appliedStub{policy: applied}, &meterStub{}, &waiterStub{},
	)
	service.routing = &routingPublicationStub{publication: &RoutingPublication{
		RoutingDocumentDigest: strings.Repeat("e", 64), Snapshot: snapshot,
	}}

	_, err := service.GetRoutingCatalog(context.Background(), testNamespaceID, testKeySubject)
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("mismatched routing pin error = %v, want ErrUnavailable", err)
	}
}

func discoveryGrant(
	bindingID string,
	resourceType accesscontrol.GrantResourceType,
	resourceID string,
	effect accesscontrol.GrantEffect,
) accessprojection.Grant {
	return accessprojection.Grant{
		BindingID: bindingID, PolicyID: "policy-" + bindingID,
		Source: accesscontrol.InheritanceLayerUser, ResourceType: resourceType,
		ResourceID: resourceID, Permission: accesscontrol.GrantPermissionDiscover, Effect: effect,
	}
}

func routingCatalogSnapshot() routingsnapshot.Snapshot {
	visibleAssignments := map[string]routingsnapshot.AssignmentSet{
		"decision-visible": {Models: []routingsnapshot.Assignment{
			{ModelID: "model-visible", ModelRevision: 1, Priority: 0, Weight: "1"},
			{ModelID: "model-hidden", ModelRevision: 1, Priority: 1, Weight: "1"},
		}},
	}
	return routingsnapshot.Snapshot{
		Bundle: routingsnapshot.Bundle{
			NamespaceID: testNamespaceID, Revision: 9, Currency: "USD",
			Models: []routingsnapshot.Model{
				{ID: "model-visible", Revision: 1, Name: "Visible Model"},
				{ID: "model-hidden", Revision: 1, Name: "Other Team Model"},
			},
			Recipes: []routingsnapshot.Recipe{
				{ID: "recipe-visible", Revision: 1, Name: "Visible Recipe", Decisions: []routingsnapshot.Decision{{ID: "decision-visible", Name: "Visible"}}},
				{ID: "recipe-hidden", Revision: 1, Name: "Other Team Recipe", Decisions: []routingsnapshot.Decision{{ID: "decision-hidden", Name: "Hidden"}}},
			},
			Entrypoints: []routingsnapshot.Entrypoint{
				{ID: "entrypoint-visible", Revision: 1, Name: "Visible MoM", Aliases: []string{"visible"}, Rules: []routingsnapshot.EntrypointRule{{
					ID: "rule-visible", Name: "Visible", RecipeID: "recipe-visible", RecipeRevision: 1,
					Assignments: visibleAssignments,
				}}},
				{ID: "entrypoint-hidden", Revision: 1, Name: "Other Team MoM", Aliases: []string{"hidden"}, Rules: []routingsnapshot.EntrypointRule{{
					ID: "rule-hidden", Name: "Hidden", RecipeID: "recipe-hidden", RecipeRevision: 1,
					Assignments: map[string]routingsnapshot.AssignmentSet{},
				}}},
			},
		},
		Digest: strings.Repeat("e", 64),
	}
}
