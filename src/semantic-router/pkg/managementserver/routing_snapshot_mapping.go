package managementserver

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func routingSnapshotMetadataDTO(snapshot routingmanagement.SnapshotMetadata) managementapi.RoutingSnapshotMetadata {
	return managementapi.RoutingSnapshotMetadata{
		NamespaceID: snapshot.NamespaceID, RoutingRevision: snapshot.RoutingRevision,
		ContentDigest: snapshot.ContentDigest, Status: string(snapshot.Status),
		FailureReason: snapshot.FailureReason, MemberCount: snapshot.MemberCount,
		CreatedAt: snapshot.CreatedAt, ActivatedAt: snapshot.ActivatedAt,
	}
}

func routingSnapshotPageDTO(
	page routingmanagement.Page[routingmanagement.SnapshotMetadata],
	pageSize int,
) managementapi.RoutingSnapshotPage {
	items := make([]managementapi.RoutingSnapshotMetadata, len(page.Items))
	for index := range page.Items {
		items[index] = routingSnapshotMetadataDTO(page.Items[index])
	}
	return managementapi.RoutingSnapshotPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageSize,
	}}
}

func routingSnapshotRecordDTO(detail routingmanagement.SnapshotDetail) managementapi.RoutingSnapshotRecord {
	members := make([]managementapi.RoutingSnapshotMember, len(detail.Members))
	for index, member := range detail.Members {
		members[index] = managementapi.RoutingSnapshotMember{
			ResourceType: member.ResourceType, ResourceID: member.ResourceID,
			ResourceRevision: member.ResourceRevision,
		}
	}
	return managementapi.RoutingSnapshotRecord{
		Metadata: routingSnapshotMetadataDTO(detail.Metadata), Members: members,
		Export: routingSnapshotExportDTO(detail.Export),
	}
}

func routingSnapshotExportDTO(snapshot routingsnapshot.Snapshot) managementapi.RoutingSnapshotExport {
	models := make([]managementapi.RoutingSnapshotModel, len(snapshot.Models))
	for index, model := range snapshot.Models {
		models[index] = routingSnapshotModelDTO(model)
	}
	recipes := make([]managementapi.RoutingSnapshotRecipe, len(snapshot.Recipes))
	for index, recipe := range snapshot.Recipes {
		recipes[index] = managementapi.RoutingSnapshotRecipe{
			ID: recipe.ID, Revision: recipe.Revision, Name: recipe.Name, Description: recipe.Description,
			Decisions: routingDecisionsDTO(recipe.Decisions), Document: append(json.RawMessage(nil), recipe.Document...),
		}
	}
	entrypoints := make([]managementapi.RoutingSnapshotEntrypoint, len(snapshot.Entrypoints))
	for index, entrypoint := range snapshot.Entrypoints {
		entrypoints[index] = managementapi.RoutingSnapshotEntrypoint{
			ID: entrypoint.ID, Revision: entrypoint.Revision, Name: entrypoint.Name,
			Aliases: append([]string{}, entrypoint.Aliases...), Rules: routingEntrypointRulesDTO(entrypoint.Rules),
		}
	}
	return managementapi.RoutingSnapshotExport{
		NamespaceID: snapshot.NamespaceID, Revision: snapshot.Revision, Currency: snapshot.Currency,
		Models: models, Recipes: recipes, Entrypoints: entrypoints, Digest: snapshot.Digest,
	}
}

func routingSnapshotModelDTO(model routingsnapshot.Model) managementapi.RoutingSnapshotModel {
	backends := make([]managementapi.RoutingSnapshotBackend, len(model.Backends))
	for index, backend := range model.Backends {
		backends[index] = managementapi.RoutingSnapshotBackend{
			ID: backend.ID, ProviderID: backend.ProviderID, WireFormat: string(backend.WireFormat),
			Origin: backend.Origin, ProviderModelID: backend.ProviderModelID,
			ProviderCredentialID: backend.ProviderCredentialID,
			Connection: managementapi.RoutingSnapshotBackendConnection{
				Path: backend.Connection.Path, Headers: cloneRoutingSnapshotHeaders(backend.Connection.Headers),
			},
			Weight: backend.Weight,
		}
	}
	return managementapi.RoutingSnapshotModel{
		ID: model.ID, Revision: model.Revision, CatalogRevision: model.CatalogRevision, Name: model.Name,
		Aliases: append([]string(nil), model.Aliases...), ParamSize: model.ParamSize,
		ContextWindowSize: model.ContextWindowSize, Description: model.Description,
		Capabilities: append([]string(nil), model.Capabilities...), Reasoning: routingReasoningDTO(model.Reasoning),
		LoRAs: append([]string(nil), model.LoRAs...), QualityScore: model.QualityScore,
		Modality: model.Modality, Tags: append([]string(nil), model.Tags...),
		Control: routingModelControlDTO(model.Execution), Pricing: managementapi.RoutingPricing(model.Pricing),
		Backends: backends,
	}
}

func cloneRoutingSnapshotHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return nil
	}
	clone := make(map[string]string, len(headers))
	for name, value := range headers {
		clone[name] = value
	}
	return clone
}
