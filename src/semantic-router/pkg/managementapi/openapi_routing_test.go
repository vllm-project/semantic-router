package managementapi

import (
	"encoding/json"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestRoutingOperationContractsMatchHTTPMutationSemantics(t *testing.T) {
	tests := []struct {
		method      HTTPMethod
		path        string
		idempotency IdempotencyMode
		revision    RevisionMode
		async       AsyncMode
		body        bool
		responseRef string
	}{
		{MethodPOST, BasePath + "/routing/imports", IdempotencyRequired, RevisionCAS, AsyncOperation, true, "RoutingManifestImportResult"},
		{MethodPOST, BasePath + "/routing/models", IdempotencyRequired, RevisionReturns, AsyncSynchronous, true, "MutationReceipt"},
		{MethodPOST, BasePath + "/routing/models:bulk-import", IdempotencyRequired, RevisionNone, AsyncOperation, true, "MutationReceipt"},
		{MethodDELETE, BasePath + "/routing/models/{modelId}", IdempotencyNone, RevisionCAS, AsyncSynchronous, false, ""},
		{MethodPOST, BasePath + "/routing/models/{modelId}:probe", IdempotencyNone, RevisionNone, AsyncSynchronous, false, "RoutingProbeResponse"},
		{MethodDELETE, BasePath + "/routing/recipes/{recipeId}", IdempotencyNone, RevisionCAS, AsyncSynchronous, false, ""},
		{MethodPOST, BasePath + "/routing/entrypoints/{entrypointId}:publish", IdempotencyRequired, RevisionCAS, AsyncOperation, false, "MutationReceipt"},
		{MethodPOST, BasePath + "/routing/entrypoints/{entrypointId}:unpublish", IdempotencyRequired, RevisionCAS, AsyncOperation, false, "MutationReceipt"},
		{MethodPOST, BasePath + "/routing/entrypoints/{entrypointId}:resolve", IdempotencyNone, RevisionNone, AsyncSynchronous, true, "RoutingResolveResponse"},
	}
	document := GenerateOpenAPI()
	for _, test := range tests {
		contract, found := LookupOperation(test.method, test.path)
		if !found {
			t.Fatalf("missing %s %s", test.method, test.path)
		}
		if contract.Idempotency != test.idempotency || contract.Revision != test.revision || contract.Async != test.async {
			t.Errorf("%s %s metadata = (%s,%s,%s), want (%s,%s,%s)", test.method, test.path,
				contract.Idempotency, contract.Revision, contract.Async,
				test.idempotency, test.revision, test.async)
		}
		operation := document.Paths[test.path][strings.ToLower(string(test.method))]
		if (operation.RequestBody != nil) != test.body {
			t.Errorf("%s %s request body present = %v, want %v", test.method, test.path, operation.RequestBody != nil, test.body)
		}
		if test.responseRef == "" {
			continue
		}
		status := "200"
		if test.async == AsyncOperation {
			status = "202"
		} else if test.method == MethodPOST && !strings.Contains(test.path, ":") {
			status = "201"
		}
		response := operation.Responses[status]
		got := response.Content[JSONMediaType].Schema.Ref
		want := "#/components/schemas/" + test.responseRef
		if got != want {
			t.Errorf("%s %s response = %q, want %q", test.method, test.path, got, want)
		}
	}
}

func TestRoutingManifestOpenAPIDocumentsDryRunAndYAMLExport(t *testing.T) {
	document := GenerateOpenAPI()
	importOperation := document.Paths[BasePath+"/routing/imports"]["post"]
	if got := importOperation.Responses["200"].Content[JSONMediaType].Schema.Ref; got != "#/components/schemas/RoutingManifestImportResult" {
		t.Fatalf("dry-run response = %q", got)
	}
	exportOperation := document.Paths[BasePath+"/routing/exports/current"]["get"]
	media, found := exportOperation.Responses["200"].Content[YAMLMediaType]
	if !found || media.Schema.Type != "string" || len(exportOperation.Responses["200"].Content) != 1 {
		t.Fatalf("export response content = %#v", exportOperation.Responses["200"].Content)
	}
}

func TestRoutingOpenAPISafeViewsOmitControlPlaneBindings(t *testing.T) {
	document := GenerateOpenAPI()
	model := document.Components.Schemas["RoutingModelView"]
	modelCardView := document.Components.Schemas["RoutingModelCardView"]
	modelCard := document.Components.Schemas["RoutingModelCard"]
	backend := document.Components.Schemas["RoutingModelBackendView"]
	for _, field := range []string{
		"credentialId", "providerCredentialId", "baseUrl", "origin", "connectionFields",
		"connection", "protocolAdapterId", "headers", "path", "secret",
	} {
		if _, exposed := model.Properties[field]; exposed {
			t.Errorf("RoutingModelView exposes %q", field)
		}
		if _, exposed := backend.Properties[field]; exposed {
			t.Errorf("RoutingModelBackendView exposes %q", field)
		}
		if _, exposed := modelCardView.Properties[field]; exposed {
			t.Errorf("RoutingModelCardView exposes %q", field)
		}
		if _, exposed := modelCard.Properties[field]; exposed {
			t.Errorf("RoutingModelCard exposes %q", field)
		}
	}
	for _, field := range []string{
		"status", "revision", "modelRevision", "catalogRevision", "control", "execution", "pricing",
		"backends", "createdAt", "updatedAt",
	} {
		if _, exposed := modelCardView.Properties[field]; exposed {
			t.Errorf("RoutingModelCardView exposes runtime field %q", field)
		}
		if _, exposed := modelCard.Properties[field]; exposed {
			t.Errorf("RoutingModelCard exposes runtime field %q", field)
		}
	}
	encoded, err := json.Marshal(document.Components.Schemas)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"protocolAdapterId", "credentialAdapterId", "discoveryAdapterId"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Errorf("public Routing schemas expose internal adapter field %q", forbidden)
		}
	}
}

func TestRoutingOpenAPIExposesPurposeSpecificModelCardPage(t *testing.T) {
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/routing/model-cards"]["get"]
	response := operation.Responses["200"]
	if got := response.Content[JSONMediaType].Schema.Ref; got != "#/components/schemas/RoutingModelCardPage" {
		t.Fatalf("Model Card response = %q", got)
	}
}

func TestRoutingModelPatchIsSparseAndDoesNotRequireBackendRoundTrip(t *testing.T) {
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/routing/models/{modelId}"]["patch"]
	if operation.RequestBody == nil {
		t.Fatal("Model PATCH omitted its request body")
	}
	ref := operation.RequestBody.Content[JSONMediaType].Schema.Ref
	if ref != "#/components/schemas/RoutingModelPatch" {
		t.Fatalf("Model PATCH schema = %q", ref)
	}
	patch := document.Components.Schemas["RoutingModelPatch"]
	if len(patch.Required) != 0 {
		t.Fatalf("Model PATCH unexpectedly requires fields: %#v", patch.Required)
	}
	for _, field := range []string{"name", "control", "pricing", "backends"} {
		if _, exists := patch.Properties[field]; !exists {
			t.Fatalf("Model PATCH omitted %q", field)
		}
	}
}

func TestRoutingModelControlOpenAPIConstraints(t *testing.T) {
	document := GenerateOpenAPI()
	if !slices.Equal(routingModelRetryEvidence, []string{"unavailable", "timeout"}) {
		t.Fatalf("routingModelRetryEvidence = %#v", routingModelRetryEvidence)
	}
	retry := document.Components.Schemas["RoutingModelRetryControl"]
	if !slices.Equal(retry.Required, []string{"count", "on"}) {
		t.Fatalf("RoutingModelRetryControl.required = %#v", retry.Required)
	}
	retryOn := retry.Properties["on"]
	if retryOn.Items == nil ||
		!slices.Equal(retryOn.Items.Enum, routingModelRetryEvidence) ||
		retryOn.MaxItems == nil || *retryOn.MaxItems != int64(len(routingModelRetryEvidence)) ||
		!retryOn.UniqueItems {
		t.Fatalf("RoutingModelRetryControl.on = %#v", retryOn)
	}

	timeout := document.Components.Schemas["RoutingModelTimeoutControl"]
	if !slices.Equal(timeout.Required, []string{"request", "stream"}) {
		t.Fatalf("RoutingModelTimeoutControl.required = %#v", timeout.Required)
	}
	for _, field := range []string{"request", "stream"} {
		schema := timeout.Properties[field]
		if schema.Pattern != routingModelDurationPattern ||
			!strings.Contains(schema.Description, "between 1s and 24h") {
			t.Fatalf("RoutingModelTimeoutControl.%s = %#v", field, schema)
		}
		pattern, err := regexp.Compile(schema.Pattern)
		if err != nil {
			t.Fatalf("RoutingModelTimeoutControl.%s pattern: %v", field, err)
		}
		for _, value := range []string{"1s", "+1.5s", "1h30m", "1000ms", ".5m"} {
			if !pattern.MatchString(value) {
				t.Errorf("RoutingModelTimeoutControl.%s rejected duration syntax %q", field, value)
			}
		}
		for _, value := range []string{"", "30", "1d", "-1s", " 1s", "1 s"} {
			if pattern.MatchString(value) {
				t.Errorf("RoutingModelTimeoutControl.%s accepted invalid duration syntax %q", field, value)
			}
		}
	}

	fallbackOn := document.Components.Schemas["RoutingFallbackPolicy"].Properties["on"]
	if fallbackOn.Items == nil ||
		!slices.Equal(fallbackOn.Items.Enum, routingModelRetryEvidence) ||
		fallbackOn.MinItems == nil || *fallbackOn.MinItems != 1 ||
		fallbackOn.MaxItems == nil || *fallbackOn.MaxItems != int64(len(routingModelRetryEvidence)) {
		t.Fatalf("RoutingFallbackPolicy.on = %#v", fallbackOn)
	}
	control := document.Components.Schemas["RoutingModelControl"]
	if !slices.Equal(control.Required, []string{"retry", "timeout"}) {
		t.Fatalf("RoutingModelControl.required = %#v", control.Required)
	}
}

func TestRoutingModelPricingOpenAPIConstraints(t *testing.T) {
	document := GenerateOpenAPI()
	pricing := document.Components.Schemas["RoutingPricing"]
	for _, field := range []string{
		"inputCostPerMillionTokens", "outputCostPerMillionTokens",
		"cacheReadCostPerMillionTokens", "cacheWriteCostPerMillionTokens",
	} {
		schema := pricing.Properties[field]
		if len(schema.OneOf) != 2 || schema.OneOf[0].Pattern != routingModelPricePattern || schema.OneOf[1].Type != "null" {
			t.Fatalf("RoutingPricing.%s = %#v", field, schema)
		}
		pattern, err := regexp.Compile(schema.OneOf[0].Pattern)
		if err != nil {
			t.Fatalf("RoutingPricing.%s pattern: %v", field, err)
		}
		for _, value := range []string{"0", "0.000000001", "999999.999999999", "1000000", "1000000.000"} {
			if !pattern.MatchString(value) {
				t.Errorf("RoutingPricing.%s rejected %q", field, value)
			}
		}
		for _, value := range []string{"01", "1e-3", "0.0000000001", "1000000.000000001", "1000001"} {
			if pattern.MatchString(value) {
				t.Errorf("RoutingPricing.%s accepted %q", field, value)
			}
		}
	}
}

func TestRoutingModelMetadataOpenAPIConstraints(t *testing.T) {
	document := GenerateOpenAPI()
	for _, name := range []string{
		"RoutingModelWrite", "RoutingModelPatch", "RoutingModelView", "RoutingModelCard", "RoutingBulkModelSelection",
	} {
		schema := document.Components.Schemas[name]
		contextWindow := schema.Properties["contextWindowSize"]
		if contextWindow.Minimum == nil || *contextWindow.Minimum != 0 ||
			contextWindow.Maximum == nil || *contextWindow.Maximum != 100_000_000 {
			t.Errorf("%s.contextWindowSize = %#v", name, contextWindow)
		}
		quality := schema.Properties["qualityScore"]
		if quality.Minimum == nil || *quality.Minimum != 0 ||
			quality.Maximum == nil || *quality.Maximum != 1 {
			t.Errorf("%s.qualityScore = %#v", name, quality)
		}
	}
}

func TestRoutingOpenAPIExposesTopologyAsExplicitDetailOption(t *testing.T) {
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/routing/entrypoints/{entrypointId}"]["get"]
	var found bool
	for _, parameter := range operation.Parameters {
		if parameter.Name == "includeTopology" && parameter.In == "query" && parameter.Schema.Type == "boolean" {
			found = true
		}
	}
	if !found {
		t.Fatal("Entrypoint detail OpenAPI omitted includeTopology")
	}
	view := document.Components.Schemas["RoutingEntrypointView"]
	if _, exists := view.Properties["rules"]; !exists {
		t.Fatal("Entrypoint detail schema omitted authorized rules")
	}
	for _, field := range []string{"recipeIds", "ruleCount", "assignedModelCount"} {
		if _, exists := view.Properties[field]; !exists {
			t.Fatalf("Entrypoint summary schema omitted %s", field)
		}
		if !slices.Contains(view.Required, field) {
			t.Fatalf("Entrypoint summary schema does not require %s", field)
		}
	}
	if items := view.Properties["recipeIds"].Items; items == nil || items.Pattern != routingResourceID.Pattern {
		t.Fatalf("Entrypoint recipeIds schema = %#v", view.Properties["recipeIds"])
	}
}

func TestRoutingOpenAPIExposesImmutableRecipeProvenance(t *testing.T) {
	document := GenerateOpenAPI()
	view := document.Components.Schemas["RoutingRecipeView"]
	for _, field := range []string{"origin", "immutable", "provenance"} {
		if _, exists := view.Properties[field]; !exists {
			t.Fatalf("RoutingRecipeView omitted %s", field)
		}
	}
	if view.Properties["provenance"].Ref != "#/components/schemas/RoutingRecipeProvenanceView" {
		t.Fatalf("Recipe provenance ref = %q", view.Properties["provenance"].Ref)
	}
	provenance := document.Components.Schemas["RoutingRecipeProvenanceView"]
	for _, field := range []string{
		"distributionId", "distributionVersion", "assetDigest", "sourceRecipeId",
		"sourceRevision", "recipeDigest", "installedAt",
	} {
		if _, exists := provenance.Properties[field]; !exists {
			t.Fatalf("RoutingRecipeProvenanceView omitted %s", field)
		}
	}
}

func TestRoutingOpenAPIExposesTypedImmutableSnapshots(t *testing.T) {
	document := GenerateOpenAPI()
	listPath := BasePath + "/namespaces/{namespaceId}/routing/snapshots"
	detailPath := listPath + "/{routingRevision}"
	if got := document.Paths[listPath]["get"].Responses["200"].Content[JSONMediaType].Schema.Ref; got != "#/components/schemas/RoutingSnapshotPage" {
		t.Fatalf("Routing snapshot page response = %q", got)
	}
	if got := document.Paths[detailPath]["get"].Responses["200"].Content[JSONMediaType].Schema.Ref; got != "#/components/schemas/RoutingSnapshotDetail" {
		t.Fatalf("Routing snapshot detail response = %q", got)
	}
	record := document.Components.Schemas["RoutingSnapshotRecord"]
	for _, field := range []string{"metadata", "members", "export"} {
		if _, exists := record.Properties[field]; !exists || !slices.Contains(record.Required, field) {
			t.Fatalf("RoutingSnapshotRecord omitted required %s", field)
		}
	}
	member := document.Components.Schemas["RoutingSnapshotMember"]
	for _, field := range []string{"resourceType", "resourceId", "resourceRevision"} {
		if _, exists := member.Properties[field]; !exists || !slices.Contains(member.Required, field) {
			t.Fatalf("RoutingSnapshotMember omitted required %s", field)
		}
	}
	model := document.Components.Schemas["RoutingSnapshotModel"]
	if _, exists := model.Properties["execution"]; exists {
		t.Fatal("RoutingSnapshotModel exposes internal execution storage")
	}
	if _, exists := document.Components.Schemas["RoutingSnapshotExecution"]; exists {
		t.Fatal("OpenAPI retains a duplicate public execution schema")
	}
	if control := model.Properties["control"]; control.Ref != "#/components/schemas/RoutingModelControl" ||
		!slices.Contains(model.Required, "control") {
		t.Fatalf("RoutingSnapshotModel control = %#v, required = %#v", control, model.Required)
	}
	control := document.Components.Schemas["RoutingModelControl"]
	if control.Properties["retry"].Ref != "#/components/schemas/RoutingModelRetryControl" ||
		control.Properties["timeout"].Ref != "#/components/schemas/RoutingModelTimeoutControl" {
		t.Fatalf("nested Model control schema = %#v", control)
	}
}
