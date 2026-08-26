package managementapi

import "testing"

func TestOpenAPIUsesTypedObservabilityResponsesAndQueries(t *testing.T) {
	document := GenerateOpenAPI()
	for path, schema := range map[string]string{
		BasePath + "/usage":                                               "#/components/schemas/UsageSummary",
		BasePath + "/users/{userId}/usage":                                "#/components/schemas/UsageSummary",
		BasePath + "/teams/{teamId}/usage":                                "#/components/schemas/UsageSummary",
		BasePath + "/api-keys/{keyId}/usage":                              "#/components/schemas/UsageSummary",
		BasePath + "/usage/series":                                        "#/components/schemas/UsageSeries",
		BasePath + "/usage/breakdowns":                                    "#/components/schemas/UsageBreakdown",
		BasePath + "/request-logs":                                        "#/components/schemas/RequestLogPage",
		BasePath + "/audit-events":                                        "#/components/schemas/AuditEventPage",
		BasePath + "/runtime-diagnostics":                                 "#/components/schemas/RuntimeDiagnostics",
		BasePath + "/namespaces/{namespaceId}/request-logs/{admissionId}": "#/components/schemas/RequestLogDetail",
	} {
		operation := document.Paths[path]["get"]
		response := operation.Responses["200"].Content[JSONMediaType].Schema
		if response.Ref != schema {
			t.Errorf("GET %s response = %q, want %q", path, response.Ref, schema)
		}
	}
	breakdown := document.Paths[BasePath+"/usage/breakdowns"]["get"]
	foundDimension := false
	for _, parameter := range breakdown.Parameters {
		if parameter.In == "query" && parameter.Name == "dimension" {
			foundDimension = len(parameter.Schema.Enum) == 11
		}
	}
	if !foundDimension {
		t.Fatal("usage breakdown OpenAPI does not publish the closed dimension vocabulary")
	}
	userUsage := document.Paths[BasePath+"/users/{userId}/usage"]["get"]
	foundGrain, foundRedundantUserFilter := false, false
	for _, parameter := range userUsage.Parameters {
		if parameter.In == "query" && parameter.Name == "grain" {
			foundGrain = true
		}
		if parameter.In == "query" && parameter.Name == "userId" {
			foundRedundantUserFilter = true
		}
	}
	if !foundGrain || foundRedundantUserFilter {
		t.Fatalf("subject usage query parameters grain=%t redundantUser=%t", foundGrain, foundRedundantUserFilter)
	}
	if _, exists := document.Components.Schemas["RequestDispatch"]; !exists {
		t.Fatal("request-log detail schemas are missing")
	}
	requestLogs := document.Paths[BasePath+"/request-logs"]["get"]
	foundRequestID := false
	for _, parameter := range requestLogs.Parameters {
		if parameter.In == "query" && parameter.Name == "requestId" {
			foundRequestID = parameter.Schema.MaxLength != nil && *parameter.Schema.MaxLength == 256
		}
	}
	if !foundRequestID {
		t.Fatal("request-log query omits the bounded exact requestId filter")
	}
	if _, exists := document.Components.Schemas["RequestLog"].Properties["externalRequestId"]; !exists {
		t.Fatal("request-log response omits externalRequestId correlation")
	}
	requestLog := document.Components.Schemas["RequestLog"]
	for _, field := range []string{"decisionId", "decisionName", "decisionTier", "models"} {
		if _, exists := requestLog.Properties[field]; !exists {
			t.Errorf("request-log response omits %s routing evidence", field)
		}
	}
	if requestLog.Properties["models"].Items == nil ||
		requestLog.Properties["models"].Items.Ref != "#/components/schemas/RequestModel" {
		t.Fatal("request-log Model snapshots are not typed")
	}
	if _, exists := document.Components.Schemas["AuditEvent"]; !exists {
		t.Fatal("audit event schema is missing")
	}
	diagnostics := document.Paths[BasePath+"/runtime-diagnostics"]["get"]
	foundNamespaceSelector := false
	for _, parameter := range diagnostics.Parameters {
		if parameter.In == "query" && parameter.Name == "namespaceId" {
			foundNamespaceSelector = parameter.Schema.Pattern != ""
		}
	}
	if !foundNamespaceSelector {
		t.Fatal("runtime diagnostics exact namespace selector is missing")
	}
	storage, exists := document.Components.Schemas["UsageStorageRuntimeDiagnostics"]
	if !exists {
		t.Fatal("usage storage diagnostics schema is missing")
	}
	for _, field := range []string{
		"status", "activeMonths", "retiredMonths", "dirtyMinuteBuckets",
		"dirtyHourBuckets", "dirtyDayBuckets", "dirtyCountsCapped",
		"oldestActiveMonth", "createdThrough",
	} {
		if _, exists := storage.Properties[field]; !exists {
			t.Errorf("usage storage diagnostics schema is missing %q", field)
		}
	}
	runtime := document.Components.Schemas["RuntimeDiagnostics"]
	if property := runtime.Properties["usageStorage"]; property.Ref != "#/components/schemas/UsageStorageRuntimeDiagnostics" {
		t.Fatalf("runtime usageStorage schema = %q", property.Ref)
	}
}
