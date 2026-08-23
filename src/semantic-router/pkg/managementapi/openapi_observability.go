package managementapi

import "strings"

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "observability", Schemas: observabilitySchemas,
		ResponseSchema:  observabilityResponseSchema,
		ExtraParameters: observabilityParameters,
	})
}

func observabilitySchemas() map[string]JSONSchema {
	stringSchema := JSONSchema{Type: "string"}
	timestamp := JSONSchema{Type: "string", Format: "date-time"}
	whole := JSONSchema{Type: "string", Pattern: wholeQuantityPattern.String()}
	costs := arraySchema(refSchema("CostSummary"))
	timing := objectSchema([]string{
		"sampleCount", "totalMilliseconds", "averageMilliseconds", "p50Milliseconds",
		"p95Milliseconds", "p99Milliseconds", "percentilesAreEstimated",
	}, map[string]JSONSchema{
		"sampleCount": whole, "totalMilliseconds": whole,
		"averageMilliseconds":     {Type: "number", Format: "double"},
		"p50Milliseconds":         {Type: "integer", Format: "int64"},
		"p95Milliseconds":         {Type: "integer", Format: "int64"},
		"p99Milliseconds":         {Type: "integer", Format: "int64"},
		"percentilesAreEstimated": {Type: "boolean"},
	})
	usageTotals := objectSchema([]string{
		"requests", "successfulRequests", "inputTokens", "outputTokens", "totalTokens",
		"incompleteDispatches", "completeness", "costs", "latency", "ttft",
	}, map[string]JSONSchema{
		"requests": whole, "successfulRequests": whole, "inputTokens": whole,
		"outputTokens": whole, "totalTokens": whole, "incompleteDispatches": whole,
		"completeness": {Type: "string", Enum: []string{"complete", "partial", "unknown"}},
		"costs":        costs, "latency": refSchema("TimingSummary"), "ttft": refSchema("TimingSummary"),
	})
	requestLog := objectSchema([]string{
		"admissionId", "eventId", "occurredAt", "completedAt", "protocol", "path",
		"statusCode", "usageState", "inputTokens", "outputTokens", "latencyMilliseconds",
		"stream", "toolCall", "costs",
	}, map[string]JSONSchema{
		"admissionId": stringSchema, "eventId": stringSchema,
		"occurredAt": timestamp, "completedAt": timestamp, "protocol": stringSchema,
		"path": stringSchema, "statusCode": {Type: "integer", Format: "int32"},
		"errorCode": stringSchema, "usageState": stringSchema,
		"inputTokens": whole, "outputTokens": whole,
		"latencyMilliseconds": {Type: "integer", Format: "int64"},
		"ttftMilliseconds":    {Type: "integer", Format: "int64"},
		"stream":              {Type: "boolean"}, "toolCall": {Type: "boolean"},
		"apiKeyId": stringSchema, "userId": stringSchema, "teamId": stringSchema,
		"entrypointId": stringSchema, "recipeId": stringSchema,
		"metadata": {Type: "object", AdditionalProperties: boolPointer(true)}, "costs": costs,
	})
	auditEvent := objectSchema([]string{
		"id", "namespaceId", "chainSequence", "actorChain", "action", "resourceType",
		"requestId", "outcome", "reason", "details", "eventHash", "createdAt",
	}, map[string]JSONSchema{
		"id": stringSchema, "namespaceId": stringSchema,
		"desiredRevision":  {Type: "integer", Format: "int64"},
		"chainSequence":    {Type: "integer", Format: "int64"},
		"actorPrincipalId": stringSchema, "actorChain": arraySchema(stringSchema),
		"action": stringSchema, "resourceType": stringSchema, "resourceId": stringSchema,
		"requestId": stringSchema, "sourceIp": stringSchema, "outcome": stringSchema,
		"reason":         stringSchema,
		"beforeRevision": {Type: "integer", Format: "int64"},
		"afterRevision":  {Type: "integer", Format: "int64"},
		"details":        {Type: "object", AdditionalProperties: boolPointer(true)},
		"previousHash":   stringSchema, "eventHash": stringSchema, "createdAt": timestamp,
	})
	publicationReadiness := objectSchema([]string{
		"ready", "reason", "runtimeEpoch", "desiredRevision", "appliedRevision", "projectorLag",
	}, map[string]JSONSchema{
		"ready": {Type: "boolean"}, "reason": stringSchema,
		"runtimeEpoch":    {Type: "integer", Format: "int64"},
		"desiredRevision": {Type: "integer", Format: "int64"},
		"appliedRevision": {Type: "integer", Format: "int64"},
		"accessGate":      stringSchema, "routingGate": stringSchema,
		"projectorLag": {Type: "integer", Format: "int64"},
	})
	publicationDiagnostics := objectSchema([]string{
		"namespaceId", "quotaPartition", "asOf", "readiness", "openPublications",
		"activeReplicas", "recordedRequiredReplicas", "barrierAcknowledgementsRequired", "barrierAcknowledgements",
		"routingAcknowledgements", "missingBarrierAcks", "missingRoutingAcks",
	}, map[string]JSONSchema{
		"namespaceId": stringSchema, "quotaPartition": stringSchema, "asOf": timestamp,
		"readiness":           refSchema("PublicationReadiness"),
		"activePublicationId": stringSchema, "candidatePublicationId": stringSchema,
		"openPublications":                {Type: "integer", Format: "int64"},
		"activeReplicas":                  arraySchema(stringSchema),
		"recordedRequiredReplicas":        arraySchema(stringSchema),
		"barrierAcknowledgementsRequired": {Type: "boolean"},
		"barrierAcknowledgements":         arraySchema(stringSchema),
		"routingAcknowledgements":         arraySchema(stringSchema),
		"missingBarrierAcks":              arraySchema(stringSchema),
		"missingRoutingAcks":              arraySchema(stringSchema),
	})
	quotaDiagnostics := objectSchema([]string{
		"partition", "asOf", "usageStreamBacklog", "pendingAdmissions",
		"expiredPendingAdmissions", "recoveryState",
	}, map[string]JSONSchema{
		"partition": stringSchema, "asOf": timestamp,
		"usageStreamBacklog":       {Type: "integer", Format: "int64"},
		"pendingAdmissions":        {Type: "integer", Format: "int64"},
		"expiredPendingAdmissions": {Type: "integer", Format: "int64"},
		"oldestPendingDeadline":    timestamp,
		"recoveryState":            {Type: "string", Enum: []string{"ready", "reconciliation_required"}},
	})
	storeStatus := objectSchema([]string{"status"}, map[string]JSONSchema{
		"status": {Type: "string", Enum: []string{"ready", "unavailable"}},
	})
	usageStorageStatus := objectSchema([]string{
		"status", "activeMonths", "retiredMonths", "dirtyMinuteBuckets",
		"dirtyHourBuckets", "dirtyDayBuckets", "dirtyCountsCapped",
	}, map[string]JSONSchema{
		"status":             {Type: "string", Enum: []string{"ready", "unavailable"}},
		"activeMonths":       {Type: "integer", Format: "int64"},
		"retiredMonths":      {Type: "integer", Format: "int64"},
		"dirtyMinuteBuckets": {Type: "integer", Format: "int64"},
		"dirtyHourBuckets":   {Type: "integer", Format: "int64"},
		"dirtyDayBuckets":    {Type: "integer", Format: "int64"},
		"dirtyCountsCapped":  {Type: "boolean"},
		"oldestActiveMonth":  timestamp,
		"createdThrough":     timestamp,
	})
	namespaceDiagnostics := objectSchema([]string{
		"namespaceId", "quotaPartition", "publication", "quota",
		"usageStreamBacklogLimit", "admissionBlockedByUsageBacklog",
	}, map[string]JSONSchema{
		"namespaceId": stringSchema, "quotaPartition": stringSchema,
		"publication":                    refSchema("PublicationRuntimeDiagnostics"),
		"quota":                          refSchema("QuotaRuntimeDiagnostics"),
		"usageStreamBacklogLimit":        {Type: "integer", Format: "int64"},
		"admissionBlockedByUsageBacklog": {Type: "boolean"},
	})
	return map[string]JSONSchema{
		"TimingSummary": timing,
		"UsageTotals":   usageTotals,
		"UsageSummary": objectSchema([]string{"totals", "grain", "final"}, map[string]JSONSchema{
			"totals": refSchema("UsageTotals"), "grain": observabilityGrainSchema(),
			"asOf": timestamp, "ledgerWatermark": timestamp,
			"ingestionLag": {Type: "integer", Format: "int64", Description: "Observed ingestion lag in nanoseconds."},
			"final":        {Type: "boolean"},
		}),
		"UsageSeriesPoint": objectSchema([]string{"bucketStart", "totals"}, map[string]JSONSchema{
			"bucketStart": timestamp, "totals": refSchema("UsageTotals"),
		}),
		"UsageSeries": objectSchema([]string{"points", "grain", "final"}, map[string]JSONSchema{
			"points": arraySchema(refSchema("UsageSeriesPoint")), "grain": observabilityGrainSchema(),
			"asOf": timestamp, "ledgerWatermark": timestamp,
			"ingestionLag": {Type: "integer", Format: "int64", Description: "Observed ingestion lag in nanoseconds."},
			"final":        {Type: "boolean"},
		}),
		"UsageBreakdownRow": objectSchema([]string{"value", "totals"}, map[string]JSONSchema{
			"value": stringSchema, "totals": refSchema("UsageTotals"),
		}),
		"UsageBreakdown": objectSchema([]string{"dimension", "rows", "grain", "final"}, map[string]JSONSchema{
			"dimension": observabilityBreakdownSchema(), "rows": arraySchema(refSchema("UsageBreakdownRow")),
			"grain": observabilityGrainSchema(), "asOf": timestamp, "ledgerWatermark": timestamp,
			"ingestionLag": {Type: "integer", Format: "int64", Description: "Observed ingestion lag in nanoseconds."},
			"final":        {Type: "boolean"},
		}),
		"RequestLog": requestLog,
		"RequestLogPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("RequestLog")), "page": refSchema("PageInfo"),
		}),
		"RequestDispatchAttempt": objectSchema([]string{"attemptId", "ordinal", "state", "startedAt", "completedAt"}, map[string]JSONSchema{
			"attemptId": stringSchema, "ordinal": {Type: "integer", Format: "int32"},
			"backendId": stringSchema, "providerId": stringSchema, "state": stringSchema,
			"statusCode": {Type: "integer", Format: "int32"}, "errorCode": stringSchema,
			"startedAt": timestamp, "completedAt": timestamp,
		}),
		"RequestDispatch": objectSchema([]string{
			"dispatchId", "ordinal", "dispatchType", "inputTokens", "cacheReadTokens",
			"cacheWriteTokens", "outputTokens", "usageState", "cost", "evidenceDigest",
			"startedAt", "completedAt", "attempts",
		}, map[string]JSONSchema{
			"dispatchId": stringSchema, "parentDispatchId": stringSchema,
			"ordinal": {Type: "integer", Format: "int32"}, "dispatchType": stringSchema,
			"modelId": stringSchema, "modelRevision": {Type: "integer", Format: "int64"},
			"backendId": stringSchema, "providerId": stringSchema, "providerModelId": stringSchema,
			"pricingRevision": {Type: "integer", Format: "int64"},
			"inputTokens":     whole, "cacheReadTokens": whole, "cacheWriteTokens": whole,
			"outputTokens": whole, "usageState": stringSchema, "cost": refSchema("CostSummary"),
			"evidenceDigest": stringSchema, "startedAt": timestamp, "completedAt": timestamp,
			"attempts": arraySchema(refSchema("RequestDispatchAttempt")),
		}),
		"RequestLogDetailData": objectSchema([]string{"request", "routing", "quotaReceipts"}, map[string]JSONSchema{
			"request":       refSchema("RequestLog"),
			"routing":       {Type: "object", AdditionalProperties: boolPointer(true)},
			"quotaReceipts": arraySchema(JSONSchema{Type: "object", AdditionalProperties: boolPointer(true)}),
			"dispatches":    arraySchema(refSchema("RequestDispatch")),
		}),
		"RequestLogDetail": objectSchema([]string{"data"}, map[string]JSONSchema{
			"data": refSchema("RequestLogDetailData"),
		}),
		"AuditEvent": auditEvent,
		"AuditEventPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(refSchema("AuditEvent")), "page": refSchema("PageInfo"),
		}),
		"PublicationReadiness":           publicationReadiness,
		"PublicationRuntimeDiagnostics":  publicationDiagnostics,
		"QuotaRuntimeDiagnostics":        quotaDiagnostics,
		"RuntimeStoreStatus":             storeStatus,
		"UsageStorageRuntimeDiagnostics": usageStorageStatus,
		"NamespaceRuntimeDiagnostics":    namespaceDiagnostics,
		"RuntimeDiagnostics": objectSchema([]string{
			"status", "asOf", "postgresql", "valkey", "usageStorage", "registeredNamespaces",
		}, map[string]JSONSchema{
			"status": {Type: "string", Enum: []string{"ready", "degraded"}},
			"asOf":   timestamp, "postgresql": refSchema("RuntimeStoreStatus"),
			"valkey":               refSchema("RuntimeStoreStatus"),
			"usageStorage":         refSchema("UsageStorageRuntimeDiagnostics"),
			"registeredNamespaces": {Type: "integer", Format: "int64"},
			"namespace":            refSchema("NamespaceRuntimeDiagnostics"),
		}),
	}
}

func observabilityResponseSchema(contract OperationContract) (JSONSchema, bool) {
	if contract.Method != MethodGET {
		return JSONSchema{}, false
	}
	schema := map[string]string{
		BasePath + "/usage":                                               "UsageSummary",
		BasePath + "/users/{userId}/usage":                                "UsageSummary",
		BasePath + "/teams/{teamId}/usage":                                "UsageSummary",
		BasePath + "/api-keys/{keyId}/usage":                              "UsageSummary",
		BasePath + "/usage/series":                                        "UsageSeries",
		BasePath + "/usage/breakdowns":                                    "UsageBreakdown",
		BasePath + "/request-logs":                                        "RequestLogPage",
		BasePath + "/audit-events":                                        "AuditEventPage",
		BasePath + "/runtime-diagnostics":                                 "RuntimeDiagnostics",
		BasePath + "/namespaces/{namespaceId}/request-logs/{admissionId}": "RequestLogDetail",
	}[contract.Path]
	if schema == "" {
		return JSONSchema{}, false
	}
	return refSchema(schema), true
}

func observabilityParameters(contract OperationContract) []OpenAPIParameter {
	if contract.Method == MethodGET && contract.Path == BasePath+"/runtime-diagnostics" {
		return []OpenAPIParameter{observabilityQueryParameter("namespaceId", JSONSchema{
			Type: "string", Pattern: `^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`,
		})}
	}
	if contract.Method != MethodGET || (!isUsageQueryPath(contract.Path) && contract.Path != BasePath+"/request-logs" && contract.Path != BasePath+"/audit-events") {
		return nil
	}
	parameters := []OpenAPIParameter{
		observabilityQueryParameter("start", JSONSchema{Type: "string", Format: "date-time"}),
		observabilityQueryParameter("end", JSONSchema{Type: "string", Format: "date-time"}),
		observabilityQueryParameter("timeZone", JSONSchema{Type: "string", Description: "IANA time-zone name."}),
	}
	if contract.Path == BasePath+"/audit-events" {
		for _, name := range []string{"actorPrincipalId", "action", "resourceType", "resourceId", "outcome", "requestId"} {
			parameters = append(parameters, observabilityQueryParameter(name, JSONSchema{Type: "string"}))
		}
		return append(parameters,
			observabilityQueryParameter("cursor", JSONSchema{Type: "string"}),
			observabilityQueryParameter("pageSize", boundedIntegerSchema(1, 200)),
		)
	}
	for _, name := range []string{
		"teamId", "userId", "apiKeyId", "entrypointId", "recipeId", "logicalModelId",
		"backendId", "providerId", "dispatchType", "protocol", "errorCode",
	} {
		if subjectUsagePathDimension(contract.Path) == name {
			continue
		}
		parameters = append(parameters, observabilityQueryParameter(name, JSONSchema{Type: "string"}))
	}
	parameters = append(parameters, observabilityQueryParameter("statusCode", boundedIntegerSchema(100, 599)))
	if isUsageQueryPath(contract.Path) {
		parameters = append(parameters, observabilityQueryParameter("grain", JSONSchema{Type: "string", Enum: []string{"auto", "minute", "hour", "day"}}))
	}
	if contract.Path == BasePath+"/usage/breakdowns" {
		parameters = append(parameters,
			observabilityQueryParameter("dimension", observabilityBreakdownSchema()),
			observabilityQueryParameter("pageSize", boundedIntegerSchema(1, 200)),
		)
	}
	if contract.Path == BasePath+"/request-logs" {
		parameters = append(parameters,
			observabilityQueryParameter("cursor", JSONSchema{Type: "string"}),
			observabilityQueryParameter("pageSize", boundedIntegerSchema(1, 200)),
		)
	}
	return parameters
}

func isUsageQueryPath(path string) bool {
	return strings.HasPrefix(path, BasePath+"/usage") || subjectUsagePathDimension(path) != ""
}

func subjectUsagePathDimension(path string) string {
	switch path {
	case BasePath + "/users/{userId}/usage":
		return "userId"
	case BasePath + "/teams/{teamId}/usage":
		return "teamId"
	case BasePath + "/api-keys/{keyId}/usage":
		return "apiKeyId"
	default:
		return ""
	}
}

func observabilityQueryParameter(name string, schema JSONSchema) OpenAPIParameter {
	return OpenAPIParameter{Name: name, In: "query", Schema: schema}
}

func observabilityGrainSchema() JSONSchema {
	return JSONSchema{Type: "string", Enum: []string{"minute", "hour", "day"}}
}

func observabilityBreakdownSchema() JSONSchema {
	return JSONSchema{Type: "string", Enum: []string{
		"api_key", "user", "team", "entrypoint", "recipe", "logical_model",
		"backend", "provider", "status", "dispatch_type",
	}}
}
