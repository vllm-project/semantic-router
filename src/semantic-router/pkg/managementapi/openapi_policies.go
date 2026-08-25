package managementapi

import "sort"

// ISODuration performs the final canonical-value check. The schema pattern
// keeps generated clients on the same fixed day/time syntax and excludes
// calendar years and months, which have different quota semantics.
const canonicalISODurationPattern = `^P(?:(0|[1-9][0-9]*)D)?(?:T(?:(0|[1-9][0-9]*)H)?(?:(0|[1-9][0-9]*)M)?(?:(0|[1-9][0-9]*)(?:\.[0-9]{1,9})?S)?)?$`

func init() {
	registerOpenAPIExtension(openAPIExtension{
		Name: "policies", Schemas: policySchemas,
		RequestSchema: policyRequestSchema, ResponseSchema: policyResponseSchema,
		ExtraParameters: policyParameters,
	})
}

func policySchemas() map[string]JSONSchema {
	textSchema := JSONSchema{Type: "string"}
	uuidSchema := JSONSchema{Type: "string", Format: "uuid"}
	dateTimeSchema := JSONSchema{Type: "string", Format: "date-time"}
	revisionSchema := JSONSchema{Type: "integer", Format: "int64", Minimum: intPointer(1)}
	policyStatus := JSONSchema{Type: "string", Enum: []string{"draft", "active", "disabled"}}
	bindingStatus := JSONSchema{Type: "string", Enum: []string{"active", "disabled"}}
	subject := objectSchema([]string{"type", "id"}, map[string]JSONSchema{
		"type": {Type: "string", Enum: []string{"user", "team", "api_key"}}, "id": uuidSchema,
	})
	grant := objectSchema([]string{"resourceType", "resourceId", "permission", "effect"}, map[string]JSONSchema{
		"resourceType": {Type: "string", Enum: []string{"entrypoint", "model"}},
		"resourceId":   textSchema, "permission": {Type: "string", Enum: []string{"discover", "invoke"}},
		"effect": {Type: "string", Enum: []string{"allow", "deny"}},
	})
	grants := arraySchema(grant)
	grants.MaxItems = intPointer(512)

	rateRuleInput := policyRateRuleSchema(false)
	rateRule := policyRateRuleSchema(true)
	rules := arraySchema(rateRuleInput)
	rules.MaxItems = intPointer(128)
	inlineRules := rules
	inlineRules.MinItems = intPointer(1)
	inlinePolicy := objectSchema([]string{"name", "rules"}, map[string]JSONSchema{
		"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
		"description": {Type: "string", MaxLength: intPointer(1000)}, "rules": inlineRules,
	})

	accessPolicy := objectSchema([]string{"policyId", "name", "description", "status", "revision", "grants", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"policyId": uuidSchema, "name": textSchema, "description": textSchema,
		"status": policyStatus, "revision": revisionSchema, "grants": grants,
		"createdAt": dateTimeSchema, "updatedAt": dateTimeSchema,
	})
	ratePolicyRules := arraySchema(rateRule)
	ratePolicyRules.MaxItems = intPointer(128)
	ratePolicy := objectSchema([]string{"policyId", "name", "description", "status", "revision", "rules", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"policyId": uuidSchema, "name": textSchema, "description": textSchema,
		"status": policyStatus, "revision": revisionSchema, "rules": ratePolicyRules,
		"createdAt": dateTimeSchema, "updatedAt": dateTimeSchema,
	})
	accessBinding := objectSchema([]string{"bindingId", "policyId", "subject", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"bindingId": uuidSchema, "policyId": uuidSchema, "subject": subject,
		"status": bindingStatus, "revision": revisionSchema,
		"createdAt": dateTimeSchema, "updatedAt": dateTimeSchema,
	})
	rateBinding := objectSchema([]string{"bindingId", "policyId", "subject", "mode", "quotaPartitionId", "status", "revision", "createdAt", "updatedAt"}, map[string]JSONSchema{
		"bindingId": uuidSchema, "policyId": uuidSchema, "subject": subject,
		"mode":             {Type: "string", Enum: []string{"allocation", "hard_cap"}},
		"quotaPartitionId": uuidSchema, "status": bindingStatus, "revision": revisionSchema,
		"createdAt": dateTimeSchema, "updatedAt": dateTimeSchema,
	})
	existingRateBinding := objectSchema([]string{"policyId", "subject", "mode"}, map[string]JSONSchema{
		"policyId": uuidSchema, "subject": subject,
		"mode": {Type: "string", Enum: []string{"allocation", "hard_cap"}},
	})
	inlineRateBinding := objectSchema([]string{"inlinePolicy", "subject", "mode"}, map[string]JSONSchema{
		"inlinePolicy": inlinePolicy, "subject": subject,
		"mode": {Type: "string", Enum: []string{"allocation", "hard_cap"}},
	})
	accessBindingCreate := objectSchema([]string{"policyId", "subject"}, map[string]JSONSchema{
		"policyId": uuidSchema, "subject": subject,
	})
	rateBindingCreate := JSONSchema{OneOf: []JSONSchema{existingRateBinding, inlineRateBinding}}
	accessBulkItem := objectSchema([]string{"itemId", "policyId", "subject"}, map[string]JSONSchema{
		"itemId": uuidSchema, "policyId": uuidSchema, "subject": subject,
	})
	accessBulkItems := arraySchema(accessBulkItem)
	accessBulkItems.MinItems, accessBulkItems.MaxItems = intPointer(1), intPointer(1000)
	existingRateBulkItem := objectSchema([]string{"itemId", "policyId", "subject", "mode"}, map[string]JSONSchema{
		"itemId": uuidSchema, "policyId": uuidSchema, "subject": subject,
		"mode": {Type: "string", Enum: []string{"allocation", "hard_cap"}},
	})
	inlineRateBulkItem := objectSchema([]string{"itemId", "inlinePolicy", "subject", "mode"}, map[string]JSONSchema{
		"itemId": uuidSchema, "inlinePolicy": inlinePolicy, "subject": subject,
		"mode": {Type: "string", Enum: []string{"allocation", "hard_cap"}},
	})
	rateBulkItems := arraySchema(JSONSchema{OneOf: []JSONSchema{existingRateBulkItem, inlineRateBulkItem}})
	rateBulkItems.MinItems, rateBulkItems.MaxItems = intPointer(1), intPointer(1000)

	return policySchemaMap(policySchemaParts{
		stringSchema: textSchema, uuidSchema: uuidSchema, revisionSchema: revisionSchema,
		policyStatus: policyStatus, bindingStatus: bindingStatus, subject: subject, grant: grant,
		grants: grants, rules: rules, inlinePolicy: inlinePolicy, rateRuleInput: rateRuleInput, rateRule: rateRule,
		accessPolicy: accessPolicy, ratePolicy: ratePolicy, accessBinding: accessBinding, rateBinding: rateBinding,
		rateBindingCreate: rateBindingCreate, accessBindingCreate: accessBindingCreate,
		accessBulkItems: accessBulkItems, rateBulkItems: rateBulkItems,
	})
}

type policySchemaParts struct {
	stringSchema, uuidSchema, revisionSchema, policyStatus, bindingStatus JSONSchema
	subject, grant, grants, rules, inlinePolicy                           JSONSchema
	rateRuleInput, rateRule, accessPolicy, ratePolicy                     JSONSchema
	accessBinding, rateBinding, rateBindingCreate, accessBindingCreate    JSONSchema
	accessBulkItems, rateBulkItems                                        JSONSchema
}

func policySchemaMap(parts policySchemaParts) map[string]JSONSchema {
	return map[string]JSONSchema{
		"PolicySubject": parts.subject, "AccessPolicyGrant": parts.grant,
		"AccessPolicyCreateRequest": objectSchema([]string{"name"}, map[string]JSONSchema{
			"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"description": {Type: "string", MaxLength: intPointer(1000)}, "status": parts.policyStatus, "grants": parts.grants,
		}),
		"AccessPolicyPatchRequest": objectSchema(nil, map[string]JSONSchema{
			"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"description": {Type: "string", MaxLength: intPointer(1000)}, "status": parts.policyStatus, "grants": parts.grants,
		}),
		"AccessPolicy": parts.accessPolicy,
		"AccessPolicyPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(parts.accessPolicy), "page": refSchema("PageInfo"),
		}),
		"AccessPolicyDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": parts.accessPolicy}),
		"RateLimitRuleInput": parts.rateRuleInput, "RateLimitRule": parts.rateRule,
		"RateLimitPolicyCreateRequest": objectSchema([]string{"name"}, map[string]JSONSchema{
			"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"description": {Type: "string", MaxLength: intPointer(1000)}, "status": parts.policyStatus, "rules": parts.rules,
		}),
		"RateLimitPolicyPatchRequest": objectSchema(nil, map[string]JSONSchema{
			"name":        {Type: "string", MinLength: intPointer(1), MaxLength: intPointer(200)},
			"description": {Type: "string", MaxLength: intPointer(1000)}, "status": parts.policyStatus, "rules": parts.rules,
		}),
		"RateLimitPolicy": parts.ratePolicy,
		"RateLimitPolicyPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(parts.ratePolicy), "page": refSchema("PageInfo"),
		}),
		"RateLimitPolicyDetail":            objectSchema([]string{"data"}, map[string]JSONSchema{"data": parts.ratePolicy}),
		"AccessPolicyBindingCreateRequest": parts.accessBindingCreate,
		"AccessPolicyBindingPatchRequest":  objectSchema([]string{"status"}, map[string]JSONSchema{"status": parts.bindingStatus}),
		"AccessPolicyBinding":              parts.accessBinding,
		"AccessPolicyBindingPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(parts.accessBinding), "page": refSchema("PageInfo"),
		}),
		"AccessPolicyBindingDetail":     objectSchema([]string{"data"}, map[string]JSONSchema{"data": parts.accessBinding}),
		"InlineRateLimitPolicy":         parts.inlinePolicy,
		"RateLimitBindingCreateRequest": parts.rateBindingCreate,
		"RateLimitBindingPatchRequest":  objectSchema([]string{"status"}, map[string]JSONSchema{"status": parts.bindingStatus}),
		"RateLimitBinding":              parts.rateBinding,
		"RateLimitBindingPage": objectSchema([]string{"data", "page"}, map[string]JSONSchema{
			"data": arraySchema(parts.rateBinding), "page": refSchema("PageInfo"),
		}),
		"RateLimitBindingDetail": objectSchema([]string{"data"}, map[string]JSONSchema{"data": parts.rateBinding}),
		"RateLimitBindingCreateReceipt": objectSchema([]string{"bindingId", "policyId", "revision", "createdPolicy"}, map[string]JSONSchema{
			"bindingId": parts.uuidSchema, "policyId": parts.uuidSchema, "revision": parts.revisionSchema,
			"createdPolicy": {Type: "boolean"}, "idempotency": refSchema("IdempotencyMetadata"),
		}),
		"AccessPolicyBindingBulkApplyRequest": objectSchema([]string{"items"}, map[string]JSONSchema{"items": parts.accessBulkItems}),
		"RateLimitBindingBulkApplyRequest":    objectSchema([]string{"items"}, map[string]JSONSchema{"items": parts.rateBulkItems}),
	}
}

func policyRateRuleSchema(output bool) JSONSchema {
	positiveInteger := JSONSchema{Type: "string", Pattern: `^[1-9][0-9]{0,41}$`}
	cost := JSONSchema{
		Type: "string", MaxLength: intPointer(43),
		Pattern: `^(?:0\.(?=[0-9]{1,15}$)(?=[0-9]*[1-9])[0-9]+|[1-9][0-9]*(?:\.[0-9]{1,15})?)$`,
	}
	duration := JSONSchema{Type: "string", Pattern: canonicalISODurationPattern}
	tokenMetrics := []string{"input_tokens", "output_tokens", "total_tokens", "served_input_tokens", "served_output_tokens", "served_total_tokens"}

	windowRule := func(metric []string, algorithm, accounting string, limit JSONSchema, fields map[string]JSONSchema) JSONSchema {
		properties := map[string]JSONSchema{
			"metric": {Type: "string", Enum: metric}, "algorithm": {Type: "string", Enum: []string{algorithm}},
			"limit": limit, "accounting": {Type: "string", Enum: []string{accounting}},
			"enforcement": {Type: "string", Enum: []string{"enforce", "shadow"}},
		}
		required := []string{"metric", "algorithm", "limit", "accounting", "enforcement"}
		fieldNames := make([]string, 0, len(fields))
		for name := range fields {
			fieldNames = append(fieldNames, name)
		}
		sort.Strings(fieldNames)
		for _, name := range fieldNames {
			schema := fields[name]
			properties[name] = schema
			required = append(required, name)
		}
		return rateRuleObjectSchema(output, required, properties)
	}

	variants := make([]JSONSchema, 0, 9)
	for _, selection := range []struct {
		metrics    []string
		accounting string
		limit      JSONSchema
	}{
		{[]string{"requests"}, "request", positiveInteger},
		{tokenMetrics, "response_actual", positiveInteger},
		{[]string{"cost"}, "response_actual", cost},
	} {
		variants = append(variants,
			windowRule(selection.metrics, "sliding_log", selection.accounting, selection.limit,
				map[string]JSONSchema{"window": duration}),
			windowRule(selection.metrics, "calendar_window", selection.accounting, selection.limit,
				map[string]JSONSchema{
					"period":   {Type: "string", Enum: []string{"day", "month"}},
					"timezone": {Type: "string", MinLength: intPointer(1)},
				}),
		)
	}
	variants = append(variants,
		rateRuleObjectSchema(output,
			[]string{"metric", "algorithm", "capacity", "refillAmount", "refillPeriod", "accounting", "enforcement"},
			map[string]JSONSchema{
				"metric": {Type: "string", Enum: []string{"requests"}}, "algorithm": {Type: "string", Enum: []string{"token_bucket"}},
				"capacity": positiveInteger, "refillAmount": positiveInteger, "refillPeriod": duration,
				"accounting": {Type: "string", Enum: []string{"request"}}, "enforcement": {Type: "string", Enum: []string{"enforce", "shadow"}},
			}),
		rateRuleObjectSchema(output,
			[]string{"metric", "algorithm", "emissionInterval", "burstTolerance", "accounting", "enforcement"},
			map[string]JSONSchema{
				"metric": {Type: "string", Enum: []string{"requests"}}, "algorithm": {Type: "string", Enum: []string{"gcra"}},
				"emissionInterval": duration, "burstTolerance": {Type: "integer", Minimum: intPointer(0)},
				"accounting": {Type: "string", Enum: []string{"request"}}, "enforcement": {Type: "string", Enum: []string{"enforce", "shadow"}},
			}),
		rateRuleObjectSchema(output,
			[]string{"metric", "algorithm", "limit", "accounting", "enforcement"},
			map[string]JSONSchema{
				"metric": {Type: "string", Enum: []string{"concurrent_requests"}}, "algorithm": {Type: "string", Enum: []string{"concurrency"}},
				"limit": positiveInteger, "accounting": {Type: "string", Enum: []string{"request"}},
				"enforcement": {Type: "string", Enum: []string{"enforce", "shadow"}},
			}),
	)
	return JSONSchema{OneOf: variants}
}

func rateRuleObjectSchema(output bool, required []string, properties map[string]JSONSchema) JSONSchema {
	if output {
		properties["ruleId"] = JSONSchema{Type: "string", Format: "uuid"}
		properties["ordinal"] = boundedIntegerSchema(0, 127)
		required = append(required, "ruleId", "ordinal")
	}
	return objectSchema(required, properties)
}

func policyRequestSchema(contract OperationContract) (string, bool) {
	switch {
	case contract.Method == MethodPOST && contract.Path == BasePath+"/access-policies":
		return "AccessPolicyCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/access-policies/{policyId}":
		return "AccessPolicyPatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/rate-limit-policies":
		return "RateLimitPolicyCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/rate-limit-policies/{policyId}":
		return "RateLimitPolicyPatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/access-policy-bindings":
		return "AccessPolicyBindingCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/access-policy-bindings/{bindingId}":
		return "AccessPolicyBindingPatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/rate-limit-bindings":
		return "RateLimitBindingCreateRequest", true
	case contract.Method == MethodPATCH && contract.Path == BasePath+"/rate-limit-bindings/{bindingId}":
		return "RateLimitBindingPatchRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/access-policy-bindings:bulk-apply":
		return "AccessPolicyBindingBulkApplyRequest", true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/rate-limit-bindings:bulk-apply":
		return "RateLimitBindingBulkApplyRequest", true
	default:
		return "", false
	}
}

func policyResponseSchema(contract OperationContract) (JSONSchema, bool) {
	switch {
	case contract.Method == MethodGET && contract.Path == BasePath+"/access-policies":
		return refSchema("AccessPolicyPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/access-policies/{policyId}":
		return refSchema("AccessPolicyDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/rate-limit-policies":
		return refSchema("RateLimitPolicyPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/rate-limit-policies/{policyId}":
		return refSchema("RateLimitPolicyDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/access-policy-bindings":
		return refSchema("AccessPolicyBindingPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/access-policy-bindings/{bindingId}":
		return refSchema("AccessPolicyBindingDetail"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/rate-limit-bindings":
		return refSchema("RateLimitBindingPage"), true
	case contract.Method == MethodGET && contract.Path == BasePath+"/rate-limit-bindings/{bindingId}":
		return refSchema("RateLimitBindingDetail"), true
	case contract.Method == MethodPOST && contract.Path == BasePath+"/rate-limit-bindings":
		return refSchema("RateLimitBindingCreateReceipt"), true
	case (contract.Method == MethodPOST || contract.Method == MethodPATCH) &&
		(contract.Path == BasePath+"/access-policies" || contract.Path == BasePath+"/access-policies/{policyId}" ||
			contract.Path == BasePath+"/rate-limit-policies" || contract.Path == BasePath+"/rate-limit-policies/{policyId}" ||
			contract.Path == BasePath+"/access-policy-bindings" || contract.Path == BasePath+"/access-policy-bindings/{bindingId}" ||
			contract.Path == BasePath+"/rate-limit-bindings/{bindingId}"):
		return refSchema("MutationReceipt"), true
	default:
		return JSONSchema{}, false
	}
}

func policyParameters(contract OperationContract) []OpenAPIParameter {
	if contract.Method != MethodGET {
		return nil
	}
	switch contract.Path {
	case BasePath + "/access-policies", BasePath + "/rate-limit-policies":
		return []OpenAPIParameter{{Name: "status", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"draft", "active", "disabled"}}}}
	case BasePath + "/access-policy-bindings":
		return policyBindingParameters(false)
	case BasePath + "/rate-limit-bindings":
		return policyBindingParameters(true)
	default:
		return nil
	}
}

func policyBindingParameters(rate bool) []OpenAPIParameter {
	parameters := []OpenAPIParameter{
		{Name: "policyId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}},
		{Name: "subjectType", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"user", "team", "api_key"}}},
		{Name: "subjectId", In: "query", Schema: JSONSchema{Type: "string", Format: "uuid"}},
		{Name: "status", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"active", "disabled"}}},
	}
	if rate {
		parameters = append(parameters, OpenAPIParameter{Name: "mode", In: "query", Schema: JSONSchema{Type: "string", Enum: []string{"allocation", "hard_cap"}}})
	}
	return parameters
}
