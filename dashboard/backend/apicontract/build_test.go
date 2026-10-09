package apicontract

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

type embeddedFields struct {
	Shared string `json:"shared"`
}

const testCookie = "test_session"

type sampleBody struct {
	embeddedFields
	Name     string          `json:"name"`
	Note     string          `json:"note,omitempty"`
	Count    *int            `json:"count"`
	Tags     []string        `json:"tags"`
	Raw      json.RawMessage `json:"raw,omitempty"`
	Labels   map[string]int  `json:"labels"`
	Ignored  string          `json:"-"`
	internal string          //nolint:unused // exercises the unexported-field skip
}

func TestSchemaForFollowsJSONTags(t *testing.T) {
	schema := SchemaFor[sampleBody]()
	if schema.Type != "object" {
		t.Fatalf("type = %q, want object", schema.Type)
	}
	for _, name := range []string{"shared", "name", "note", "count", "tags", "raw", "labels"} {
		if _, ok := schema.Properties[name]; !ok {
			t.Errorf("missing property %q", name)
		}
	}
	for _, name := range []string{"Ignored", "-", "internal"} {
		if _, ok := schema.Properties[name]; ok {
			t.Errorf("unexpected property %q", name)
		}
	}
	if want := []string{"shared", "name", "count", "tags", "labels"}; !slices.Equal(schema.Required, want) {
		t.Errorf("required = %v, want %v", schema.Required, want)
	}
	if !schema.Properties["count"].Nullable || schema.Properties["count"].Type != "integer" {
		t.Errorf("count = %+v, want nullable integer", schema.Properties["count"])
	}
	if raw := schema.Properties["raw"]; raw.Type != "" {
		t.Errorf("raw = %+v, want an unconstrained schema", raw)
	}
	if items := schema.Properties["tags"].Items; items == nil || items.Type != "string" {
		t.Errorf("tags items = %+v, want string", items)
	}
}

func TestBuildRendersPolicyAndSchemaStatus(t *testing.T) {
	spec, err := Build(Info{Title: "test"}, testCookie, []Route{
		{
			Pattern: "/healthz", Method: "GET", Public: true, Sensitivity: "public", ResourceOwner: "public", AuditMode: "none",
			Operation: &Operation{ID: "getHealthz", Responses: map[int]Response{200: JSONResponse[sampleBody]("ok")}},
		},
		{
			Pattern: "/api/items/{id}", Method: "DELETE", Permission: "config.write", Sensitivity: "sensitive",
			ResourceOwner: "config", AuditMode: "required", AuditAction: "item.delete", CSRF: true, MaxBodyBytes: 1024,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	health := spec.Paths["/healthz"]["get"]
	if health.SchemaStatus != SchemaTyped || health.OperationID != "getHealthz" {
		t.Fatalf("health = %+v", health)
	}
	if len(health.Security) != 0 {
		t.Errorf("public operation security = %v, want none", health.Security)
	}
	if _, ok := health.Responses["401"]; ok {
		t.Error("public operation documents an authentication failure")
	}

	remove := spec.Paths["/api/items/{id}"]["delete"]
	if remove.SchemaStatus != SchemaUndocumented || remove.OperationID != "deleteApiItemsId" {
		t.Fatalf("delete = %+v", remove)
	}
	if !remove.CSRF || remove.AuditAction != "item.delete" || remove.Permission != "config.write" || remove.MaxBodyBytes != 1024 {
		t.Errorf("delete policy = %+v", remove)
	}
	if len(remove.Parameters) != 1 || remove.Parameters[0].Name != "id" || !remove.Parameters[0].Required {
		t.Errorf("delete parameters = %+v", remove.Parameters)
	}
	if len(remove.Security) != 2 {
		t.Errorf("protected security = %v, want bearer and cookie alternatives", remove.Security)
	}
}

func TestBuildDocumentsTypedProtectedAuthenticationFailures(t *testing.T) {
	spec, err := Build(Info{}, testCookie, []Route{{
		Pattern: "/api/settings", Method: "GET", Permission: "config.read", Sensitivity: "operational",
		ResourceOwner: "config", AuditMode: "none", AuditAction: "config.read.read",
		Operation: &Operation{Responses: map[int]Response{200: JSONResponse[sampleBody]("ok")}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	responses := spec.Paths["/api/settings"]["get"].Responses
	for _, status := range []string{"200", "401", "403"} {
		if _, ok := responses[status]; !ok {
			t.Errorf("missing %s response", status)
		}
	}
}

func TestBuildRendersSubtreePatterns(t *testing.T) {
	spec, err := Build(Info{}, testCookie, []Route{
		{Pattern: "/api/traces/", Method: "GET", Permission: "logs.read"},
		{Pattern: "/api/traces", Method: "GET", Permission: "logs.read"},
	})
	if err != nil {
		t.Fatal(err)
	}
	subtree := spec.Paths["/api/traces/{subpath}"]["get"]
	if subtree == nil || subtree.RoutePattern != "/api/traces/" || subtree.OperationID != "getApiTracesSubpath" {
		t.Fatalf("subtree = %+v", subtree)
	}
	if exact := spec.Paths["/api/traces"]["get"]; exact == nil || exact.RoutePattern != "" {
		t.Fatalf("exact = %+v", exact)
	}
}

func TestBuildRejectsAmbiguousDocuments(t *testing.T) {
	for name, routes := range map[string][]Route{
		"duplicate explicit id": {
			{Pattern: "/a", Method: "GET", Operation: &Operation{ID: "same"}},
			{Pattern: "/b", Method: "GET", Operation: &Operation{ID: "same"}},
		},
		"invalid explicit id": {
			{Pattern: "/a", Method: "GET", Operation: &Operation{ID: "has space"}},
		},
		"equivalent templates": {
			{Pattern: "/api/rooms/{id}", Method: "GET"},
			{Pattern: "/api/rooms/{name}", Method: "DELETE"},
		},
		"subtree overlaps template": {
			{Pattern: "/api/rooms/{id}", Method: "GET"},
			{Pattern: "/api/rooms/", Method: "GET"},
		},
		"relative pattern": {
			{Pattern: "api/rooms", Method: "GET"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Build(Info{}, testCookie, routes); err == nil {
				t.Fatal("Build succeeded, want an ambiguity error")
			}
		})
	}
}

func TestEitherResponseKeepsEveryMediaType(t *testing.T) {
	response := EitherResponse("failed", JSONResponse[sampleBody]("structured"), TextResponse("plain"))
	if response.Description != "failed" || response.Content["application/json"].Schema == nil || response.Content["text/plain"].Schema == nil {
		t.Fatalf("response = %+v", response)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("repeated media type was accepted")
		}
	}()
	EitherResponse("failed", TextResponse("a"), TextResponse("b"))
}

func TestBuildIsDeterministic(t *testing.T) {
	routes := []Route{
		{Pattern: "/api/a", Method: "GET", Permission: "config.read"},
		{Pattern: "/api/b/{id}", Method: "POST", Permission: "config.write", CSRF: true},
	}
	first, _ := Build(Info{}, testCookie, routes)
	second, _ := Build(Info{}, testCookie, routes)
	a, _ := json.Marshal(first)
	b, _ := json.Marshal(second)
	if string(a) != string(b) || !strings.Contains(string(a), `"x-vllm-sr-csrf":true`) {
		t.Fatalf("documents differ or omit policy:\n%s\n%s", a, b)
	}
}

func TestBuildNamesTheSessionCookie(t *testing.T) {
	spec, err := Build(Info{}, "custom_session", nil)
	if err != nil {
		t.Fatal(err)
	}
	if name := spec.Components.SecuritySchemes[cookieScheme].Name; name != "custom_session" {
		t.Fatalf("cookie scheme name = %q, want custom_session", name)
	}
}

func TestBuildOmitsAuditActionWhenNothingIsAudited(t *testing.T) {
	spec, err := Build(Info{}, testCookie, []Route{
		{Pattern: "/api/settings", Method: "GET", Permission: "config.read", AuditMode: "none", AuditAction: "config.read.read"},
		{Pattern: "/api/secrets", Method: "GET", Permission: "config.read", AuditMode: "required", AuditAction: "config.read.read"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if action := spec.Paths["/api/settings"]["get"].AuditAction; action != "" {
		t.Errorf("unaudited read publishes audit action %q", action)
	}
	if action := spec.Paths["/api/secrets"]["get"].AuditAction; action != "config.read.read" {
		t.Errorf("audited read audit action = %q", action)
	}
}

func TestBuildRecordsPatternForEveryRewrittenPath(t *testing.T) {
	spec, err := Build(Info{}, testCookie, []Route{
		{Pattern: "/api/files/{path...}", Method: "GET", Permission: "config.read"},
		{Pattern: "/api/rooms/{id}", Method: "GET", Permission: "config.read"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if pattern := spec.Paths["/api/files/{path}"]["get"].RoutePattern; pattern != "/api/files/{path...}" {
		t.Errorf("wildcard route pattern = %q", pattern)
	}
	if pattern := spec.Paths["/api/rooms/{id}"]["get"].RoutePattern; pattern != "" {
		t.Errorf("unchanged path records pattern %q", pattern)
	}
}

func TestOperationIDsAndTags(t *testing.T) {
	for _, test := range []struct{ method, path, id, tag string }{
		{"GET", "/healthz", "getHealthz", "healthz"},
		{"GET", "/openapi.json", "getOpenapiJson", "openapi"},
		{"DELETE", "/api/mcp/servers/{id}", "deleteApiMcpServersId", "mcp"},
		{"POST", "/api/sr-bench/v1/runs/{id}/recover-plan", "postApiSrBenchV1RunsIdRecoverPlan", "sr-bench"},
		{"GET", "/", "get", "root"},
	} {
		if id := defaultOperationID(test.method, test.path); id != test.id {
			t.Errorf("defaultOperationID(%s %s) = %q, want %q", test.method, test.path, id, test.id)
		}
		if tag := tagFor(test.path); tag != test.tag {
			t.Errorf("tagFor(%s) = %q, want %q", test.path, tag, test.tag)
		}
	}
}

func TestSchemaAccepts(t *testing.T) {
	open := Schema{Required: []string{"error", "message"}, Properties: map[string]Schema{
		"error": {Type: "string"}, "message": {Type: "string"},
	}}
	closed := open
	closed.AdditionalProperties = false

	for _, test := range []struct {
		name   string
		schema Schema
		keys   []string
		want   bool
	}{
		{"open schema ignores extra keys", open, []string{"error", "message", "stage"}, true},
		{"missing a required key", open, []string{"error"}, false},
		{"closed schema rejects an extra key", closed, []string{"error", "message", "stage"}, false},
		{"closed schema accepts its exact keys", closed, []string{"error", "message"}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := test.schema.Accepts(test.keys); got != test.want {
				t.Errorf("Accepts(%v) = %v, want %v", test.keys, got, test.want)
			}
		})
	}
}
