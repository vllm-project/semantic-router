//go:build !windows

package apiserver

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/openapischema"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// A named string with a closed set of wire values is published as an enum.
// The history-reset recovery status lists only the values that can appear;
// the unset status is omitted from the wire, so it is not advertised.
func TestOpenAPIPublishesClosedStringSetsAsEnums(t *testing.T) {
	schema := openapischema.FromType(reflect.TypeOf(store.HistoryResetDiagnostics{}), routerSchemaOverrides)
	status, ok := schema.Properties["recovery_status"]
	if !ok {
		t.Fatalf("recovery_status missing from %+v", schema.Properties)
	}
	want := []string{"not_required", "stored", "failed"}
	if status.Type != "string" || !reflect.DeepEqual(status.Enum, want) {
		t.Fatalf("recovery_status schema = %+v, want enum %v", status, want)
	}
	if entries, ok := schema.Properties["recovery_entries"]; !ok || entries.Type != "integer" {
		t.Fatalf("recovery_entries schema = %+v", entries)
	}
	// An ordinary string field stays an unconstrained string.
	if reason := schema.Properties["reason"]; reason.Type != "string" || len(reason.Enum) != 0 {
		t.Fatalf("reason schema = %+v", reason)
	}
}
