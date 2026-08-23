package accesscontrol

import (
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestPublishedBuiltInRoleMatrixMatchesRuntime(t *testing.T) {
	document, err := os.ReadFile("../../../../website/docs/proposals/router-native-access-control-authorization.md")
	if err != nil {
		t.Fatalf("read authorization specification: %v", err)
	}
	const heading = "## Exact built-in roles"
	section := string(document)
	sectionIndex := strings.Index(section, heading)
	if sectionIndex < 0 {
		t.Fatal("authorization specification omits the built-in role section")
	}
	section = section[sectionIndex+len(heading):]
	const fence = "~~~yaml\n"
	fenceIndex := strings.Index(section, fence)
	if fenceIndex < 0 {
		t.Fatal("authorization specification omits the built-in role matrix")
	}
	section = section[fenceIndex+len(fence):]
	endIndex := strings.Index(section, "\n~~~")
	if endIndex < 0 {
		t.Fatal("authorization specification has an unterminated built-in role matrix")
	}

	documented := map[string][]string{}
	if err := yaml.Unmarshal([]byte(section[:endIndex]), &documented); err != nil {
		t.Fatalf("decode built-in role matrix: %v", err)
	}
	names := []BuiltInRoleName{
		BuiltInRoleClusterAdmin, BuiltInRolePlatformAdmin, BuiltInRoleOperator,
		BuiltInRoleAccessAdmin, BuiltInRoleCredentialRevealer, BuiltInRoleAnalyst,
		BuiltInRoleViewer, BuiltInRoleConsumer,
	}
	if len(documented) != len(names) {
		t.Fatalf("documented built-in role count = %d, want %d", len(documented), len(names))
	}
	for _, name := range names {
		values, exists := documented[string(name)]
		if !exists {
			t.Errorf("authorization specification omits %s", name)
			continue
		}
		permissions := make([]Permission, len(values))
		for index, value := range values {
			permissions[index] = Permission(value)
		}
		got, err := NewPermissionSet(permissions...)
		if err != nil {
			t.Errorf("authorization specification has an invalid %s permission: %v", name, err)
			continue
		}
		role, _ := BuiltInRole(name)
		if !role.Permissions.Equal(got) {
			t.Errorf("authorization specification %s permissions = %v, runtime = %v",
				name, got.Permissions(), role.Permissions.Permissions())
		}
	}
}
