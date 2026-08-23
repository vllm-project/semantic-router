package accesscontrol

import "testing"

func TestScopeContainment(t *testing.T) {
	team := TeamScope("ns-1", "team-1")
	user := UserScope("ns-1", "user-1")
	key := ResourceScope("ns-1", ScopeResourceAPIKey, "key-1")
	credential := ResourceScope("ns-1", ScopeResourceAPIKeyCredential, "credential-1")

	tests := []struct {
		name      string
		container Scope
		target    ScopedTarget
		want      bool
	}{
		{name: "cluster contains namespace", container: ClusterScope(), target: ScopedTarget{Scope: NamespaceScope("ns-1")}, want: true},
		{name: "namespace contains resource", container: NamespaceScope("ns-1"), target: ScopedTarget{Scope: key}, want: true},
		{name: "namespace rejects another namespace", container: NamespaceScope("ns-2"), target: ScopedTarget{Scope: key}},
		{name: "team contains team-owned key", container: team, target: ScopedTarget{Scope: key, Ancestors: []Scope{team}}, want: true},
		{name: "team does not contain user-owned key", container: team, target: ScopedTarget{Scope: key, Ancestors: []Scope{user}}},
		{name: "membership never makes team contain user", container: team, target: ScopedTarget{Scope: user}},
		{name: "user contains user-owned key", container: user, target: ScopedTarget{Scope: key, Ancestors: []Scope{user}}, want: true},
		{name: "key contains credential child", container: key, target: ScopedTarget{Scope: credential, Ancestors: []Scope{key, user}}, want: true},
		{name: "resource scope is otherwise exact", container: key, target: ScopedTarget{Scope: ResourceScope("ns-1", ScopeResourceAPIKey, "key-2")}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.container.Contains(test.target); got != test.want {
				t.Fatalf("Contains() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestOperationTargetsRequireContainmentOfEveryTarget(t *testing.T) {
	namespace := NamespaceScope("ns-1")
	targets := []ScopedTarget{
		{Scope: ResourceScope("ns-1", ScopeResourceAPIKey, "key-1")},
		{Scope: ResourceScope("ns-1", ScopeResourceRateLimitPolicy, "rate-1")},
	}
	if !namespace.ContainsAll(targets) {
		t.Fatal("namespace should contain every same-namespace target")
	}
	team := TeamScope("ns-1", "team-1")
	if team.ContainsAll(targets) {
		t.Fatal("team must not gain authority over unrelated operation targets")
	}
}
