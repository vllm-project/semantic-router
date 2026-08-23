package accesspublisher

import "testing"

func TestBarrierAcknowledgementsRequiredUsesImmutablePlan(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name    string
		plan    string
		want    bool
		wantErr bool
	}{
		{name: "expansive", plan: `{"barriers":[],"supersedes":[],"priorAccessGate":"","priorRoutingGate":""}`},
		{
			name: "restrictive",
			plan: `{"barriers":[{"kind":"api_key","resourceId":"key-1","reason":"disabled"}],"supersedes":[],"priorAccessGate":"pub-1","priorRoutingGate":"pub-1"}`,
			want: true,
		},
		{name: "missing", plan: "", wantErr: true},
		{name: "malformed", plan: `{"barriers":`, wantErr: true},
		{
			name:    "unknown field",
			plan:    `{"barriers":[],"supersedes":[],"priorAccessGate":"","priorRoutingGate":"","unexpected":true}`,
			wantErr: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := barrierAcknowledgementsRequired(test.plan)
			if (err != nil) != test.wantErr {
				t.Fatalf("barrierAcknowledgementsRequired() error = %v, wantErr %t", err, test.wantErr)
			}
			if got != test.want {
				t.Fatalf("barrierAcknowledgementsRequired() = %t, want %t", got, test.want)
			}
		})
	}
}
