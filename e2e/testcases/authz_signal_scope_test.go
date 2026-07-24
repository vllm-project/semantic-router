package testcases

import "testing"

func TestSignalExtractionMetricValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		metrics    string
		signalType string
		signalName string
		want       float64
		wantErr    bool
	}{
		{
			name: "finds exact signal labels",
			metrics: `# HELP llm_signal_extraction_total Total number of signal extractions by type and name
llm_signal_extraction_total{signal_name="admin_only_marker",signal_type="keyword"} 3
`,
			signalType: "keyword",
			signalName: "admin_only_marker",
			want:       3,
		},
		{
			name: "ignores other signals",
			metrics: `llm_signal_extraction_total{signal_name="code_request",signal_type="keyword"} 5
`,
			signalType: "keyword",
			signalName: "admin_only_marker",
			want:       0,
		},
		{
			name: "rejects malformed matching sample",
			metrics: `llm_signal_extraction_total{signal_name="admin_only_marker",signal_type="keyword"} invalid
`,
			signalType: "keyword",
			signalName: "admin_only_marker",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := signalExtractionMetricValue(tt.metrics, tt.signalType, tt.signalName)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected an error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("signalExtractionMetricValue() error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("signalExtractionMetricValue() = %v, want %v", got, tt.want)
			}
		})
	}
}
