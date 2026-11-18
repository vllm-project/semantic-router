package testcases

import "testing"

func TestSummarizeHallucinationResults(t *testing.T) {
	tests := []struct {
		name       string
		results    []HallucinationResult
		wantWarned int
		wantFailed int
		wantInfra  int
	}{
		{
			name: "all warned",
			results: []HallucinationResult{
				{HallucinationWarned: true},
				{HallucinationWarned: true},
			},
			wantWarned: 2, wantFailed: 0, wantInfra: 0,
		},
		{
			name: "issue reproduction: 1 warned, 1 real miss, 2 infra",
			results: []HallucinationResult{
				{Description: "warned", HallucinationWarned: true},
				{Description: "real miss", HallucinationWarned: false, Error: ""},
				{Description: "connection refused", HallucinationWarned: false, Error: "connection refused"},
				{Description: "status 500", HallucinationWarned: false, Error: "status 500"},
			},
			wantWarned: 1, wantFailed: 1, wantInfra: 2,
		},
		{
			name:       "empty results",
			results:    nil,
			wantWarned: 0, wantFailed: 0, wantInfra: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			warned, failed, infra := summarizeHallucinationResults(tt.results)
			if warned != tt.wantWarned || failed != tt.wantFailed || infra != tt.wantInfra {
				t.Fatalf("got warned=%d failed=%d infra=%d, want %d/%d/%d",
					warned, failed, infra, tt.wantWarned, tt.wantFailed, tt.wantInfra)
			}
		})
	}
}
