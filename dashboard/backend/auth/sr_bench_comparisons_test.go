package auth

import (
	"net/http"
	"reflect"
	"testing"
)

func TestSRBenchComparisonIsAnExactReadPermission(t *testing.T) {
	for _, tc := range []struct {
		method, path string
		permissions  []string
	}{
		{http.MethodPost, "/comparisons", []string{PermEvalRead}},
		{http.MethodPost, "/comparisons/", []string{PermEvalWrite}},
		{http.MethodPost, "/comparisons/anything", []string{PermEvalWrite}},
		{http.MethodDelete, "/comparisons", []string{PermEvalWrite}},
		{http.MethodPost, "/replays", []string{PermEvalWrite}},
		{http.MethodPost, "/experiments", []string{PermEvalWrite}},
		{http.MethodPost, "/runs", []string{PermEvalWrite, PermEvalRun}},
		{http.MethodPost, "/runs/run-1/cancel", []string{PermEvalRun}},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			actual := RequiredPermissions(tc.method, "/api/sr-bench/v1"+tc.path)
			if !reflect.DeepEqual(actual, tc.permissions) {
				t.Fatalf("permissions=%v want=%v", actual, tc.permissions)
			}
		})
	}
}
