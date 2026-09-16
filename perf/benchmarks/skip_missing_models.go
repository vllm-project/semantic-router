//go:build !windows && cgo

package benchmarks

import (
	"os"
	"strings"
)

func missingBenchModels(err error) bool {
	if err == nil {
		return false
	}
	if os.IsNotExist(err) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "does not exist") || strings.Contains(msg, "not found")
}
