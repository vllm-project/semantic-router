//go:build race

package extproc

// raceDetector reports a test binary built with -race, which slows the
// Router's count of a long prompt far more than it slows a guard's waits.
const raceDetector = true
