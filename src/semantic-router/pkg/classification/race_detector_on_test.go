//go:build race

package classification

// raceDetector reports a test binary built with -race, which slows a guard
// scan's own work between reads far more than it slows the fake runtime's.
const raceDetector = true
