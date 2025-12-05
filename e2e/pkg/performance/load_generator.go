package performance

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// LoadGenerator generates load for performance testing
type LoadGenerator struct {
	concurrency int
	rateLimit   int // requests per second (0 = unlimited)
	duration    time.Duration
}

// NewLoadGenerator creates a new load generator
func NewLoadGenerator(concurrency, rateLimit int, duration time.Duration) *LoadGenerator {
	return &LoadGenerator{
		concurrency: concurrency,
		rateLimit:   rateLimit,
		duration:    duration,
	}
}

// LoadResult contains the results of a load test
type LoadResult struct {
	TotalRequests  int
	SuccessfulReqs int
	FailedReqs     int
	Duration       time.Duration
	AvgLatencyMs   float64
	P50LatencyMs   float64
	P90LatencyMs   float64
	P95LatencyMs   float64
	P99LatencyMs   float64
	MaxLatencyMs   float64
	MinLatencyMs   float64
	ThroughputQPS  float64
	Latencies      []time.Duration
	Errors         []error
}

// RequestFunc is a function that executes a single request
type RequestFunc func(ctx context.Context) error

// GenerateLoad generates load using the specified request function
func (lg *LoadGenerator) GenerateLoad(ctx context.Context, reqFunc RequestFunc) (*LoadResult, error) {
	result := &LoadResult{
		Latencies: make([]time.Duration, 0),
		Errors:    make([]error, 0),
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	var successCount, failCount atomic.Int64

	// Rate limiting setup
	var ticker *time.Ticker
	var tickerChan <-chan time.Time
	if lg.rateLimit > 0 {
		interval := time.Second / time.Duration(lg.rateLimit)
		ticker = time.Ticker(interval)
		tickerChan = ticker.C
		defer ticker.Stop()
	}

	// Create timeout context
	loadCtx, cancel := context.WithTimeout(ctx, lg.duration)
	defer cancel()

	// Create semaphore for concurrency control
	semaphore := make(chan struct{}, lg.concurrency)

	startTime := time.Now()
	requestCount := 0

	// Generate load loop
loadLoop:
	for {
		select {
		case <-loadCtx.Done():
			break loadLoop
		default:
			// Rate limiting
			if lg.rateLimit > 0 {
				select {
				case <-tickerChan:
					// Continue
				case <-loadCtx.Done():
					break loadLoop
				}
			}

			// Acquire semaphore
			select {
			case semaphore <- struct{}{}:
				// Got slot
			case <-loadCtx.Done():
				break loadLoop
			}

			requestCount++
			wg.Add(1)

			go func() {
				defer wg.Done()
				defer func() { <-semaphore }() // Release semaphore

				reqStart := time.Now()
				err := reqFunc(ctx)
				latency := time.Since(reqStart)

				mu.Lock()
				result.Latencies = append(result.Latencies, latency)
				if err != nil {
					result.Errors = append(result.Errors, err)
					failCount.Add(1)
				} else {
					successCount.Add(1)
				}
				mu.Unlock()
			}()
		}
	}

	// Wait for all requests to complete
	wg.Wait()

	result.Duration = time.Since(startTime)
	result.TotalRequests = requestCount
	result.SuccessfulReqs = int(successCount.Load())
	result.FailedReqs = int(failCount.Load())

	// Calculate statistics
	if len(result.Latencies) > 0 {
		calculateLatencyStats(result)
	}

	// Calculate throughput
	if result.Duration > 0 {
		result.ThroughputQPS = float64(result.TotalRequests) / result.Duration.Seconds()
	}

	return result, nil
}

// calculateLatencyStats calculates percentile statistics
func calculateLatencyStats(result *LoadResult) {
	latencies := make([]float64, len(result.Latencies))
	var sum float64

	for i, latency := range result.Latencies {
		ms := float64(latency.Microseconds()) / 1000.0
		latencies[i] = ms
		sum += ms
	}

	sort.Float64s(latencies)

	result.AvgLatencyMs = sum / float64(len(latencies))
	result.P50LatencyMs = percentile(latencies, 50)
	result.P90LatencyMs = percentile(latencies, 90)
	result.P95LatencyMs = percentile(latencies, 95)
	result.P99LatencyMs = percentile(latencies, 99)
	result.MinLatencyMs = latencies[0]
	result.MaxLatencyMs = latencies[len(latencies)-1]
}

// percentile calculates the Nth percentile from sorted data
func percentile(sortedData []float64, p int) float64 {
	if len(sortedData) == 0 {
		return 0
	}

	if p >= 100 {
		return sortedData[len(sortedData)-1]
	}

	index := int(math.Ceil(float64(len(sortedData))*float64(p)/100.0)) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sortedData) {
		index = len(sortedData) - 1
	}

	return sortedData[index]
}

// PrintResults prints the load test results
func (r *LoadResult) PrintResults() {
	fmt.Println("\n" + "===================================================================================")
	fmt.Println("                              LOAD TEST RESULTS")
	fmt.Println("===================================================================================")
	fmt.Printf("Duration:          %v\n", r.Duration.Round(time.Millisecond))
	fmt.Printf("Total Requests:    %d\n", r.TotalRequests)
	fmt.Printf("Successful:        %d (%.2f%%)\n", r.SuccessfulReqs, float64(r.SuccessfulReqs)/float64(r.TotalRequests)*100)
	fmt.Printf("Failed:            %d (%.2f%%)\n", r.FailedReqs, float64(r.FailedReqs)/float64(r.TotalRequests)*100)
	fmt.Printf("Throughput:        %.2f req/s\n", r.ThroughputQPS)
	fmt.Println("-----------------------------------------------------------------------------------")
	fmt.Println("Latency Statistics (ms):")
	fmt.Printf("  Min:     %8.2f\n", r.MinLatencyMs)
	fmt.Printf("  Average: %8.2f\n", r.AvgLatencyMs)
	fmt.Printf("  P50:     %8.2f\n", r.P50LatencyMs)
	fmt.Printf("  P90:     %8.2f\n", r.P90LatencyMs)
	fmt.Printf("  P95:     %8.2f\n", r.P95LatencyMs)
	fmt.Printf("  P99:     %8.2f\n", r.P99LatencyMs)
	fmt.Printf("  Max:     %8.2f\n", r.MaxLatencyMs)
	fmt.Println("===================================================================================")

	if len(r.Errors) > 0 {
		fmt.Printf("\nFirst 5 errors:\n")
		for i, err := range r.Errors {
			if i >= 5 {
				break
			}
			fmt.Printf("  %d. %v\n", i+1, err)
		}
	}
}

// RampUpLoadGenerator generates load with a ramp-up pattern
type RampUpLoadGenerator struct {
	startQPS int
	endQPS   int
	duration time.Duration
	steps    int
}

// NewRampUpLoadGenerator creates a new ramp-up load generator
func NewRampUpLoadGenerator(startQPS, endQPS int, duration time.Duration, steps int) *RampUpLoadGenerator {
	return &RampUpLoadGenerator{
		startQPS: startQPS,
		endQPS:   endQPS,
		duration: duration,
		steps:    steps,
	}
}

// GenerateLoad generates ramped load
func (rlg *RampUpLoadGenerator) GenerateLoad(ctx context.Context, reqFunc RequestFunc) ([]*LoadResult, error) {
	results := make([]*LoadResult, 0, rlg.steps)
	stepDuration := rlg.duration / time.Duration(rlg.steps)
	qpsIncrement := float64(rlg.endQPS-rlg.startQPS) / float64(rlg.steps)

	for i := 0; i < rlg.steps; i++ {
		currentQPS := rlg.startQPS + int(float64(i)*qpsIncrement)
		fmt.Printf("\nRamp-up step %d/%d: QPS=%d for %v\n", i+1, rlg.steps, currentQPS, stepDuration)

		lg := NewLoadGenerator(currentQPS, currentQPS, stepDuration)
		result, err := lg.GenerateLoad(ctx, reqFunc)
		if err != nil {
			return results, fmt.Errorf("load generation failed at step %d: %w", i+1, err)
		}

		results = append(results, result)
		result.PrintResults()

		// Brief pause between steps
		time.Sleep(time.Second)
	}

	return results, nil
}
