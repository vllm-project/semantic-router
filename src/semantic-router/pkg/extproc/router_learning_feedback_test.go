package extproc

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

const testStrategyLinUCB = "linucb"

func newFeedbackTestRuntime(strategy string, dim int, lambda float64) *routerLearningRuntime {
	rt := newRouterLearningRuntime(nil, nil, nil)
	rt.contextualState(strategy, dim, lambda)
	return rt
}

// TestFeedbackLoopLinUCBConvergence drives a complete score → pending →
// consume → rescore cycle over multiple steps and verifies the matrix state
// converges towards the expected reward direction.
func TestFeedbackLoopLinUCBConvergence(t *testing.T) {
	const dim = 2
	const alpha = 1.0
	const lambda = 1.0
	const strategy = testStrategyLinUCB
	const replayID = "feedback-test-1"
	const model = "model_a"
	rt := newFeedbackTestRuntime(strategy, dim, lambda)

	// Fixed feature: x = [2, 0], reward = 1.0 (good fit). After N repeats,
	// theta should converge to (approx) [1.0, 0].
	x := []float64{2, 0}
	armKey := contextualBanditKey(strategy, "decision_x", 1, model)

	// Before any updates, score = dotTheta(x) + alpha * sqrt(quadInv(x)).
	// At identity prior: theta = [0,0], quadInv = x^T I x / lambda = 4/1 = 4.
	// Score = 0 + 1.0 * 2.0 = 2.0.
	arm := rt.contextualStates[strategy].arm(armKey)
	initialScore := arm.dotTheta(x) + alpha*math.Sqrt(arm.quadInv(x))
	if math.Abs(initialScore-2.0) > 1e-9 {
		t.Fatalf("identity prior score = %v, want 2.0", initialScore)
	}

	for i := 0; i < 100; i++ {
		// Record pending (Phase 1).
		rt.recordPendingContextualUpdate(replayID, strategy, x)
		// Consume as GoodFit (Phase 2).
		rt.consumePendingContextualUpdate(replayID, "decision_x", 1, model, routerLearningOutcomeGoodFit)
	}

	// After 100 GoodFit updates with x=[2,0], reward=1.0:
	// A = lambda*I + 100 * xx^T = [[1+400, 0], [0, 1]]
	// A_inv ≈ [[1/401, 0], [0, 1]]
	// b = [200, 0]
	// theta = A_inv * b ≈ [200/401, 0] ≈ [0.4988, 0]
	theta := arm.theta()
	wantTheta0 := 200.0 / 401.0
	if math.Abs(theta[0]-wantTheta0) > 1e-2 {
		t.Errorf("after 100 good-fit updates: theta[0] = %v, want ~%v", theta[0], wantTheta0)
	}
	if math.Abs(theta[1]) > 1e-9 {
		t.Errorf("theta[1] = %v, want 0", theta[1])
	}

	// quadInv should shrink as A_inv shrinks.
	// A_inv[0,0] = 1/(1+400) = 1/401 ≈ 0.002494
	// quadInv([2,0]) = 4 * A_inv[0,0] ≈ 0.00998
	// bonus = alpha * sqrt(0.00998) ≈ 0.0999
	qi := arm.quadInv(x)
	wantQI := 4.0 / 401.0
	if math.Abs(qi-wantQI) > 1e-6 {
		t.Errorf("quadInv after 100 updates = %v, want ~%v", qi, wantQI)
	}
}

// ===== Reward mapping =====

func TestContextualBanditRewardMapping(t *testing.T) {
	if v := contextualBanditReward(routerLearningOutcomeGoodFit); v != 1.0 {
		t.Errorf("GoodFit reward = %v, want 1.0", v)
	}
	if v := contextualBanditReward(routerLearningOutcomeOverprovisioned); v != 0.5 {
		t.Errorf("Overprovisioned reward = %v, want 0.5", v)
	}
	if v := contextualBanditReward(routerLearningOutcomeUnderpowered); v != 0.1 {
		t.Errorf("Underpowered reward = %v, want 0.1", v)
	}
	if v := contextualBanditReward(routerLearningOutcomeFailed); v != 0 {
		t.Errorf("Failed reward = %v, want 0.0", v)
	}
}

// ===== Different rewards produce different theta =====

func TestFeedbackLoopRewardDifferential(t *testing.T) {
	const dim = 2
	const lambda = 1.0
	const strategy = "test_reward_diff"
	const model = "model_a"

	arm0 := func() *learningMatrix {
		rt := newFeedbackTestRuntime(strategy+"_good", dim, lambda)
		x := []float64{1, 0}
		key := contextualBanditKey(strategy+"_good", "d", 0, model)
		rt.recordPendingContextualUpdate("r1", strategy+"_good", x)
		for range 50 {
			rt.consumePendingContextualUpdate("r1", "d", 0, model, routerLearningOutcomeGoodFit)
			rt.recordPendingContextualUpdate("r1", strategy+"_good", x)
		}
		return rt.contextualStates[strategy+"_good"].arm(key)
	}()

	arm1 := func() *learningMatrix {
		rt := newFeedbackTestRuntime(strategy+"_failed", dim, lambda)
		x := []float64{1, 0}
		key := contextualBanditKey(strategy+"_failed", "d", 0, model)
		rt.recordPendingContextualUpdate("r2", strategy+"_failed", x)
		for range 50 {
			rt.consumePendingContextualUpdate("r2", "d", 0, model, routerLearningOutcomeFailed)
			rt.recordPendingContextualUpdate("r2", strategy+"_failed", x)
		}
		return rt.contextualStates[strategy+"_failed"].arm(key)
	}()

	gfTheta := arm0.theta()[0]
	ffTheta := arm1.theta()[0]
	if gfTheta <= ffTheta {
		t.Errorf("GoodFit theta[0] = %v should be > Failed theta[0] = %v", gfTheta, ffTheta)
	}
}

// ===== Pending cleanup =====

func TestFeedbackLoopPendingCleanup(t *testing.T) {
	rt := newFeedbackTestRuntime("test_cleanup", 4, 1.0)
	x := []float64{1, 0, 0, 0}

	// Record 100 pending updates.
	for i := range 100 {
		rid := "pending-" + intToString(i)
		rt.recordPendingContextualUpdate(rid, "test_cleanup", x)
	}

	rt.pendingUpdatesMu.Lock()
	pendingBefore := len(rt.pendingUpdates)
	rt.pendingUpdatesMu.Unlock()
	if pendingBefore != 100 {
		t.Fatalf("expected 100 pending, got %d", pendingBefore)
	}

	// Consume half.
	for i := range 50 {
		rid := "pending-" + intToString(i)
		rt.consumePendingContextualUpdate(rid, "d", 0, "m", routerLearningOutcomeGoodFit)
	}

	rt.pendingUpdatesMu.Lock()
	pendingAfter := len(rt.pendingUpdates)
	rt.pendingUpdatesMu.Unlock()
	if pendingAfter != 50 {
		t.Errorf("expected 50 remaining pending after consume, got %d", pendingAfter)
	}
}

// ===== No-op when no pending =====

func TestFeedbackLoopNoopOnMissingPending(t *testing.T) {
	rt := newFeedbackTestRuntime("test_noop", 4, 1.0)
	// Consume without recording — must not panic.
	rt.consumePendingContextualUpdate("never-recorded", "d", 0, "m", routerLearningOutcomeGoodFit)
	t.Log("no-op on missing pending: ok")
}

// ===== Bounded map safety =====

func TestFeedbackLoopBoundedMap(t *testing.T) {
	rt := newFeedbackTestRuntime("test_bound", 4, 1.0)
	x := []float64{1, 0, 0, 0}

	// Record 2500 entries — should trigger the 2000-entry safety cap.
	for i := range 2500 {
		rid := "bound-" + intToString(i)
		rt.recordPendingContextualUpdate(rid, "test_bound", x)
	}

	rt.pendingUpdatesMu.Lock()
	sz := len(rt.pendingUpdates)
	rt.pendingUpdatesMu.Unlock()
	if sz > 2000 {
		t.Errorf("pending map size %d exceeds cap 2000", sz)
	}
}

// ===== Concurrent safety =====

func TestFeedbackLoopConcurrent(t *testing.T) {
	rt := newFeedbackTestRuntime("test_concurrent", 4, 1.0)
	var wg sync.WaitGroup
	const goroutines = 50
	const iters = 100

	// Concurrent recorders.
	for range goroutines {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for j := range iters {
				rid := "conc-" + intToString(base*iters+j)
				x := []float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}
				rt.recordPendingContextualUpdate(rid, "test_concurrent", x)
			}
		}(goroutines * 2) // offset to avoid ID collision
	}

	// Concurrent consumers.
	for range goroutines {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for j := range iters {
				rid := "conc-" + intToString(base*iters+j)
				rt.consumePendingContextualUpdate(rid, "d", 0, "m", routerLearningOutcomeGoodFit)
			}
		}(0)
	}

	wg.Wait()

	rt.pendingUpdatesMu.Lock()
	remaining := len(rt.pendingUpdates)
	rt.pendingUpdatesMu.Unlock()

	// With 2*goroutines recorders and goroutines consumers, some entries
	// will be leftover. That's expected. The test passes if no panic/race.
	t.Logf("concurrent test done: %d pending entries remaining", remaining)
}

// ===== Exact theta convergence for every verdict =====

func TestFeedbackLoopPartialRewardExactTheta(t *testing.T) {
	const dim = 2
	const lambda = 1.0
	x := []float64{2, 0}

	// GoodFit (reward=1.0): after 100 updates,
	//   b = 100*2.0*[1,0] = [200,0]
	//   A = I + 100*[[4,0],[0,0]] = [[401,0],[0,1]]
	//   theta = A^{-1} b = [200/401, 0]
	rtGF := newFeedbackTestRuntime("gf", dim, lambda)
	armGF := rtGF.contextualStates["gf"].arm(contextualBanditKey("gf", "d", 0, "m"))
	for range 100 {
		rtGF.recordPendingContextualUpdate("r", "gf", x)
		rtGF.consumePendingContextualUpdate("r", "d", 0, "m", routerLearningOutcomeGoodFit)
	}
	gf := armGF.theta()[0]
	wantGF := 200.0 / 401.0
	if math.Abs(gf-wantGF) > 1e-9 {
		t.Errorf("GoodFit theta[0]=%v, want %v", gf, wantGF)
	}

	// Overprovisioned (reward=0.5): b = 100*0.5*[2,0] = [100,0]
	//   theta = [100/401, 0] — exactly half of GoodFit
	rtOV := newFeedbackTestRuntime("ov", dim, lambda)
	armOV := rtOV.contextualStates["ov"].arm(contextualBanditKey("ov", "d", 0, "m"))
	for range 100 {
		rtOV.recordPendingContextualUpdate("r", "ov", x)
		rtOV.consumePendingContextualUpdate("r", "d", 0, "m", routerLearningOutcomeOverprovisioned)
	}
	ov := armOV.theta()[0]
	wantOV := 100.0 / 401.0
	if math.Abs(ov-wantOV) > 1e-9 {
		t.Errorf("Overprovisioned theta[0]=%v, want %v", ov, wantOV)
	}

	// Underpowered (reward=0.1): b = 100*0.1*[2,0] = [20,0]
	//   theta = [20/401, 0]
	rtUN := newFeedbackTestRuntime("un", dim, lambda)
	armUN := rtUN.contextualStates["un"].arm(contextualBanditKey("un", "d", 0, "m"))
	for range 100 {
		rtUN.recordPendingContextualUpdate("r", "un", x)
		rtUN.consumePendingContextualUpdate("r", "d", 0, "m", routerLearningOutcomeUnderpowered)
	}
	un := armUN.theta()[0]
	wantUN := 20.0 / 401.0
	if math.Abs(un-wantUN) > 1e-9 {
		t.Errorf("Underpowered theta[0]=%v, want %v", un, wantUN)
	}

	// Proportionality check: reward doubled → theta doubled
	if math.Abs(gf/ov-2.0) > 1e-9 {
		t.Errorf("gf/ov=%v, want 2.0 (theta should be proportional to reward)", gf/ov)
	}
	if math.Abs(ov/un-5.0) > 1e-9 {
		t.Errorf("ov/un=%v, want 5.0", ov/un)
	}
}

// ===== Full control loop: contextual + experience both live =====

func TestFeedbackLoopFullControlLoop(t *testing.T) {
	const dim = 3
	const lambda = 1.0
	rt := newFeedbackTestRuntime("full", dim, lambda)
	x := []float64{1, 0.5, -0.3}
	arm := rt.contextualStates["full"].arm(contextualBanditKey("full", "d", 0, "m"))

	// Baseline: identity prior → theta = [0,0,0]
	for i, v := range arm.theta() {
		if v != 0 {
			t.Errorf("prior theta[%d] = %v, want 0", i, v)
		}
	}

	// 200 mixed-verdict cycles through the feedback loop.
	for i := range 200 {
		rt.recordPendingContextualUpdate("r", "full", x)
		var v routerLearningOutcomeVerdict
		switch i % 3 {
		case 0:
			v = routerLearningOutcomeGoodFit // 1.0
		case 1:
			v = routerLearningOutcomeOverprovisioned // 0.5
		case 2:
			v = routerLearningOutcomeUnderpowered // 0.1
		}
		rt.consumePendingContextualUpdate("r", "d", 0, "m", v)
	}

	// Avg reward = (1.0+0.5+0.1)/3 ≈ 0.5333. Theta should point roughly
	// in x's direction (cosine > 0.99) with magnitude bounded by avg reward.
	theta := arm.theta()
	nx := math.Sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
	dot, nt := 0.0, 0.0
	for i := range theta {
		dot += theta[i] * x[i]
		nt += theta[i] * theta[i]
	}
	nt = math.Sqrt(nt)
	cosine := dot / (nt * nx)
	if cosine < 0.99 {
		t.Errorf("theta should align with x (cosine=%v, want >0.99)", cosine)
	}
	if nt > 0.6 || nt < 0.3 {
		t.Errorf("theta norm=%v, expect ~0.4–0.55 for avg reward 0.533", nt)
	}
}
