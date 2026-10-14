package classification

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type securityWindowResult[T any] struct {
	result T
	err    error
}

// Native security questions keep the whole request when it fits. A model
// that rejects the input length can still inspect every overlapping security
// window. Each call retains the strict input-coverage contract; a failed
// window remains unknown. Explicit scan budgets and other errors never retry.
func nativeSecurityWindows[T any](ctx context.Context, text string, native bool, classify func(context.Context, string) (T, error)) []securityWindowResult[T] {
	result, err := classify(ctx, text)
	if ctx.Err() != nil {
		return []securityWindowResult[T]{{result, signalDeadline(ctx, ctx.Err())}}
	}
	if !native || !errors.Is(err, binding.ErrInputLimit) {
		return []securityWindowResult[T]{{result, signalDeadline(ctx, err)}}
	}
	chunks := jailbreakSignalChunks(text)
	if len(chunks) <= 1 {
		return []securityWindowResult[T]{{result, err}}
	}
	windows := make([]securityWindowResult[T], len(chunks))
	// Bound admitted work without parking bundle participants behind another
	// semaphore. Each worker can ask the next window once its current one ends.
	workers := min(4, len(chunks))
	modelservice.Fan(ctx, workers, func(worker int) {
		for i := worker; i < len(chunks); i += workers {
			if ctx.Err() != nil {
				windows[i].err = signalDeadline(ctx, ctx.Err())
				continue
			}
			windows[i].result, windows[i].err = classify(ctx, chunks[i])
			if ctx.Err() != nil {
				windows[i].err = ctx.Err()
			}
			windows[i].err = signalDeadline(ctx, windows[i].err)
		}
	})
	return windows
}

func (c *Classifier) classifyJailbreakWindows(ctx context.Context, text string) []cachedJailbreakResult {
	if backend := jailbreakDecisionBackend(c.jailbreakInference); backend != nil {
		decision, err := backend.Decide(ctx, text)
		return []cachedJailbreakResult{{decision: &decision, err: signalDeadline(ctx, err)}}
	}
	windows := nativeSecurityWindows(ctx, text, c.signalReadsWholeText(config.SignalTypeJailbreak), c.jailbreakInference.Classify)
	results := make([]cachedJailbreakResult, len(windows))
	for i, window := range windows {
		results[i] = cachedJailbreakResult{result: window.result, err: window.err}
	}
	return results
}

func classifySafetyWindows(ctx context.Context, classifier labelClassifier, text string) (labelClassification, error) {
	reader, native := classifier.(interface{ readsWholeText() bool })
	windows := nativeSecurityWindows(ctx, text, native && reader.readsWholeText(), classifier.Classify)
	if len(windows) == 1 {
		return windows[0].result, windows[0].err
	}
	var result labelClassification
	for _, window := range windows {
		if window.err != nil {
			return labelClassification{}, window.err
		}
		if len(window.result.ScoreWindows) > 0 {
			result.ScoreWindows = append(result.ScoreWindows, window.result.ScoreWindows...)
		} else {
			result.ScoreWindows = append(result.ScoreWindows, window.result.Scores)
		}
	}
	return result, nil
}
