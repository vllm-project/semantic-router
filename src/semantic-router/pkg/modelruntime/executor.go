package modelruntime

import (
	"context"
	"fmt"
	"runtime"
)

type TaskStatus string

const (
	TaskPending   TaskStatus = "pending"
	TaskRunning   TaskStatus = "running"
	TaskSucceeded TaskStatus = "succeeded"
	TaskFailed    TaskStatus = "failed"
	TaskSkipped   TaskStatus = "skipped"
)

type Task struct {
	Name         string
	Dependencies []string
	BestEffort   bool
	Run          func(context.Context) error
}

type Event struct {
	Task       string
	Status     TaskStatus
	BestEffort bool
	Error      error
}

type Options struct {
	MaxParallelism int
	OnEvent        func(Event)
}

type TaskResult struct {
	Name       string
	Status     TaskStatus
	BestEffort bool
	Error      error
}

type Summary struct {
	Results map[string]TaskResult
}

type taskState struct {
	task          Task
	remainingDeps int
	dependents    []string
	result        TaskResult
}

type taskOutcome struct {
	name string
	err  error
}

type executorRun struct {
	states         map[string]*taskState
	ready          []*taskState
	resultCh       chan taskOutcome
	emit           func(Event)
	maxParallelism int
	cancel         context.CancelFunc
	running        int
	finished       int
	firstErr       error
}

func DefaultParallelism(taskCount int) int {
	if taskCount <= 1 {
		return 1
	}
	if cpus := runtime.NumCPU(); cpus > 0 && cpus < taskCount {
		return cpus
	}
	return taskCount
}

func Execute(ctx context.Context, tasks []Task, options Options) (Summary, error) {
	summary := Summary{Results: make(map[string]TaskResult, len(tasks))}
	if len(tasks) == 0 {
		return summary, nil
	}

	states, err := buildTaskStates(tasks)
	if err != nil {
		return summary, err
	}

	ready := initialReadyTasks(states)
	if len(ready) == 0 {
		return summary, fmt.Errorf("modelruntime: no runnable tasks; dependency graph is cyclic or invalid")
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	run := newExecutorRun(states, ready, normalizeParallelism(len(tasks), options.MaxParallelism), cancel, len(tasks), options.OnEvent)
	if err := run.execute(runCtx, len(tasks)); err != nil {
		return summary, err
	}

	for name, state := range states {
		summary.Results[name] = state.result
	}

	return summary, run.firstErr
}

func newExecutorRun(
	states map[string]*taskState,
	ready []*taskState,
	maxParallelism int,
	cancel context.CancelFunc,
	taskCount int,
	onEvent func(Event),
) *executorRun {
	return &executorRun{
		states:         states,
		ready:          ready,
		resultCh:       make(chan taskOutcome, taskCount),
		emit:           buildEventEmitter(onEvent),
		maxParallelism: maxParallelism,
		cancel:         cancel,
	}
}

func normalizeParallelism(taskCount int, maxParallelism int) int {
	if maxParallelism <= 0 {
		maxParallelism = DefaultParallelism(taskCount)
	}
	if maxParallelism > taskCount {
		return taskCount
	}
	return maxParallelism
}

func buildEventEmitter(onEvent func(Event)) func(Event) {
	if onEvent == nil {
		return func(Event) {}
	}
	return onEvent
}

func (r *executorRun) execute(ctx context.Context, taskCount int) error {
	for r.finished < taskCount {
		r.scheduleReady(ctx)
		done, err := r.handleIdleState()
		if err != nil {
			return err
		}
		if done {
			return nil
		}
		r.handleOutcome(<-r.resultCh)
	}
	return nil
}

func (r *executorRun) scheduleReady(ctx context.Context) {
	for r.firstErr == nil && r.running < r.maxParallelism && len(r.ready) > 0 {
		state := r.ready[0]
		r.ready = r.ready[1:]
		r.startTask(ctx, state)
	}
}

func (r *executorRun) handleIdleState() (bool, error) {
	if r.running != 0 {
		return false, nil
	}
	if r.firstErr != nil {
		r.skipAllPending(r.firstErr)
		return true, nil
	}
	if len(r.ready) == 0 {
		return false, fmt.Errorf("modelruntime: unresolved task graph; dependency graph is cyclic or blocked")
	}
	return false, nil
}

func (r *executorRun) skipAllPending(cause error) {
	for name := range r.states {
		r.markSkipped(name, cause)
	}
}

func (r *executorRun) markSkipped(name string, cause error) {
	state := r.states[name]
	if state == nil || state.result.Status != TaskPending {
		return
	}

	state.result.Status = TaskSkipped
	state.result.Error = cause
	r.finished++
	r.emit(Event{
		Task:       name,
		Status:     TaskSkipped,
		BestEffort: state.task.BestEffort,
		Error:      cause,
	})

	for _, dependent := range state.dependents {
		r.markSkipped(dependent, fmt.Errorf("dependency %s did not complete: %w", name, cause))
	}
}

func (r *executorRun) startTask(ctx context.Context, state *taskState) {
	state.result.Status = TaskRunning
	r.running++
	r.emit(Event{
		Task:       state.task.Name,
		Status:     TaskRunning,
		BestEffort: state.task.BestEffort,
	})

	go func(task Task) {
		r.resultCh <- taskOutcome{
			name: task.Name,
			err:  task.Run(ctx),
		}
	}(state.task)
}

func (r *executorRun) handleOutcome(outcome taskOutcome) {
	state := r.states[outcome.name]
	r.running--
	if outcome.err != nil {
		r.failTask(state, outcome.err)
		return
	}
	r.succeedTask(state)
}

func (r *executorRun) failTask(state *taskState, err error) {
	state.result.Status = TaskFailed
	state.result.Error = err
	r.finished++
	r.emit(Event{
		Task:       state.task.Name,
		Status:     TaskFailed,
		BestEffort: state.task.BestEffort,
		Error:      err,
	})

	for _, dependent := range state.dependents {
		r.markSkipped(dependent, fmt.Errorf("dependency %s failed: %w", state.task.Name, err))
	}

	if state.task.BestEffort || r.firstErr != nil {
		return
	}
	r.firstErr = fmt.Errorf("task %s failed: %w", state.task.Name, err)
	r.cancel()
}

func (r *executorRun) succeedTask(state *taskState) {
	state.result.Status = TaskSucceeded
	r.finished++
	r.emit(Event{
		Task:       state.task.Name,
		Status:     TaskSucceeded,
		BestEffort: state.task.BestEffort,
	})

	if r.firstErr != nil {
		return
	}
	r.enqueueDependents(state)
}

func (r *executorRun) enqueueDependents(state *taskState) {
	for _, dependentName := range state.dependents {
		dependent := r.states[dependentName]
		if dependent == nil || dependent.result.Status != TaskPending {
			continue
		}
		dependent.remainingDeps--
		if dependent.remainingDeps == 0 {
			r.ready = append(r.ready, dependent)
		}
	}
}

func buildTaskStates(tasks []Task) (map[string]*taskState, error) {
	states := make(map[string]*taskState, len(tasks))
	for _, task := range tasks {
		if task.Name == "" {
			return nil, fmt.Errorf("modelruntime: task name cannot be empty")
		}
		if task.Run == nil {
			return nil, fmt.Errorf("modelruntime: task %s has nil Run function", task.Name)
		}
		if _, exists := states[task.Name]; exists {
			return nil, fmt.Errorf("modelruntime: duplicate task name %s", task.Name)
		}
		states[task.Name] = &taskState{
			task: task,
			result: TaskResult{
				Name:       task.Name,
				Status:     TaskPending,
				BestEffort: task.BestEffort,
			},
		}
	}

	for _, task := range tasks {
		state := states[task.Name]
		for _, dependency := range task.Dependencies {
			dependencyState := states[dependency]
			if dependencyState == nil {
				return nil, fmt.Errorf("modelruntime: task %s depends on unknown task %s", task.Name, dependency)
			}
			state.remainingDeps++
			dependencyState.dependents = append(dependencyState.dependents, task.Name)
		}
	}

	return states, nil
}

func initialReadyTasks(states map[string]*taskState) []*taskState {
	ready := make([]*taskState, 0, len(states))
	for _, state := range states {
		if state.remainingDeps == 0 {
			ready = append(ready, state)
		}
	}
	return ready
}
