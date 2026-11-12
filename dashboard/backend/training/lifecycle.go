package training

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

func save(tx *workflowstore.TrainingTx, owner string, g *c.RunGraph, taskID, diagnostic string) error {
	g.Run.UpdatedAt = time.Now().UTC()
	r, err := record("runs", owner, g.Run.ID, g)
	if err != nil {
		return err
	}
	if err := tx.SaveRun(r); err != nil {
		return err
	}
	status := g.Run.Status
	for _, task := range g.Tasks {
		if task.ID == taskID {
			status = task.Status
		}
	}
	return tx.Event(c.Event{RunID: g.Run.ID, TaskID: taskID, Status: status, Diagnostic: diagnostic, RecordedAt: g.Run.UpdatedAt})
}

func (s *Service) mutate(ctx context.Context, owner, id string, fn func(*workflowstore.TrainingTx, *c.RunGraph) error) (c.RunGraph, error) {
	var g c.RunGraph
	err := s.update(ctx, func(tx *workflowstore.TrainingTx) error {
		var err error
		g, err = read[c.RunGraph](tx, owner, "runs", id)
		if err != nil {
			return err
		}
		return fn(tx, &g)
	})
	return g, err
}

func (s *Service) Cancel(ctx context.Context, owner, id string) (c.RunGraph, error) {
	return s.mutate(ctx, owner, id, func(tx *workflowstore.TrainingTx, g *c.RunGraph) error {
		if g.Run.Status == c.Cancelled || g.Run.Status == c.Cancelling {
			return nil
		}
		if err := c.ValidateTransition(g.Run.Status, c.Cancelling); err != nil {
			return fmt.Errorf("%w: %w", ErrConflict, err)
		}
		g.Run.Status = c.Cancelling
		if err := save(tx, owner, g, "", "cancellation requested"); err != nil {
			return err
		}
		for i := range g.Tasks {
			task := &g.Tasks[i]
			if task.Status == c.Pending {
				task.Status = c.Cancelled
				if err := save(tx, owner, g, task.ID, "cancelled before execution"); err != nil {
					return err
				}
			}
		}
		return settle(tx, owner, g)
	})
}

func (s *Service) Retry(ctx context.Context, owner, id string) (c.RunGraph, error) {
	return s.mutate(ctx, owner, id, func(tx *workflowstore.TrainingTx, g *c.RunGraph) error {
		if err := c.ValidateTransition(g.Run.Status, c.Pending); err != nil {
			return fmt.Errorf("%w: %w", ErrConflict, err)
		}
		g.Run.Status = c.Pending
		needsExecution := false
		for i := range g.Tasks {
			if g.Tasks[i].Status != c.Succeeded {
				g.Tasks[i].Status = c.Pending
				needsExecution = true
			}
		}
		if err := save(tx, owner, g, "", "retry requested; successful tasks and outputs retained"); err != nil {
			return err
		}
		if needsExecution {
			return nil
		}
		// Every task may have succeeded while cancellation was in progress. Keep
		// the normal run transitions, but settle atomically without new attempts.
		g.Run.Status = c.Running
		if err := save(tx, owner, g, "", "run resumed with retained successful tasks"); err != nil {
			return err
		}
		return settle(tx, owner, g)
	})
}

// StartAttempt is the internal executor boundary, not a public HTTP operation.
// It persists a fresh identity before dispatch, or returns the active request on
// repeated calls. Dependency readiness is authoritative in the Go control plane.
func (s *Service) StartAttempt(ctx context.Context, owner, runID, taskID string) (c.WorkerRequest, error) {
	var request c.WorkerRequest
	_, err := s.mutate(ctx, owner, runID, func(tx *workflowstore.TrainingTx, g *c.RunGraph) error {
		if g.Run.Status != c.Pending && g.Run.Status != c.Running {
			return fmt.Errorf("%w: run is not executable", ErrConflict)
		}
		for i := range g.Tasks {
			task := &g.Tasks[i]
			if task.ID != taskID {
				continue
			}
			if task.Status == c.Running {
				var err error
				request, err = workerRequest(tx, owner, g, task)
				return err
			}
			if task.Status != c.Pending {
				return fmt.Errorf("%w: task is not pending", ErrConflict)
			}
			for _, key := range task.Spec.DependsOn {
				for _, dep := range g.Tasks {
					if dep.Spec.Key == key && dep.Status != c.Succeeded {
						return fmt.Errorf("%w: dependency %s has not succeeded", ErrConflict, key)
					}
				}
			}
			if g.Run.Status == c.Pending {
				g.Run.Status = c.Running
				if err := save(tx, owner, g, "", "run started"); err != nil {
					return err
				}
			}
			now := time.Now().UTC()
			task.Attempts = append(task.Attempts, c.Attempt{Metadata: metadata("attempt"), Number: len(task.Attempts) + 1, Status: c.Running, StartedAt: &now})
			task.Status = c.Running
			if err := save(tx, owner, g, task.ID, "attempt persisted before dispatch"); err != nil {
				return err
			}
			var err error
			request, err = workerRequest(tx, owner, g, task)
			return err
		}
		return ErrNotFound
	})
	return request, err
}

func workerRequest(tx *workflowstore.TrainingTx, owner string, g *c.RunGraph, task *c.RunTask) (c.WorkerRequest, error) {
	snapshot, err := read[c.DataSnapshot](tx, owner, "data-snapshots", g.Run.Spec.SnapshotID)
	if err != nil {
		return c.WorkerRequest{}, err
	}
	inputs, err := dependencyInputs(tx, owner, g, task)
	if err != nil {
		return c.WorkerRequest{}, err
	}
	return c.WorkerRequest{SchemaVersion: c.Version, RunID: g.Run.ID, TaskID: task.ID, AttemptID: task.Attempts[len(task.Attempts)-1].ID, BaseModel: g.Run.Spec.BaseModel, Executor: task.Spec.Executor, Trainer: g.Run.Spec.Trainer, Parameters: g.Run.Spec.Parameters, Snapshot: snapshot, Inputs: inputs}, nil
}

func activeTask(g *c.RunGraph, attemptID string) (*c.RunTask, error) {
	for i := range g.Tasks {
		task := &g.Tasks[i]
		if task.Status == c.Running && task.Attempts[len(task.Attempts)-1].ID == attemptID {
			return task, nil
		}
	}
	return nil, fmt.Errorf("%w: attempt is not active", ErrConflict)
}

func (s *Service) Acknowledge(ctx context.Context, owner, runID, attemptID string, ack c.WorkerSubmission) error {
	if ack.SchemaVersion != c.Version {
		return invalid(errors.New("unsupported worker schema_version"))
	}
	if err := c.ValidateHandle(ack.WorkerHandle); err != nil {
		return invalid(err)
	}
	_, err := s.mutate(ctx, owner, runID, func(tx *workflowstore.TrainingTx, g *c.RunGraph) error {
		task, err := activeTask(g, attemptID)
		if err != nil {
			return err
		}
		a := &task.Attempts[len(task.Attempts)-1]
		if a.WorkerHandle == ack.WorkerHandle {
			return nil
		}
		if a.WorkerHandle != "" {
			return fmt.Errorf("%w: attempt already acknowledged by a different worker", ErrConflict)
		}
		a.WorkerHandle = ack.WorkerHandle
		return save(tx, owner, g, task.ID, "worker acknowledged attempt")
	})
	return err
}

// Recovery returns active requests with their original attempt identities. It
// never creates attempts or marks jobs failed merely because management restarted.
// Worker redispatch, polling, leases and scheduling belong to the coordinator.
type Recovery struct {
	Owner        string
	Request      c.WorkerRequest
	WorkerHandle string
	Cancel       bool
}

func (s *Service) Recover(ctx context.Context) ([]Recovery, error) {
	records, err := s.store.ListTrainingAll(ctx, "runs")
	if err != nil {
		return nil, err
	}
	out := []Recovery{}
	for _, r := range records {
		err := s.update(ctx, func(tx *workflowstore.TrainingTx) error {
			g, err := read[c.RunGraph](tx, r.Owner, "runs", r.ID)
			if err != nil {
				return err
			}
			for i := range g.Tasks {
				task := &g.Tasks[i]
				if task.Status != c.Running {
					continue
				}
				request, err := workerRequest(tx, r.Owner, &g, task)
				if err != nil {
					return err
				}
				out = append(out, Recovery{Owner: r.Owner, Request: request, WorkerHandle: task.Attempts[len(task.Attempts)-1].WorkerHandle, Cancel: g.Run.Status == c.Cancelling})
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func settle(tx *workflowstore.TrainingTx, owner string, g *c.RunGraph) error {
	for changed := true; changed; {
		changed = false
		for i := range g.Tasks {
			task := &g.Tasks[i]
			if task.Status != c.Pending {
				continue
			}
			blocked := false
			for _, key := range task.Spec.DependsOn {
				for _, dep := range g.Tasks {
					blocked = blocked || (dep.Spec.Key == key && c.Terminal(dep.Status) && dep.Status != c.Succeeded)
				}
			}
			if blocked {
				task.Status = c.Skipped
				changed = true
				if err := save(tx, owner, g, task.ID, "dependency did not succeed"); err != nil {
					return err
				}
			}
		}
	}
	failed := false
	for _, task := range g.Tasks {
		if !c.Terminal(task.Status) {
			return nil
		}
		failed = failed || task.Status != c.Succeeded
	}
	status := c.Succeeded
	if g.Run.Status == c.Cancelling {
		status = c.Cancelled
	} else if failed {
		status = c.Failed
	}
	g.Run.Status = status
	return save(tx, owner, g, "", "run settled")
}
