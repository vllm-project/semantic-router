package mlpipeline

import (
	"errors"
	"log"
	"sync"
)

const (
	maxActivePipelineJobs  = 3
	maxActiveBenchmarkJobs = 2
	maxActiveTrainingJobs  = 1
)

// ErrJobCapacityExceeded is returned synchronously when admitting another
// asynchronous ML job would exceed the runner's fixed resource budget.
var ErrJobCapacityExceeded = errors.New("ML pipeline job capacity exceeded")

// ErrPipelineInputRejected identifies a caller-controlled file or work budget
// violation without exposing internal paths or parsing details to HTTP clients.
var ErrPipelineInputRejected = errors.New("ML pipeline input rejected")

type pipelineJobClass uint8

const (
	pipelineBenchmarkJob pipelineJobClass = iota + 1
	pipelineTrainingJob
)

func (r *Runner) acquireJobSlot(class pipelineJobClass) (func(), error) {
	r.admissionMu.Lock()
	defer r.admissionMu.Unlock()
	if r.activeJobs >= maxActivePipelineJobs ||
		(class == pipelineBenchmarkJob && r.activeBenchmarkJobs >= maxActiveBenchmarkJobs) ||
		(class == pipelineTrainingJob && r.activeTrainingJobs >= maxActiveTrainingJobs) {
		return nil, ErrJobCapacityExceeded
	}
	r.activeJobs++
	switch class {
	case pipelineBenchmarkJob:
		r.activeBenchmarkJobs++
	case pipelineTrainingJob:
		r.activeTrainingJobs++
	default:
		r.activeJobs--
		return nil, errors.New("unknown ML pipeline job class")
	}

	var once sync.Once
	release := func() {
		once.Do(func() {
			r.admissionMu.Lock()
			defer r.admissionMu.Unlock()
			r.activeJobs--
			switch class {
			case pipelineBenchmarkJob:
				r.activeBenchmarkJobs--
			case pipelineTrainingJob:
				r.activeTrainingJobs--
			}
		})
	}
	return release, nil
}

func (r *Runner) startAsyncJob(jobID string, release func(), run func()) {
	go func() {
		defer release()
		defer func() {
			if recover() != nil {
				log.Printf("[ml-pipeline] recovered panic in job %s", jobID)
				r.failJob(jobID, "ML pipeline job failed unexpectedly")
				r.sendProgress(jobID, 100, "Failed", "ML pipeline job failed unexpectedly")
			}
		}()
		run()
	}()
}
