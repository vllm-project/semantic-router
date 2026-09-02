package evaluationplane

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type Options struct {
	DataDir                    string
	PythonPath                 string
	RouterAPIURL               string
	EnvoyURL                   string
	ConfigPath                 string
	DeploymentsDir             string
	CodeRevision               string
	RouterAPIKeyEnv            string
	EnvoyAPIKeyEnv             string
	AgentTaskLedger            *ServiceEndpoint
	FaultRecoveryLedger        *ServiceEndpoint
	HardPolicyLedger           *ServiceEndpoint
	ProductionExperimentLedger *ServiceEndpoint
	MaxConcurrent              int
	WorkerTimeout              time.Duration
	Process                    Process
	CredentialProvider         CredentialProvider
	// DiagnosticSink is server-owned and must never be exposed through the
	// Evaluation API. It receives detailed execution failures while durable run
	// status retains only the generic public error contract.
	DiagnosticSink  io.Writer
	LifecycleLimits LifecycleLimits
}

const defaultWorkerTimeout = 6 * time.Hour

const maxWorkerEventsPerRun = 4096

const (
	maxSubscribersPerRun       = 16
	maxSubscribersPerPrincipal = 32
	maxSubscribersGlobal       = 256
	maxConcurrentEvidenceReads = 8
)

type Service struct {
	store            *Store
	process          Process
	codeRevision     string
	registrySource   runtimeRegistrySource
	workerTimeout    time.Duration
	operationMu      sync.RWMutex
	leaseMu          sync.Mutex
	mu               sync.Mutex
	active           map[string]context.CancelFunc
	workerEvents     map[string]int
	workers          sync.WaitGroup
	prelaunches      sync.WaitGroup
	prelaunchCount   int
	prelaunchContext context.Context
	prelaunchCancel  context.CancelFunc
	closeOnce        sync.Once
	shutdown         chan struct{}
	lifecycleErr     error
	diagnosticLogger *log.Logger
	activity         *evaluationRootCoordinator
	ownership        *evaluationStoreOwnership
	closed           bool
}

// NewService opens one lease on the canonical evaluation root and performs
// process-wide recovery only for the first lease holder.
func NewService(options Options) (*Service, error) {
	return newService(options, NewRegistry)
}

func newService(options Options, constructor registryConstructor) (*Service, error) {
	if options.DataDir == "" {
		return nil, fmt.Errorf("%w: evaluation data directory is required", ErrInvalid)
	}
	root, err := filepath.Abs(options.DataDir)
	if err != nil {
		return nil, fmt.Errorf("resolve evaluation data directory: %w", err)
	}
	if directoryErr := ensureDurablePrivateDirectoryTree(root); directoryErr != nil {
		return nil, fmt.Errorf("create evaluation store root: %w", directoryErr)
	}
	ownership, err := acquireEvaluationStoreOwnership(root)
	if err != nil {
		return nil, err
	}
	ownershipTransferred := false
	defer func() {
		if !ownershipTransferred {
			_ = ownership.release()
		}
	}()
	var service *Service
	err = ownership.initialize(func(startupAuthority bool) error {
		store, openErr := newStoreWithRootCoordinator(
			root, options.LifecycleLimits, ownership.coordinator, startupAuthority,
		)
		if openErr != nil {
			return openErr
		}
		diagnosticSink := options.DiagnosticSink
		if diagnosticSink == nil {
			diagnosticSink = os.Stderr
		}
		options.DiagnosticSink = diagnosticSink
		setup, setupErr := prepareServiceRuntime(&options, store, constructor)
		if setupErr != nil {
			return setupErr
		}
		if capacityErr := store.lifecycle.validateWorkerCapacity(options.MaxConcurrent); capacityErr != nil {
			return capacityErr
		}
		prelaunchContext, prelaunchCancel := context.WithCancel(context.Background())
		service = &Service{
			store:            store,
			process:          setup.process,
			codeRevision:     setup.codeRevision,
			registrySource:   setup.registrySource,
			workerTimeout:    options.WorkerTimeout,
			active:           make(map[string]context.CancelFunc),
			workerEvents:     make(map[string]int),
			prelaunchContext: prelaunchContext,
			prelaunchCancel:  prelaunchCancel,
			shutdown:         make(chan struct{}),
			diagnosticLogger: log.New(diagnosticSink, "", log.LstdFlags|log.LUTC),
			activity:         store.lifecycle,
			ownership:        ownership,
		}
		if startupAuthority {
			if recoveryErr := service.RecoverInterruptedRuns(); recoveryErr != nil {
				return recoveryErr
			}
		}
		return store.lifecycle.commitWorkerCapacity(options.MaxConcurrent)
	})
	if err != nil {
		if service != nil {
			service.ownership = nil
			_ = service.Close()
		}
		return nil, err
	}
	ownershipTransferred = true
	return service, nil
}

func configureServiceProcess(options *Options, store *Store) (Process, error) {
	if options.MaxConcurrent <= 0 {
		options.MaxConcurrent = 2
	}
	if options.WorkerTimeout < 0 {
		return nil, fmt.Errorf("evaluation worker timeout cannot be negative")
	}
	if options.WorkerTimeout == 0 {
		options.WorkerTimeout = defaultWorkerTimeout
	}
	process := options.Process
	if process == nil {
		commandProcess := NewCommandProcess(options.PythonPath)
		commandProcess.routerAPIKeyEnv = strings.TrimSpace(options.RouterAPIKeyEnv)
		commandProcess.envoyAPIKeyEnv = strings.TrimSpace(options.EnvoyAPIKeyEnv)
		commandProcess.cpuSeconds = workerCPULimit(options.WorkerTimeout)
		commandProcess.diagnosticSink = options.DiagnosticSink
		commandProcess.publishEvidence = store.importWorkerEvidence
		process = commandProcess
	}
	return process, nil
}

// Close prevents new workers from starting, cancels every active worker, and
// waits until each worker has released its process and concurrency slot.
func (s *Service) Close() error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		s.closed = true
		close(s.shutdown)
		s.prelaunchCancel()
		cancellations := make([]context.CancelFunc, 0, len(s.active))
		for _, cancel := range s.active {
			cancellations = append(cancellations, cancel)
		}
		s.mu.Unlock()

		for _, cancel := range cancellations {
			cancel()
		}
		// Marking the Service closed and cancelling prelaunch work must happen
		// before waiting for the operation writer. Long-running pair setup holds
		// a read lease and relies on this cancellation to return.
		s.operationMu.Lock()
		defer s.operationMu.Unlock()
		s.prelaunches.Wait()
		s.workers.Wait()

		s.mu.Lock()
		s.activity.eventSubscribers.closeOwner(s)
		s.mu.Unlock()
	})
	s.leaseMu.Lock()
	if s.ownership != nil {
		if err := s.ownership.release(); err != nil {
			s.mu.Lock()
			s.recordLifecycleErrorLocked(fmt.Errorf("release evaluation store ownership: %w", err))
			s.mu.Unlock()
		} else {
			s.ownership = nil
		}
	}
	s.leaseMu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lifecycleErr
}
