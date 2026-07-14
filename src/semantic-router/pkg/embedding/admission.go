package embedding

import (
	"context"
	"errors"
	"sync"
)

const DefaultProcessAdmissionCapacity = 2

// ErrOverloaded reports that process-local native embedding work is already at
// capacity. Callers must fail fast rather than queue request-owned image
// decodes or native inference allocations without bound.
var ErrOverloaded = errors.New("embedding inference overloaded")

// DefaultProcessAdmission is shared by every request surface in this process.
// Keeping one budget prevents the API server and ExtProc paths from each
// admitting their own full allowance concurrently.
var DefaultProcessAdmission = NewProcessAdmission(DefaultProcessAdmissionCapacity)

// ProcessAdmission is the process-level seam protecting expensive image
// decode and native embedding work. It never queues: callers either acquire a
// slot immediately or receive ErrOverloaded.
type ProcessAdmission struct {
	slots chan struct{}
}

func NewProcessAdmission(capacity int) *ProcessAdmission {
	if capacity <= 0 {
		panic("embedding admission capacity must be positive")
	}
	return &ProcessAdmission{slots: make(chan struct{}, capacity)}
}

func (a *ProcessAdmission) TryAcquire(ctx context.Context) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case a.slots <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-a.slots
			return nil, err
		}
		var once sync.Once
		return func() {
			once.Do(func() { <-a.slots })
		}, nil
	default:
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return nil, ErrOverloaded
	}
}
