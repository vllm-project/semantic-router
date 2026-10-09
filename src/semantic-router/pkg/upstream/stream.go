package upstream

import (
	"errors"
	"io"
	"sync"
	"time"
)

const (
	streamOutcomeComplete = "complete"
	// streamOutcomeClosed labels a body the caller closed before its end.
	streamOutcomeClosed = "closed"
)

// stream is a response body. It reads only when its caller reads, bounds each
// wait for data by the idle timeout, and ends its attempt exactly once: at
// EOF, at the first read error, or on Close.
type stream struct {
	run  *run
	body io.ReadCloser
	// head holds bytes read before Do returned, to see the first byte.
	head  []byte
	idle  *time.Timer
	every time.Duration
	once  sync.Once
	// onEnd ends the call. Do sets it on the body it returns; a body the call
	// retried instead ends only its attempt.
	onEnd func()
}

func newStream(r *run, body io.ReadCloser, head []byte, idle time.Duration) *stream {
	s := &stream{run: r, body: body, head: head, every: idle}
	if enabled(idle) {
		s.idle = time.AfterFunc(idle, func() { r.cancel(&timeoutCause{stage: StageIdle}) })
		s.idle.Stop()
	}
	return s
}

func (s *stream) Read(p []byte) (int, error) {
	if len(s.head) > 0 {
		n := copy(p, s.head)
		s.head = s.head[n:]
		return n, nil
	}
	if s.idle != nil {
		s.idle.Reset(s.every)
	}
	n, err := s.body.Read(p)
	if s.idle != nil {
		s.idle.Stop()
	}
	if err == nil {
		return n, nil
	}
	if errors.Is(err, io.EOF) {
		s.end(streamOutcomeComplete, nil)
		return n, io.EOF
	}
	failure := s.failure(err)
	s.end(string(failure.Kind), failure)
	return n, failure
}

// Close releases the body. Closing before EOF abandons the rest of the
// response and its connection.
func (s *stream) Close() error {
	err := s.body.Close()
	s.end(streamOutcomeClosed, nil)
	return err
}

// end finishes the attempt, with "complete", "closed" or the failure, and
// then the call if this is its served body.
func (s *stream) end(outcome string, failure *Error) {
	s.once.Do(func() {
		if s.idle != nil {
			s.idle.Stop()
		}
		s.run.finish(outcome, failure)
		if s.onEnd != nil {
			s.onEnd()
		}
	})
}

// failure classifies a read error: a timeout or cancellation of the attempt,
// otherwise a connection that broke mid-body.
func (s *stream) failure(err error) *Error {
	if s.run.ctx.Err() != nil {
		return s.run.located(contextError(s.run.ctx))
	}
	return s.run.located(&Error{Kind: KindReset, Err: err})
}

// copyBufferSize bounds one Copy write.
const copyBufferSize = 32 << 10

type flusher interface{ Flush() }

type errorFlusher interface{ FlushError() error }

// Copy writes body to dst as it arrives, flushing after every chunk when dst
// can flush, so streamed tokens reach the client without delay. It reads no
// faster than dst accepts, which keeps memory bounded under a slow client.
func Copy(dst io.Writer, body io.Reader) (int64, error) {
	buf := make([]byte, copyBufferSize)
	var written int64
	for {
		n, readErr := body.Read(buf)
		if n > 0 {
			m, err := dst.Write(buf[:n])
			written += int64(m)
			if err != nil {
				return written, err
			}
			if err := flush(dst); err != nil {
				return written, err
			}
		}
		if errors.Is(readErr, io.EOF) {
			return written, nil
		}
		if readErr != nil {
			return written, readErr
		}
	}
}

func flush(dst io.Writer) error {
	switch f := dst.(type) {
	case errorFlusher:
		return f.FlushError()
	case flusher:
		f.Flush()
	}
	return nil
}
