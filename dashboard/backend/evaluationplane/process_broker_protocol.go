package evaluationplane

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
)

func (broker *workerHTTPBroker) serve(ctx context.Context, reader io.Reader, writer io.Writer) (resultErr error) {
	var workers sync.WaitGroup
	sessionContext, cancel := context.WithCancel(ctx)
	broker.bindSessionCancellation(cancel)
	defer func() {
		if resultErr != nil {
			cancel()
		}
		workers.Wait()
		if resultErr == nil {
			resultErr = broker.sessionFailure()
		}
		cancel()
		broker.releaseSessionCancellation()
	}()
	buffered := bufio.NewReaderSize(reader, 64*1024)
	var lastID uint64
	var transferred int64
	for count := 0; ; count++ {
		if err := broker.sessionFailure(); err != nil {
			return err
		}
		frame, err := readWorkerBrokerFrame(buffered)
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		transferred += int64(len(frame))
		if count >= broker.requestMax || transferred > maxWorkerBrokerRequestBytes {
			return fmt.Errorf("evaluation worker HTTP broker request limit exceeded")
		}
		request, err := decodeWorkerBrokerRequest(frame, lastID, broker.operations)
		if err != nil {
			return err
		}
		lastID = request.ID
		if err := broker.admitRequest(request); err != nil {
			broker.abortSession(err)
			return err
		}
		select {
		case broker.semaphore <- struct{}{}:
		case <-sessionContext.Done():
			if err := broker.sessionFailure(); err != nil {
				return err
			}
			return sessionContext.Err()
		}
		workers.Add(1)
		go func() {
			defer workers.Done()
			defer func() { <-broker.semaphore }()
			response := broker.executeAdmitted(sessionContext, request)
			responseFrame, frameErr := encodeWorkerBrokerFrame(response)
			if frameErr == nil {
				frameErr = broker.reserveResponseBytes(int64(len(responseFrame)))
			}
			broker.writeMu.Lock()
			if frameErr == nil {
				frameErr = writeWorkerBrokerFrame(writer, responseFrame)
			}
			if frameErr != nil {
				// Closing the response pipe is the fail-closed signal to the worker.
				if closer, ok := writer.(io.Closer); ok {
					_ = closer.Close()
				}
				// Closing the request side interrupts a worker that keeps its pipe
				// open after the response channel has been revoked.
				if closer, ok := reader.(io.Closer); ok {
					_ = closer.Close()
				}
			}
			broker.writeMu.Unlock()
			if frameErr != nil {
				broker.abortSession(frameErr)
			}
		}()
	}
}

func decodeWorkerBrokerRequest(
	frame []byte,
	lastID uint64,
	operations map[string]workerBrokerOperation,
) (workerBrokerRequest, error) {
	if err := rejectDuplicateJSONKeys(frame); err != nil {
		return workerBrokerRequest{}, fmt.Errorf("validate evaluation worker HTTP broker request JSON: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(frame))
	decoder.DisallowUnknownFields()
	var request workerBrokerRequest
	if err := decoder.Decode(&request); err != nil {
		return workerBrokerRequest{}, fmt.Errorf("decode evaluation worker HTTP broker request: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return workerBrokerRequest{}, err
	}
	if request.ID == 0 || request.ID != lastID+1 ||
		request.TimeoutMS < 1 || request.TimeoutMS > maxWorkerBrokerTimeoutMS {
		return workerBrokerRequest{}, fmt.Errorf("evaluation worker HTTP broker request envelope is invalid")
	}
	operation, ok := operations[request.Operation]
	if !ok || operation.method == "" || operation.url == "" {
		return workerBrokerRequest{}, fmt.Errorf("evaluation worker requested an unapproved HTTP operation")
	}
	payload := bytes.TrimSpace(request.Payload)
	if operation.maxTimeoutMS > 0 && request.TimeoutMS > operation.maxTimeoutMS {
		return workerBrokerRequest{}, fmt.Errorf("evaluation worker HTTP broker timeout exceeds the endpoint contract")
	}
	if operation.method == http.MethodGet {
		if !bytes.Equal(payload, []byte("null")) {
			return workerBrokerRequest{}, fmt.Errorf("evaluation worker HTTP GET payload is invalid")
		}
		switch request.Operation {
		case workerBrokerListModels:
			if request.TrackID != "" || request.CaseID != "" || request.AttemptID != "" {
				return workerBrokerRequest{}, fmt.Errorf("model discovery cannot bind evidence")
			}
		case workerBrokerAgentTaskLedger:
			if validateMethodLedgerRequestIdentity(request) != nil {
				return workerBrokerRequest{}, fmt.Errorf("agent-task ledger requests must bind agentic evidence")
			}
		case workerBrokerFaultRecoveryLedger:
			if validateMethodLedgerRequestIdentity(request) != nil {
				return workerBrokerRequest{}, fmt.Errorf("fault-recovery ledger requests must bind agentic evidence")
			}
		case workerBrokerHardPolicyLedger:
			if validateMethodLedgerRequestIdentity(request) != nil {
				return workerBrokerRequest{}, fmt.Errorf("hard-policy ledger requests must bind safety evidence")
			}
		case workerBrokerProductionExperimentLedger:
			if validateMethodLedgerRequestIdentity(request) != nil {
				return workerBrokerRequest{}, fmt.Errorf("production experiment ledger requests must bind preference evidence")
			}
		}
	} else {
		if !validBrokerEvidenceAttempt(request) || len(payload) < 2 || payload[0] != '{' ||
			payload[len(payload)-1] != '}' || !json.Valid(payload) {
			return workerBrokerRequest{}, fmt.Errorf("evaluation worker HTTP POST evidence envelope is invalid")
		}
		if request.Operation == workerBrokerRouterEvaluate && request.TrackID != "routing" {
			return workerBrokerRequest{}, fmt.Errorf("router evaluation requests must bind routing evidence")
		}
		if request.Operation == workerBrokerRoutedChatCompletion && request.TrackID != "joint" &&
			request.TrackID != "multimodal" && request.TrackID != "capacity" {
			return workerBrokerRequest{}, fmt.Errorf("routed chat requests must bind joint, multimodal, or capacity evidence")
		}
		if request.Operation == workerBrokerArmChatCompletion && request.TrackID != "model_pool" {
			return workerBrokerRequest{}, fmt.Errorf("arm chat requests must bind model_pool evidence")
		}
	}
	return request, nil
}

func validBrokerEvidenceAttempt(request workerBrokerRequest) bool {
	return containsTrack(allTrackIDs, request.TrackID) && evidenceIDPattern.MatchString(request.CaseID) &&
		evidenceIDPattern.MatchString(request.AttemptID)
}

func readWorkerBrokerFrame(reader io.Reader) ([]byte, error) {
	var header [4]byte
	if _, err := io.ReadFull(reader, header[:]); err != nil {
		return nil, err
	}
	size := binary.BigEndian.Uint32(header[:])
	if size < 2 || size > maxWorkerBrokerFrameBytes {
		return nil, fmt.Errorf("evaluation worker HTTP broker frame is outside its bound")
	}
	frame := make([]byte, int(size))
	if _, err := io.ReadFull(reader, frame); err != nil {
		return nil, err
	}
	return frame, nil
}

func encodeWorkerBrokerFrame(value workerBrokerResponse) ([]byte, error) {
	data, err := json.Marshal(value)
	if err != nil || len(data) < 2 || len(data) > maxWorkerBrokerFrameBytes {
		return nil, fmt.Errorf("encode evaluation worker HTTP broker response")
	}
	frame := make([]byte, 4+len(data))
	// The protocol limit above is 4 MiB, well within uint32's range.
	//nolint:gosec // Conversion is bounded by maxWorkerBrokerFrameBytes.
	binary.BigEndian.PutUint32(frame[:4], uint32(len(data)))
	copy(frame[4:], data)
	return frame, nil
}

func writeWorkerBrokerFrame(writer io.Writer, frame []byte) error {
	for len(frame) > 0 {
		written, err := writer.Write(frame)
		if err != nil {
			return err
		}
		if written <= 0 || written > len(frame) {
			return io.ErrShortWrite
		}
		frame = frame[written:]
	}
	return nil
}

type workerBrokerSession struct {
	requestReader  *os.File
	requestWriter  *os.File
	responseReader *os.File
	responseWriter *os.File
	broker         *workerHTTPBroker
	done           chan error
}

func newWorkerBrokerSession(broker *workerHTTPBroker) (*workerBrokerSession, error) {
	requestReader, requestWriter, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("create evaluation worker broker request pipe: %w", err)
	}
	responseReader, responseWriter, err := os.Pipe()
	if err != nil {
		_ = requestReader.Close()
		_ = requestWriter.Close()
		return nil, fmt.Errorf("create evaluation worker broker response pipe: %w", err)
	}
	return &workerBrokerSession{
		requestReader: requestReader, requestWriter: requestWriter,
		responseReader: responseReader, responseWriter: responseWriter,
		broker: broker, done: make(chan error, 1),
	}, nil
}

func (session *workerBrokerSession) childFiles() []*os.File {
	return []*os.File{session.requestWriter, session.responseReader}
}

func (session *workerBrokerSession) start(ctx context.Context) {
	_ = session.requestWriter.Close()
	_ = session.responseReader.Close()
	go func() {
		err := session.broker.serve(ctx, session.requestReader, session.responseWriter)
		_ = session.responseWriter.Close()
		session.done <- err
	}()
}

func (session *workerBrokerSession) wait() error { return <-session.done }

func (session *workerBrokerSession) close() {
	_ = session.requestReader.Close()
	_ = session.requestWriter.Close()
	_ = session.responseReader.Close()
	_ = session.responseWriter.Close()
}
