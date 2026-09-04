package extproc

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/emptypb"
)

type shutdownTestService interface{}

func TestServerShutdownDrainsGenerationAfterForcedGRPCStopWithinDeadline(t *testing.T) {
	resourcesClosed := make(chan struct{})
	resources := newResourceScope()
	resources.add(func() error {
		close(resourcesClosed)
		return nil
	})
	router := (&routerComponents{resources: resources}).buildRouter()
	service := NewRouterService(router)

	rpcStarted := make(chan struct{})
	grpcServer := grpc.NewServer()
	grpcServer.RegisterService(&grpc.ServiceDesc{
		ServiceName: "extproc.shutdown.test",
		HandlerType: (*shutdownTestService)(nil),
		Streams: []grpc.StreamDesc{{
			StreamName:    "Hold",
			ClientStreams: true,
			ServerStreams: true,
			Handler: func(_ any, stream grpc.ServerStream) error {
				release, acquired := service.current.Load().acquire()
				if !acquired {
					return errors.New("failed to acquire generation")
				}
				defer release()
				close(rpcStarted)
				<-stream.Context().Done()
				return stream.Context().Err()
			},
		}},
	}, struct{}{})

	listener := bufconn.Listen(1024 * 1024)
	go func() { _ = grpcServer.Serve(listener) }()
	t.Cleanup(func() { _ = listener.Close() })
	connection, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	stream, err := connection.NewStream(context.Background(), &grpc.StreamDesc{ClientStreams: true, ServerStreams: true}, "/extproc.shutdown.test/Hold")
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.SendMsg(&emptypb.Empty{}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-rpcStarted:
	case <-time.After(time.Second):
		t.Fatal("test RPC did not start")
	}

	server := &Server{server: grpcServer, service: service}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	started := time.Now()
	err = server.Shutdown(shutdownCtx)
	if elapsed := time.Since(started); elapsed > 250*time.Millisecond {
		t.Fatalf("Shutdown() took %v, want it bounded by the caller deadline", elapsed)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Shutdown() error = %v, want deadline exceeded after forced stop", err)
	}
	select {
	case <-resourcesClosed:
	default:
		t.Fatal("generation resources were not closed before forced shutdown returned")
	}
}
