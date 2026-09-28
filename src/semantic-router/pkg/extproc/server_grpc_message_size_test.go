package extproc

import (
	"context"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Envoy sends a BUFFERED request body to ExtProc as one message, so the
// server's default message limit bounds the largest request the Router sees.
func TestDefaultExtProcLimitAdmitsBufferedBodyOverFourMiB(t *testing.T) {
	router, err := CreateTestRouter(CreateTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{service: NewRouterService(router)}
	limit := server.configuredGRPCMaxMessageSize()
	grpcServer := grpc.NewServer(grpc.MaxRecvMsgSize(limit), grpc.MaxSendMsgSize(limit))
	ext_proc.RegisterExternalProcessorServer(grpcServer, server.service)
	listener := bufconn.Listen(1 << 20)
	go func() { _ = grpcServer.Serve(listener) }()
	t.Cleanup(grpcServer.Stop)

	connection, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(256<<20)),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	stream, err := ext_proc.NewExternalProcessorClient(connection).Process(ctx)
	if err != nil {
		t.Fatal(err)
	}

	headers := &ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/chat/completions"},
			{Key: "content-type", Value: "application/json"},
		}}},
	}}
	if err = stream.Send(headers); err != nil {
		t.Fatal(err)
	}
	if _, err = stream.Recv(); err != nil {
		t.Fatalf("request headers: %v", err)
	}
	body := fmt.Sprintf(`{"model":"model-a","messages":[{"role":"user","content":%q}]}`, strings.Repeat("lorem ipsum ", 5<<20/12))
	if err = stream.Send(&ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}}); err != nil {
		t.Fatal(err)
	}
	response, err := stream.Recv()
	if err != nil {
		t.Fatalf("%d-byte request body did not reach the Router: %v", len(body), err)
	}
	if response.GetRequestBody() == nil {
		t.Fatalf("request body response = %T, want the routed request body", response.GetResponse())
	}
}

func TestDefaultExtProcLimitCoversCodecBodyLimit(t *testing.T) {
	server := &Server{service: NewRouterService(&OpenAIRouter{Config: &config.RouterConfig{}})}
	if got, codec := server.configuredGRPCMaxMessageSize(), llmprotocol.DefaultPolicy().Limits.BodyBytes; got <= codec {
		t.Fatalf("default ExtProc message limit = %d bytes, want room above the %d-byte codec body limit", got, codec)
	}
}
