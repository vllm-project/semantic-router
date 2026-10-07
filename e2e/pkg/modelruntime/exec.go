package modelruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
)

// A managed runtime listens only on a Unix socket in a private directory of
// the Router container. The E2E reaches it the way an operator would debug
// it: by running a small Python HTTP client inside that container, which
// ships the runtime and therefore Python.
const unixHTTPClient = `
import http.client, socket, sys
class Conn(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=120)
        self.unix_path = path
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(120)
        self.sock.connect(self.unix_path)
path, method, target = sys.argv[1:4]
body = sys.stdin.buffer.read() or None
headers = {"Content-Type": "application/json"} if body else {}
conn = Conn(path)
conn.request(method, target, body=body, headers=headers)
response = conn.getresponse()
payload = response.read()
sys.stdout.write("%d\n" % response.status)
sys.stdout.flush()
sys.stdout.buffer.write(payload)
`

// listSockets prints every socket under a directory, one per line.
const listSockets = `
import glob, os, stat, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*"), recursive=True)):
    try:
        if stat.S_ISSOCK(os.stat(path).st_mode):
            print(path)
    except OSError:
        pass
`

// findRuntime defines runtime(socket_path): the PID and command line of the
// runtime process that serves one socket.
const findRuntime = `
import os
def runtime(socket_path):
    for pid in sorted(int(p) for p in os.listdir("/proc") if p.isdigit()):
        # This script's own command line names the socket and the runtime too.
        if pid == os.getpid():
            continue
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as stream:
                argv = stream.read().split(b"\0")
        except OSError:
            continue
        if socket_path.encode() in argv and any(b"vllm-srun" in arg or b"vllm_srun" in arg for arg in argv):
            return pid, [arg.decode() for arg in argv if arg]
    raise SystemExit("no runtime process serves " + socket_path)
`

// killRuntime sends SIGKILL to the runtime process that serves one socket and
// prints its PID, so a test can observe the supervisor restart it.
const killRuntime = findRuntime + `
import signal, sys
pid, _ = runtime(sys.argv[1])
os.kill(pid, signal.SIGKILL)
print(pid)
`

// runtimeArgs prints, as JSON, the command line of the runtime process that
// serves one socket.
const runtimeArgs = findRuntime + `
import json, sys
print(json.dumps(runtime(sys.argv[1])[1]))
`

// PodTarget names a container that can reach managed runtime sockets.
type PodTarget struct {
	Client     kubernetes.Interface
	RestConfig *rest.Config
	Namespace  string
	Pod        string
	Container  string
}

// execTimeout bounds one exec. The in-pod HTTP client gives up after 120 s, so
// a stream still open after this is stuck, and the case fails instead of hanging.
const execTimeout = 3 * time.Minute

// Exec runs argv in the target container with stdin and returns stdout.
func (t PodTarget) Exec(ctx context.Context, argv []string, stdin []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, execTimeout)
	defer cancel()
	request := t.Client.CoreV1().RESTClient().Post().
		Resource("pods").Namespace(t.Namespace).Name(t.Pod).SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: t.Container,
			Command:   argv,
			Stdin:     stdin != nil,
			Stdout:    true,
			Stderr:    true,
		}, scheme.ParameterCodec)
	executor, err := remotecommand.NewSPDYExecutor(t.RestConfig, "POST", request.URL())
	if err != nil {
		return nil, fmt.Errorf("exec in %s/%s: %w", t.Namespace, t.Pod, err)
	}
	var stdout, stderr bytes.Buffer
	options := remotecommand.StreamOptions{Stdout: &stdout, Stderr: &stderr}
	if stdin != nil {
		options.Stdin = bytes.NewReader(stdin)
	}
	if err := executor.StreamWithContext(ctx, options); err != nil {
		return stdout.Bytes(), fmt.Errorf("exec %s in %s/%s: %w: %s", argv[0], t.Namespace, t.Pod, err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

// SocketTransport reaches one managed runtime socket through a PodTarget.
type SocketTransport struct {
	Target PodTarget
	Socket string
}

// Do runs one HTTP exchange over the socket.
func (t SocketTransport) Do(ctx context.Context, method, path string, body []byte) (int, []byte, error) {
	stdin := body
	if stdin == nil {
		stdin = []byte{}
	}
	output, err := t.Target.Exec(ctx, []string{"python3", "-c", unixHTTPClient, t.Socket, method, path}, stdin)
	if err != nil {
		return 0, nil, err
	}
	return parseSocketClientOutput(output)
}

// parseSocketClientOutput splits the socket client's "<status>\n<body>" output.
func parseSocketClientOutput(output []byte) (int, []byte, error) {
	line, payload, found := bytes.Cut(output, []byte("\n"))
	if !found {
		return 0, nil, fmt.Errorf("unexpected socket client output %q", output)
	}
	status, err := strconv.Atoi(string(line))
	if err != nil {
		return 0, nil, fmt.Errorf("unexpected socket client status %q", line)
	}
	return status, payload, nil
}

// Sockets lists the runtime sockets under dir in the target container.
func (t PodTarget) Sockets(ctx context.Context, dir string) ([]string, error) {
	output, err := t.Exec(ctx, []string{"python3", "-c", listSockets, dir}, nil)
	if err != nil {
		return nil, err
	}
	return strings.Fields(string(output)), nil
}

// KillRuntime kills the runtime process serving socket and returns its PID.
func (t PodTarget) KillRuntime(ctx context.Context, socket string) (int, error) {
	output, err := t.Exec(ctx, []string{"python3", "-c", killRuntime, socket}, nil)
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(output)))
}

// RuntimeArgs returns the command line of the runtime process serving socket.
func (t PodTarget) RuntimeArgs(ctx context.Context, socket string) ([]string, error) {
	output, err := t.Exec(ctx, []string{"python3", "-c", runtimeArgs, socket}, nil)
	if err != nil {
		return nil, err
	}
	var argv []string
	if err := json.Unmarshal(output, &argv); err != nil {
		return nil, fmt.Errorf("unexpected command line of the runtime serving %s: %w", socket, err)
	}
	return argv, nil
}

// ManagedRuntimes maps every managed runtime socket under dir to the IDs of
// the models it serves. A socket whose process is not answering yet is
// skipped, so callers poll until the processes they expect appear.
func (t PodTarget) ManagedRuntimes(ctx context.Context, dir string) (map[string][]string, error) {
	sockets, err := t.Sockets(ctx, dir)
	if err != nil {
		return nil, err
	}
	runtimes := make(map[string][]string, len(sockets))
	for _, socket := range sockets {
		models, err := NewClient(SocketTransport{Target: t, Socket: socket}).Models(ctx)
		if err != nil {
			continue
		}
		for _, card := range models.Data {
			runtimes[socket] = append(runtimes[socket], card.ID)
		}
	}
	return runtimes, nil
}

// ClientFor returns a client for the managed runtime that serves model.
func (t PodTarget) ClientFor(ctx context.Context, dir, model string) (*Client, string, error) {
	runtimes, err := t.ManagedRuntimes(ctx, dir)
	if err != nil {
		return nil, "", err
	}
	for socket, models := range runtimes {
		for _, id := range models {
			if id == model {
				return NewClient(SocketTransport{Target: t, Socket: socket}), socket, nil
			}
		}
	}
	return nil, "", fmt.Errorf("no managed runtime under %s serves %q (running: %v)", dir, model, runtimes)
}
