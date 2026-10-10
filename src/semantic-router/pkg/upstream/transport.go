package upstream

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"net"
	"net/http"
	"sync"
	"time"
)

const (
	// maxIdleConnsPerHost bounds the idle pool at Envoy's default
	// max_connections circuit breaker.
	maxIdleConnsPerHost = 1024
	// idleConnTimeout is Envoy's default upstream connection idle timeout.
	idleConnTimeout = time.Hour
	// tlsSessionCacheSize bounds resumable TLS sessions per security domain.
	tlsSessionCacheSize = 64
)

// DialFunc opens a network connection. It matches net.Dialer.DialContext.
type DialFunc func(ctx context.Context, network, address string) (net.Conn, error)

// poolKey identifies a security domain: endpoints with equal keys share one
// connection pool, across clusters and across Sets.
type poolKey struct {
	tls        bool
	serverName string
	connect    time.Duration
	ipv4Only   bool
}

func poolKeyFor(spec *ClusterSpec, connect time.Duration) poolKey {
	key := poolKey{connect: connect, ipv4Only: len(spec.Endpoints) == 1 && spec.Endpoints[0].IPv4Only}
	if spec.TLS != nil {
		key.tls = true
		key.serverName = spec.TLS.ServerName
	}
	return key
}

type pool struct {
	key       poolKey
	transport *http.Transport
	refs      int
}

// poolRegistry hands out connection pools by security domain and closes a
// pool once no cluster uses it.
type poolRegistry struct {
	mu      sync.Mutex
	pools   map[poolKey]*pool
	rootCAs *x509.CertPool
	dial    DialFunc
}

func newPoolRegistry(rootCAs *x509.CertPool, dial DialFunc) *poolRegistry {
	if dial == nil {
		dial = (&net.Dialer{}).DialContext
	}
	return &poolRegistry{pools: map[poolKey]*pool{}, rootCAs: rootCAs, dial: dial}
}

func (r *poolRegistry) acquire(key poolKey) *pool {
	r.mu.Lock()
	defer r.mu.Unlock()
	p, ok := r.pools[key]
	if !ok {
		p = &pool{key: key, transport: r.newTransport(key)}
		r.pools[key] = p
	}
	p.refs++
	return p
}

func (r *poolRegistry) release(p *pool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	p.refs--
	if p.refs > 0 {
		return
	}
	delete(r.pools, p.key)
	p.transport.CloseIdleConnections()
}

// newTransport builds an HTTP/1.1 client transport for one security domain.
// It never consults proxy environment variables, never negotiates
// compression, and bounds TCP connect plus TLS handshake by one connect
// timeout, as Envoy's connect_timeout does.
func (r *poolRegistry) newTransport(key poolKey) *http.Transport {
	network := "tcp"
	if key.ipv4Only {
		network = "tcp4"
	}
	var tlsConfig *tls.Config
	if key.tls {
		tlsConfig = r.tlsConfig(key)
	}
	dial := func(ctx context.Context, address string) (net.Conn, error) {
		if enabled(key.connect) {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, key.connect)
			defer cancel()
		}
		conn, err := r.dial(ctx, network, address)
		if err != nil {
			return nil, &connectError{err: err}
		}
		if tlsConfig == nil {
			return conn, nil
		}
		tlsConn := tls.Client(conn, tlsConfig)
		if err := tlsConn.HandshakeContext(ctx); err != nil {
			_ = conn.Close()
			return nil, &connectError{err: err, handshake: true}
		}
		return tlsConn, nil
	}
	transport := &http.Transport{
		DisableCompression:  true,
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		IdleConnTimeout:     idleConnTimeout,
		// A non-nil empty map keeps the transport on HTTP/1.1, which is what
		// the Envoy template speaks to model backends.
		TLSNextProto: map[string]func(string, *tls.Conn) http.RoundTripper{},
	}
	if key.tls {
		transport.DialTLSContext = func(ctx context.Context, _, address string) (net.Conn, error) {
			return dial(ctx, address)
		}
	} else {
		transport.DialContext = func(ctx context.Context, _, address string) (net.Conn, error) {
			return dial(ctx, address)
		}
	}
	return transport
}

func (r *poolRegistry) tlsConfig(key poolKey) *tls.Config {
	return &tls.Config{
		ServerName:         key.serverName,
		RootCAs:            r.rootCAs,
		MinVersion:         tls.VersionTLS12,
		MaxVersion:         tls.VersionTLS13,
		ClientSessionCache: tls.NewLRUClientSessionCache(tlsSessionCacheSize),
	}
}

// connectError marks a failure to open a connection, so it classifies as a
// connect failure whatever the underlying error is. handshake is set when TCP
// connected and the TLS handshake failed.
type connectError struct {
	err       error
	handshake bool
}

func (e *connectError) Error() string { return "connect: " + e.err.Error() }

func (e *connectError) Unwrap() error { return e.err }
