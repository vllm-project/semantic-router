package gateway

import (
	"crypto/tls"
	"os"
	"sync"
	"time"
)

// keyPairCheckInterval bounds how often handshakes look at a listener's
// certificate files.
var keyPairCheckInterval = time.Second

// LoadTLS returns the TLS settings of a listener that serves the certificate
// chain in certFile with the private key in keyFile. Like Envoy's downstream
// TLS, it accepts TLS 1.2 and later.
//
// The pair reloads when either file changes, as a renewed Kubernetes secret or
// a cert-manager rotation changes them, so new connections get the new
// certificate without a restart. A pair that fails to load leaves the previous
// one serving. reloaded, when set, learns the outcome of each reload, and of a
// file that can no longer be read, once until it can again.
func LoadTLS(certFile, keyFile string, reloaded func(error)) (*tls.Config, error) {
	pair := &keyPair{certFile: certFile, keyFile: keyFile, reloaded: reloaded}
	stamp, err := pair.stamp()
	if err == nil {
		err = pair.load()
	}
	if err != nil {
		return nil, err
	}
	pair.attempted = stamp
	return &tls.Config{GetCertificate: pair.certificate, MinVersion: tls.VersionTLS12}, nil
}

type keyPair struct {
	certFile, keyFile string
	reloaded          func(error)

	mu         sync.Mutex
	current    *tls.Certificate
	attempted  [2]fileStamp
	unreadable bool
	checked    time.Time
}

type fileStamp struct {
	modTime time.Time
	size    int64
}

func (p *keyPair) certificate(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if now := time.Now(); now.Sub(p.checked) >= keyPairCheckInterval {
		p.checked = now
		p.reloadIfChanged()
	}
	return p.current, nil
}

func (p *keyPair) reloadIfChanged() {
	stamp, err := p.stamp()
	if err != nil {
		if !p.unreadable {
			p.unreadable = true
			p.report(err)
		}
		return
	}
	p.unreadable = false
	if stamp == p.attempted {
		return
	}
	p.attempted = stamp
	p.report(p.load())
}

func (p *keyPair) load() error {
	cert, err := tls.LoadX509KeyPair(p.certFile, p.keyFile)
	if err != nil {
		return err
	}
	p.current = &cert
	return nil
}

func (p *keyPair) stamp() ([2]fileStamp, error) {
	var stamp [2]fileStamp
	for i, name := range []string{p.certFile, p.keyFile} {
		info, err := os.Stat(name)
		if err != nil {
			return stamp, err
		}
		stamp[i] = fileStamp{modTime: info.ModTime(), size: info.Size()}
	}
	return stamp, nil
}

func (p *keyPair) report(err error) {
	if p.reloaded != nil {
		p.reloaded(err)
	}
}
