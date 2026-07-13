package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestWeatherHandlerReturnsCurrentConditions(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/search":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"results": []map[string]any{
					{
						"name":      "Chengdu",
						"admin1":    "Sichuan",
						"country":   "China",
						"latitude":  30.67,
						"longitude": 104.06,
						"timezone":  "Asia/Shanghai",
					},
				},
			})
		case "/v1/forecast":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"current_units": map[string]any{
					"temperature_2m":       "°C",
					"apparent_temperature": "°C",
					"wind_speed_10m":       "km/h",
					"precipitation":        "mm",
				},
				"current": map[string]any{
					"time":                 "2026-04-02T16:00",
					"temperature_2m":       24.2,
					"apparent_temperature": 25.1,
					"is_day":               1,
					"precipitation":        0.0,
					"weather_code":         2,
					"wind_speed_10m":       12.3,
					"wind_direction_10m":   160.0,
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/tools/weather", strings.NewReader(`{"location":"Chengdu","unit":"celsius"}`))
	rec := httptest.NewRecorder()

	weatherHandlerWithClient(
		newControlledTestOutboundClient(upstream.Client()),
		upstream.URL,
		upstream.URL,
	).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, rec.Code, rec.Body.String())
	}

	var payload weatherLookupResult
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if payload.Location.Name != "Chengdu" {
		t.Fatalf("expected Chengdu, got %q", payload.Location.Name)
	}
	if payload.Current.Condition != "Partly cloudy" {
		t.Fatalf("expected Partly cloudy, got %q", payload.Current.Condition)
	}
	if payload.Current.TemperatureUnit != "°C" {
		t.Fatalf("expected °C, got %q", payload.Current.TemperatureUnit)
	}
	if !payload.Current.IsDay {
		t.Fatalf("expected daytime current conditions")
	}
}

func TestWeatherHandlerReturnsNotFoundWhenLocationMissing(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{},
		})
	}))
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/tools/weather", strings.NewReader(`{"location":"Missing"}`))
	rec := httptest.NewRecorder()

	weatherHandlerWithClient(
		newControlledTestOutboundClient(upstream.Client()),
		upstream.URL,
		upstream.URL,
	).ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected status %d, got %d: %s", http.StatusNotFound, rec.Code, rec.Body.String())
	}
}

func TestWeatherHandlerRejectsOversizedLocationBeforeOutboundRequest(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected outbound request")
	}}
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/tools/weather",
		strings.NewReader(`{"location":"`+strings.Repeat("界", weatherMaxLocationRunes+1)+`"}`),
	)
	resp := httptest.NewRecorder()
	weatherHandlerWithClient(client, "https://weather.example", "https://weather.example").ServeHTTP(resp, req)
	if resp.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413: %s", resp.Code, resp.Body.String())
	}
	if calls.Load() != 0 {
		t.Fatalf("outbound calls = %d, want zero", calls.Load())
	}
}

func TestWeatherFetchJSONRejectsOversizedResponse(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(strings.Repeat(
				"x",
				weatherMaxResponseSize+1,
			))),
		}, nil
	}}
	var output map[string]any
	err := fetchJSON(context.Background(), client, "https://example.com/weather", &output)
	if !errors.Is(err, errOutboundResponseTooLarge) {
		t.Fatalf("fetchJSON() error = %v, want response too large", err)
	}
}

func TestWeatherFetchJSONRejectsRedirectToPrivateDestination(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://private.weather/data?token=redirect-secret", http.StatusFound)
	}))
	defer server.Close()

	var dialCalls atomic.Int32
	client := newPublicOutboundHTTPClientWithOptions(2*time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(_ context.Context, _ string, host string) ([]netip.Addr, error) {
			switch host {
			case "public.weather":
				return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
			case "private.weather":
				return []netip.Addr{netip.MustParseAddr("127.0.0.1")}, nil
			default:
				return nil, errors.New("unexpected host")
			}
		}),
		dialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			dialCalls.Add(1)
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})
	var output map[string]any
	err := fetchJSON(context.Background(), client, "http://public.weather/data", &output)
	if !errors.Is(err, errWeatherUpstream) {
		t.Fatalf("fetchJSON() error = %v, want safe weather upstream error", err)
	}
	if got := dialCalls.Load(); got != 1 {
		t.Fatalf("dial calls = %d, want private redirect rejected before second dial", got)
	}
	if strings.Contains(err.Error(), "redirect-secret") || strings.Contains(err.Error(), "private.weather") {
		t.Fatalf("redirect error leaked target data: %v", err)
	}
}

func TestWeatherFetchJSONIgnoresAmbientProxy(t *testing.T) {
	t.Setenv("HTTP_PROXY", "http://proxy.invalid:3128")
	t.Setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
	t.Setenv("ALL_PROXY", "http://proxy.invalid:3128")
	t.Setenv("NO_PROXY", "")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer server.Close()

	var dialedAddress atomic.Value
	client := newPublicOutboundHTTPClientWithOptions(2*time.Second, outboundHTTPClientOptions{
		resolver: testOutboundResolver(func(context.Context, string, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("1.1.1.1")}, nil
		}),
		dialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			dialedAddress.Store(address)
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		},
	})
	var output map[string]any
	if err := fetchJSON(context.Background(), client, "http://public.weather/data", &output); err != nil {
		t.Fatalf("fetchJSON() error = %v", err)
	}
	if got, _ := dialedAddress.Load().(string); got != "1.1.1.1:80" {
		t.Fatalf("dialed %q; ambient proxy was not disabled", got)
	}
}
