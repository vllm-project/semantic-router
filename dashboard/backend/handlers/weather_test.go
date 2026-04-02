package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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

	originalGeocodingURL := weatherGeocodingBaseURL
	originalForecastURL := weatherForecastBaseURL
	weatherGeocodingBaseURL = upstream.URL
	weatherForecastBaseURL = upstream.URL
	defer func() {
		weatherGeocodingBaseURL = originalGeocodingURL
		weatherForecastBaseURL = originalForecastURL
	}()

	req := httptest.NewRequest(http.MethodPost, "/api/tools/weather", strings.NewReader(`{"location":"Chengdu","unit":"celsius"}`))
	rec := httptest.NewRecorder()

	WeatherHandler().ServeHTTP(rec, req)

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

	originalGeocodingURL := weatherGeocodingBaseURL
	originalForecastURL := weatherForecastBaseURL
	weatherGeocodingBaseURL = upstream.URL
	weatherForecastBaseURL = upstream.URL
	defer func() {
		weatherGeocodingBaseURL = originalGeocodingURL
		weatherForecastBaseURL = originalForecastURL
	}()

	req := httptest.NewRequest(http.MethodPost, "/api/tools/weather", strings.NewReader(`{"location":"Missing"}`))
	rec := httptest.NewRecorder()

	WeatherHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected status %d, got %d: %s", http.StatusNotFound, rec.Code, rec.Body.String())
	}
}
