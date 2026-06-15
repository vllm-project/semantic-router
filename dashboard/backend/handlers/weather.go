package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const weatherRequestTimeout = 12 * time.Second

var (
	weatherGeocodingBaseURL = "https://geocoding-api.open-meteo.com"
	weatherForecastBaseURL  = "https://api.open-meteo.com"
)

type weatherRequest struct {
	Location string `json:"location"`
	Unit     string `json:"unit,omitempty"`
}

type weatherLookupResult struct {
	Location weatherLocationResult `json:"location"`
	Current  weatherCurrentResult  `json:"current"`
}

type weatherLocationResult struct {
	Name      string  `json:"name"`
	Admin1    string  `json:"admin1,omitempty"`
	Country   string  `json:"country,omitempty"`
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Timezone  string  `json:"timezone"`
}

type weatherCurrentResult struct {
	Time                string   `json:"time"`
	Temperature         float64  `json:"temperature"`
	ApparentTemperature *float64 `json:"apparent_temperature,omitempty"`
	TemperatureUnit     string   `json:"temperature_unit"`
	Condition           string   `json:"condition"`
	WeatherCode         int      `json:"weather_code"`
	IsDay               bool     `json:"is_day"`
	WindSpeed           *float64 `json:"wind_speed,omitempty"`
	WindSpeedUnit       string   `json:"wind_speed_unit,omitempty"`
	WindDirection       *float64 `json:"wind_direction,omitempty"`
	Precipitation       *float64 `json:"precipitation,omitempty"`
	PrecipitationUnit   string   `json:"precipitation_unit,omitempty"`
}

type weatherGeocodingResponse struct {
	Results []struct {
		Name      string  `json:"name"`
		Admin1    string  `json:"admin1"`
		Country   string  `json:"country"`
		Latitude  float64 `json:"latitude"`
		Longitude float64 `json:"longitude"`
		Timezone  string  `json:"timezone"`
	} `json:"results"`
}

type weatherForecastResponse struct {
	CurrentUnits struct {
		Temperature2M       string `json:"temperature_2m"`
		ApparentTemperature string `json:"apparent_temperature"`
		WindSpeed10M        string `json:"wind_speed_10m"`
		Precipitation       string `json:"precipitation"`
	} `json:"current_units"`
	Current struct {
		Time                string   `json:"time"`
		Temperature2M       float64  `json:"temperature_2m"`
		ApparentTemperature *float64 `json:"apparent_temperature"`
		IsDay               int      `json:"is_day"`
		Precipitation       *float64 `json:"precipitation"`
		WeatherCode         int      `json:"weather_code"`
		WindSpeed10M        *float64 `json:"wind_speed_10m"`
		WindDirection10M    *float64 `json:"wind_direction_10m"`
	} `json:"current"`
}

func WeatherHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req weatherRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		location := strings.TrimSpace(req.Location)
		if location == "" {
			http.Error(w, "location is required", http.StatusBadRequest)
			return
		}

		unit := normalizeWeatherUnit(req.Unit)

		ctx, cancel := context.WithTimeout(r.Context(), weatherRequestTimeout)
		defer cancel()

		result, err := fetchWeather(ctx, location, unit)
		if err != nil {
			status := http.StatusBadGateway
			if strings.Contains(err.Error(), "no weather results") {
				status = http.StatusNotFound
			}
			http.Error(w, err.Error(), status)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(result); err != nil {
			http.Error(w, fmt.Sprintf("Failed to encode weather response: %v", err), http.StatusInternalServerError)
		}
	}
}

func normalizeWeatherUnit(unit string) string {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "fahrenheit":
		return "fahrenheit"
	default:
		return "celsius"
	}
}

func fetchWeather(ctx context.Context, location string, unit string) (*weatherLookupResult, error) {
	target, err := geocodeLocation(ctx, location)
	if err != nil {
		return nil, err
	}

	forecast, err := fetchWeatherForecast(ctx, target, unit)
	if err != nil {
		return nil, err
	}

	return &weatherLookupResult{
		Location: weatherLocationResult{
			Name:      target.Name,
			Admin1:    target.Admin1,
			Country:   target.Country,
			Latitude:  target.Latitude,
			Longitude: target.Longitude,
			Timezone:  target.Timezone,
		},
		Current: weatherCurrentResult{
			Time:                forecast.Current.Time,
			Temperature:         forecast.Current.Temperature2M,
			ApparentTemperature: forecast.Current.ApparentTemperature,
			TemperatureUnit:     forecast.CurrentUnits.Temperature2M,
			Condition:           describeWeatherCode(forecast.Current.WeatherCode, forecast.Current.IsDay == 1),
			WeatherCode:         forecast.Current.WeatherCode,
			IsDay:               forecast.Current.IsDay == 1,
			WindSpeed:           forecast.Current.WindSpeed10M,
			WindSpeedUnit:       forecast.CurrentUnits.WindSpeed10M,
			WindDirection:       forecast.Current.WindDirection10M,
			Precipitation:       forecast.Current.Precipitation,
			PrecipitationUnit:   forecast.CurrentUnits.Precipitation,
		},
	}, nil
}

func geocodeLocation(ctx context.Context, location string) (*weatherLocationResult, error) {
	query := url.Values{}
	query.Set("name", location)
	query.Set("count", "1")
	query.Set("language", "en")
	query.Set("format", "json")

	var payload weatherGeocodingResponse
	if err := fetchJSON(ctx, weatherGeocodingBaseURL+"/v1/search?"+query.Encode(), &payload); err != nil {
		return nil, err
	}
	if len(payload.Results) == 0 {
		return nil, fmt.Errorf("no weather results found for %q", location)
	}

	result := payload.Results[0]
	return &weatherLocationResult{
		Name:      result.Name,
		Admin1:    result.Admin1,
		Country:   result.Country,
		Latitude:  result.Latitude,
		Longitude: result.Longitude,
		Timezone:  result.Timezone,
	}, nil
}

func fetchWeatherForecast(ctx context.Context, location *weatherLocationResult, unit string) (*weatherForecastResponse, error) {
	query := url.Values{}
	query.Set("latitude", fmt.Sprintf("%.6f", location.Latitude))
	query.Set("longitude", fmt.Sprintf("%.6f", location.Longitude))
	query.Set("current", "temperature_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m,wind_direction_10m")
	query.Set("timezone", location.Timezone)
	query.Set("temperature_unit", unit)
	query.Set("wind_speed_unit", "kmh")
	query.Set("precipitation_unit", "mm")

	var payload weatherForecastResponse
	if err := fetchJSON(ctx, weatherForecastBaseURL+"/v1/forecast?"+query.Encode(), &payload); err != nil {
		return nil, err
	}

	return &payload, nil
}

func fetchJSON(ctx context.Context, targetURL string, output interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return fmt.Errorf("failed to build request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to fetch weather data: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("weather upstream returned %s", resp.Status)
	}

	if err := json.NewDecoder(resp.Body).Decode(output); err != nil {
		return fmt.Errorf("failed to decode weather response: %w", err)
	}
	return nil
}

func describeWeatherCode(code int, isDay bool) string {
	switch code {
	case 0:
		if isDay {
			return "Clear sky"
		}
		return "Clear night"
	case 1:
		return "Mainly clear"
	case 2:
		return "Partly cloudy"
	case 3:
		return "Overcast"
	case 45, 48:
		return "Fog"
	case 51, 53, 55:
		return "Drizzle"
	case 56, 57:
		return "Freezing drizzle"
	case 61, 63, 65:
		return "Rain"
	case 66, 67:
		return "Freezing rain"
	case 71, 73, 75, 77:
		return "Snow"
	case 80, 81, 82:
		return "Rain showers"
	case 85, 86:
		return "Snow showers"
	case 95:
		return "Thunderstorm"
	case 96, 99:
		return "Thunderstorm with hail"
	default:
		return "Unknown conditions"
	}
}
