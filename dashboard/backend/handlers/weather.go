package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	weatherRequestTimeout   = 12 * time.Second
	weatherMaxResponseSize  = 2 * 1024 * 1024
	weatherMaxLocationRunes = 256
	weatherGeocodingBaseURL = "https://geocoding-api.open-meteo.com"
	weatherForecastBaseURL  = "https://api.open-meteo.com"
)

var errWeatherResultsNotFound = errors.New("no weather results found")

var errWeatherUpstream = errors.New("weather upstream request failed")

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
	return weatherHandlerWithClient(
		newPublicOutboundHTTPClient(weatherRequestTimeout),
		weatherGeocodingBaseURL,
		weatherForecastBaseURL,
	)
}

func weatherHandlerWithClient(
	client outboundHTTPClient,
	geocodingBaseURL string,
	forecastBaseURL string,
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req weatherRequest
		if status, err := decodeBoundedJSON(w, r, outboundMaxRequestBodyBytes, &req); err != nil {
			http.Error(w, "Invalid request body", status)
			return
		}

		location := strings.TrimSpace(req.Location)
		if location == "" {
			http.Error(w, "location is required", http.StatusBadRequest)
			return
		}
		if utf8.RuneCountInString(location) > weatherMaxLocationRunes {
			http.Error(w, "location is too long", http.StatusRequestEntityTooLarge)
			return
		}

		unit := normalizeWeatherUnit(req.Unit)

		ctx, cancel := context.WithTimeout(r.Context(), weatherRequestTimeout)
		defer cancel()

		result, err := fetchWeather(ctx, client, geocodingBaseURL, forecastBaseURL, location, unit)
		if err != nil {
			status := http.StatusBadGateway
			message := "Weather service temporarily unavailable"
			if errors.Is(err, errWeatherResultsNotFound) {
				status = http.StatusNotFound
				message = "No weather results found"
			}
			http.Error(w, message, status)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(result); err != nil {
			http.Error(w, "Failed to encode weather response", http.StatusInternalServerError)
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

func fetchWeather(
	ctx context.Context,
	client outboundHTTPClient,
	geocodingBaseURL string,
	forecastBaseURL string,
	location string,
	unit string,
) (*weatherLookupResult, error) {
	target, err := geocodeLocation(ctx, client, geocodingBaseURL, location)
	if err != nil {
		return nil, err
	}

	forecast, err := fetchWeatherForecast(ctx, client, forecastBaseURL, target, unit)
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

func geocodeLocation(
	ctx context.Context,
	client outboundHTTPClient,
	baseURL string,
	location string,
) (*weatherLocationResult, error) {
	query := url.Values{}
	query.Set("name", location)
	query.Set("count", "1")
	query.Set("language", "en")
	query.Set("format", "json")

	var payload weatherGeocodingResponse
	if err := fetchJSON(ctx, client, baseURL+"/v1/search?"+query.Encode(), &payload); err != nil {
		return nil, err
	}
	if len(payload.Results) == 0 {
		return nil, errWeatherResultsNotFound
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

func fetchWeatherForecast(
	ctx context.Context,
	client outboundHTTPClient,
	baseURL string,
	location *weatherLocationResult,
	unit string,
) (*weatherForecastResponse, error) {
	query := url.Values{}
	query.Set("latitude", fmt.Sprintf("%.6f", location.Latitude))
	query.Set("longitude", fmt.Sprintf("%.6f", location.Longitude))
	query.Set("current", "temperature_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m,wind_direction_10m")
	query.Set("timezone", location.Timezone)
	query.Set("temperature_unit", unit)
	query.Set("wind_speed_unit", "kmh")
	query.Set("precipitation_unit", "mm")

	var payload weatherForecastResponse
	if err := fetchJSON(ctx, client, baseURL+"/v1/forecast?"+query.Encode(), &payload); err != nil {
		return nil, err
	}

	return &payload, nil
}

func fetchJSON(
	ctx context.Context,
	client outboundHTTPClient,
	targetURL string,
	output interface{},
) error {
	parsedURL, err := parseOutboundHTTPURL(targetURL)
	if err != nil {
		return errWeatherUpstream
	}
	if validationErr := client.ValidateURL(ctx, parsedURL.String()); validationErr != nil {
		return errWeatherUpstream
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, parsedURL.String(), nil)
	if err != nil {
		return errWeatherUpstream
	}

	resp, err := client.Do(req)
	if err != nil {
		return errWeatherUpstream
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return errWeatherUpstream
	}

	body, err := readBoundedOutboundBody(resp.Body, weatherMaxResponseSize)
	if err != nil {
		return err
	}
	if decodeErr := json.Unmarshal(body, output); decodeErr != nil {
		return errWeatherUpstream
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
