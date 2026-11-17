package systemonecodec

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// mediaProblem refuses an image or video a codec cannot carry as a runtime
// refuses it: invalid_request with reason image_format or video_format.
func mediaProblem(kind, detail string) *systemoneprotocol.Problem {
	return &systemoneprotocol.Problem{
		Status: http.StatusBadRequest, Kind: systemoneprotocol.ProblemRejected,
		Code: "invalid_request", Reason: kind + "_format", Detail: detail,
	}
}

// mediaPlaceholder names the n-th image or video of a request in its state.
// The brackets keep it from occurring in ordinary text.
func mediaPlaceholder(kind string, n int) string {
	return "⟦vsr-" + kind + "-" + strconv.Itoa(n) + "⟧"
}

// mediaOf reads a request's images and videos members, data URL lists.
func mediaOf(extensions []systemoneprotocol.Member) ([]string, []string, error) {
	var images, videos []string
	for _, member := range extensions {
		if member.Key != "images" && member.Key != "videos" {
			continue
		}
		for index, item := range member.Value.Items() {
			url, ok := item.Text()
			if !ok {
				return nil, nil, mediaProblem(strings.TrimSuffix(member.Key, "s"), fmt.Sprintf("%s[%d] is not a data URL", member.Key, index))
			}
			if member.Key == "images" {
				images = append(images, url)
			} else {
				videos = append(videos, url)
			}
		}
	}
	return images, videos, nil
}

// placeMedia puts a request's images, then its videos, in front of its state
// as assets: the state becomes [placeholders..., state], which a vllm-sr
// server reads as media in front of the state (UnplaceMedia), as vllm-sr/4.0
// read its images and videos. The placeholders do not collide with the
// request's own assets.
func placeMedia(state systemoneprotocol.Value, images, videos []string, existing []systemoneprotocol.Member) (systemoneprotocol.Value, []systemoneprotocol.Member, error) {
	assets := append([]systemoneprotocol.Member{}, existing...)
	if len(images)+len(videos) == 0 {
		return state, assets, nil
	}
	taken := make(map[string]bool, len(existing))
	for _, asset := range existing {
		taken[asset.Key] = true
	}
	placeholders := make([]systemoneprotocol.Value, 0, len(images)+len(videos)+1)
	add := func(kind string, urls []string, decode func(string) (systemoneprotocol.Value, error)) error {
		for index, url := range urls {
			asset, err := decode(url)
			if err != nil {
				return mediaProblem(kind, fmt.Sprintf("%ss[%d]: %v", kind, index, err))
			}
			name := mediaPlaceholder(kind, index+1)
			for n := index + 2; taken[name]; n++ {
				name = mediaPlaceholder(kind, n)
			}
			taken[name] = true
			assets = append(assets, systemoneprotocol.Member{Key: name, Value: asset})
			placeholders = append(placeholders, systemoneprotocol.String(name))
		}
		return nil
	}
	if err := add("image", images, imageAsset); err != nil {
		return systemoneprotocol.Value{}, nil, err
	}
	if err := add("video", videos, videoAsset); err != nil {
		return systemoneprotocol.Value{}, nil, err
	}
	return systemoneprotocol.Array(append(placeholders, state)...), assets, nil
}

// UnplaceMedia reads media placed in front of a state, as a vllm-sr/5.0
// server reads a request: a state that is an array of the placeholders of
// image assets, then of video assets, then exactly one other item has that
// item as its state, and the images and videos (as data URLs) in front of
// it. The other assets stay where their placeholders are. ok is false for
// any other state, which keeps every asset.
func UnplaceMedia(state systemoneprotocol.Value, assets []systemoneprotocol.Member) (systemoneprotocol.Value, []string, []string, []systemoneprotocol.Member, bool, error) {
	items := state.Items()
	if state.Kind() != systemoneprotocol.KindArray || len(items) < 2 {
		return state, nil, nil, assets, false, nil
	}
	byName := make(map[string]systemoneprotocol.Value, len(assets))
	for _, asset := range assets {
		byName[asset.Key] = asset.Value
	}
	var images, videos []string
	used := map[string]bool{}
	lead := 0
	for ; lead < len(items)-1; lead++ {
		name, isText := items[lead].Text()
		asset, isAsset := byName[name]
		if !isText || !isAsset || used[name] {
			break
		}
		kind := textOf(mustGet(asset, "type"))
		if kind == "" {
			kind = "image"
		}
		if kind == "image" && len(videos) > 0 {
			break
		}
		url, err := assetDataURL(kind, asset)
		if err != nil {
			return state, nil, nil, assets, false, fmt.Errorf("asset %q: %w", name, err)
		}
		used[name] = true
		if kind == "video" {
			videos = append(videos, url)
		} else {
			images = append(images, url)
		}
	}
	if lead == 0 || lead != len(items)-1 {
		return state, nil, nil, assets, false, nil
	}
	rest := make([]systemoneprotocol.Member, 0, len(assets))
	for _, asset := range assets {
		if !used[asset.Key] {
			rest = append(rest, asset)
		}
	}
	return items[lead], images, videos, rest, true, nil
}

// imageAsset is a base64 image data URL as a standard image asset; its format
// is given for PNG and JPEG and read from the bytes otherwise.
func imageAsset(url string) (systemoneprotocol.Value, error) {
	mime, data, err := splitDataURL(url)
	if err != nil {
		return systemoneprotocol.Value{}, err
	}
	asset := systemoneprotocol.Object(systemoneprotocol.Member{Key: "type", Value: systemoneprotocol.String("image")})
	switch mime {
	case "image/png":
		asset = asset.With("format", systemoneprotocol.String("png"))
	case "image/jpeg", "image/jpg":
		asset = asset.With("format", systemoneprotocol.String("jpeg"))
	case "image/webp":
	default:
		return systemoneprotocol.Value{}, fmt.Errorf("an image must be a PNG, JPEG or WebP data URL")
	}
	return asset.With("data_base64", systemoneprotocol.String(data)), nil
}

var videoFormats = []struct{ mime, format string }{
	{"video/mp4", "mp4"}, {"video/webm", "webm"}, {"video/quicktime", "mov"}, {"video/x-matroska", "mkv"},
}

// videoAsset is a base64 video data URL as a vllm-sr/5.0 video asset.
func videoAsset(url string) (systemoneprotocol.Value, error) {
	mime, data, err := splitDataURL(url)
	if err != nil {
		return systemoneprotocol.Value{}, err
	}
	for _, known := range videoFormats {
		if known.mime == mime {
			return systemoneprotocol.Object(
				systemoneprotocol.Member{Key: "type", Value: systemoneprotocol.String("video")},
				systemoneprotocol.Member{Key: "format", Value: systemoneprotocol.String(known.format)},
				systemoneprotocol.Member{Key: "data_base64", Value: systemoneprotocol.String(data)},
			), nil
		}
	}
	return systemoneprotocol.Value{}, fmt.Errorf("a video must be an MP4, WebM, QuickTime or Matroska data URL")
}

// assetDataURL is an image or video asset as the data URL vllm-sr/4.0 and the
// internal form carry.
func assetDataURL(kind string, asset systemoneprotocol.Value) (string, error) {
	data := textOf(mustGet(asset, "data_base64"))
	if data == "" {
		return "", fmt.Errorf("a placed medium needs its data_base64")
	}
	format := strings.ToLower(textOf(mustGet(asset, "format")))
	if kind == "video" {
		for _, known := range videoFormats {
			if known.format == format {
				return "data:" + known.mime + ";base64," + data, nil
			}
		}
		return "", fmt.Errorf("a video asset needs its format: mp4, webm, mov or mkv")
	}
	switch format {
	case "png":
		return "data:image/png;base64," + data, nil
	case "jpg", "jpeg":
		return "data:image/jpeg;base64," + data, nil
	case "":
		return "data:" + sniffImage(data) + ";base64," + data, nil
	}
	return "", fmt.Errorf("an image asset's format is png, jpg or jpeg")
}

// sniffImage is the MIME type the leading bytes of a base64 image give.
func sniffImage(data string) string {
	head := data
	if len(head) > 24 {
		head = head[:24]
	}
	raw, _ := base64.StdEncoding.DecodeString(head[:len(head)/4*4])
	switch {
	case len(raw) >= 8 && string(raw[:8]) == "\x89PNG\r\n\x1a\n":
		return "image/png"
	case len(raw) >= 3 && raw[0] == 0xFF && raw[1] == 0xD8 && raw[2] == 0xFF:
		return "image/jpeg"
	case len(raw) >= 12 && string(raw[:4]) == "RIFF" && string(raw[8:12]) == "WEBP":
		return "image/webp"
	}
	return "image/png"
}

func splitDataURL(url string) (string, string, error) {
	header, data, found := strings.Cut(url, ",")
	if !found || !strings.HasPrefix(header, "data:") || !strings.HasSuffix(header, ";base64") {
		return "", "", fmt.Errorf("a medium must be a base64 data URL")
	}
	return strings.ToLower(strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")), data, nil
}
