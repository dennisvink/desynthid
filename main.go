package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	serverPath         = "bin/sd-server"
	diffusionModelPath = "models/z_image_turbo-Q4_0.gguf"
	llmPath            = "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
	vaePath            = "models/ae.safetensors"
	serverPort         = 8188

	// The original workflow reduced a 4096x4096 image to 2.5 megapixels.
	referenceDimension  = 4096
	referenceMegapixels = 2.5
)

type options struct {
	input  string
	output string
}

type diffOptions struct {
	original string
	newImage string
	output   string
}

type img2imgRequest struct {
	InitImages        []string `json:"init_images"`
	Prompt            string   `json:"prompt"`
	NegativePrompt    string   `json:"negative_prompt"`
	Steps             int      `json:"steps"`
	CFGScale          float64  `json:"cfg_scale"`
	DenoisingStrength float64  `json:"denoising_strength"`
	SamplerName       string   `json:"sampler_name"`
	Scheduler         string   `json:"scheduler"`
	Seed              int64    `json:"seed"`
	Width             int      `json:"width"`
	Height            int      `json:"height"`
}

type img2imgResponse struct {
	Images []string `json:"images"`
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "diff" {
		opts, err := parseDiffOptions(os.Args[2:])
		if err == nil {
			err = runDiff(opts)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			fmt.Fprintln(os.Stderr, "usage: desynthid diff <original> <new> [-output <diff.png>]")
			os.Exit(2)
		}
		return
	}

	opts, err := parseOptions()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "usage: desynthid <input> [-output <output.png>]")
		os.Exit(2)
	}
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func parseDiffOptions(args []string) (diffOptions, error) {
	var opts diffOptions
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-output", "--output":
			if i+1 >= len(args) {
				return diffOptions{}, errors.New("-output requires a path")
			}
			i++
			opts.output = args[i]
		default:
			if strings.HasPrefix(args[i], "-") {
				return diffOptions{}, fmt.Errorf("unknown option %q", args[i])
			}
			switch {
			case opts.original == "":
				opts.original = args[i]
			case opts.newImage == "":
				opts.newImage = args[i]
			default:
				return diffOptions{}, errors.New("diff expects exactly two input filenames")
			}
		}
	}
	if opts.original == "" || opts.newImage == "" {
		return diffOptions{}, errors.New("diff requires an original and a new image")
	}
	if opts.output == "" {
		ext := filepath.Ext(opts.newImage)
		base := strings.TrimSuffix(opts.newImage, ext)
		opts.output = base + "_diff.png"
	}
	return opts, nil
}

func runDiff(opts diffOptions) error {
	original, err := loadImage(opts.original)
	if err != nil {
		return err
	}
	newImage, err := loadImage(opts.newImage)
	if err != nil {
		return err
	}
	originalNRGBA := toNRGBA(original)
	newNRGBA := toNRGBA(newImage)
	if originalNRGBA.Bounds().Size() != newNRGBA.Bounds().Size() {
		return fmt.Errorf("images must have the same dimensions: %s vs %s", originalNRGBA.Bounds().Size(), newNRGBA.Bounds().Size())
	}

	result := image.NewNRGBA(originalNRGBA.Bounds())
	var frequencies [256]int64
	for y := 0; y < originalNRGBA.Bounds().Dy(); y++ {
		for x := 0; x < originalNRGBA.Bounds().Dx(); x++ {
			a := originalNRGBA.NRGBAAt(x, y)
			b := newNRGBA.NRGBAAt(x, y)
			delta := max(max(absInt(int(a.R)-int(b.R)), absInt(int(a.G)-int(b.G))), absInt(int(a.B)-int(b.B)))
			frequencies[delta]++
			intensity := delta * 32
			if intensity > 255 {
				intensity = 255
			}
			result.SetNRGBA(x, y, color.NRGBA{G: uint8(intensity), A: 255})
		}
	}

	if err := writePNG(opts.output, result); err != nil {
		return err
	}
	changed := int64(result.Bounds().Dx())*int64(result.Bounds().Dy()) - frequencies[0]
	fmt.Printf("Compared %s and %s\n", opts.original, opts.newImage)
	fmt.Printf("Changed pixels: %d/%d\n", changed, int64(result.Bounds().Dx())*int64(result.Bounds().Dy()))
	fmt.Printf("Wrote %s (black = no difference; green intensity = max RGB offset)\n", opts.output)
	return nil
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func parseOptions() (options, error) {
	var opts options
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-output", "--output":
			if i+1 >= len(args) {
				return options{}, errors.New("-output requires a path")
			}
			i++
			opts.output = args[i]
		default:
			if strings.HasPrefix(args[i], "-") {
				return options{}, fmt.Errorf("unknown option %q", args[i])
			}
			if opts.input != "" {
				return options{}, errors.New("only one input filename is allowed")
			}
			opts.input = args[i]
		}
	}
	if opts.input == "" {
		return options{}, errors.New("an input filename is required")
	}
	if opts.output == "" {
		opts.output = newPath(opts.input)
	}
	return opts, nil
}

func run(opts options) error {
	for _, path := range []string{serverPath, diffusionModelPath, llmPath, vaePath, opts.input} {
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("required file %q: %w", path, err)
		}
	}

	src, err := loadImage(opts.input)
	if err != nil {
		return err
	}
	originalBounds := src.Bounds()
	originalWidth := originalBounds.Dx()
	originalHeight := originalBounds.Dy()
	working := toNRGBA(src)

	megapixels := equivalentMegapixels(originalWidth, originalHeight)
	workingWidth, workingHeight := modelSize(originalWidth, originalHeight, megapixels)
	working = resizeBilinear(working, workingWidth, workingHeight)
	fmt.Printf("Working size: %dx%d (%.3f megapixels)\n", workingWidth, workingHeight, megapixels)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server, done, err := startServer(ctx)
	if err != nil {
		return err
	}
	defer stopServer(server, done)

	client := &http.Client{}
	if err := waitForServer(client, 90*time.Second); err != nil {
		return err
	}

	inputPNG, err := encodePNG(working)
	if err != nil {
		return fmt.Errorf("encode VAE input: %w", err)
	}
	working, err = vaeRoundTrip(client, inputPNG, workingWidth, workingHeight)
	if err != nil {
		return fmt.Errorf("VAE round trip: %w", err)
	}

	working = resizeBilinear(working, originalWidth, originalHeight)
	if err := writePNG(opts.output, working); err != nil {
		return err
	}
	fmt.Printf("Wrote %s (VAE only; no denoising; input metadata was not copied)\n", opts.output)
	return nil
}

func equivalentMegapixels(width, height int) float64 {
	longestSide := width
	if height > longestSide {
		longestSide = height
	}
	scale := float64(longestSide) / referenceDimension
	megapixels := referenceMegapixels * scale * scale
	inputMegapixels := float64(width*height) / 1_000_000
	if megapixels > inputMegapixels {
		return inputMegapixels
	}
	return megapixels
}

func startServer(ctx context.Context) (*exec.Cmd, <-chan error, error) {
	args := []string{
		"--listen-port", fmt.Sprint(serverPort),
		"--diffusion-model", diffusionModelPath,
		"--llm", llmPath,
		"--vae", vaePath,
		"--cfg-scale", "1",
		"--steps", "9",
		"--seed", "119704256080775",
		"--sampling-method", "euler",
		"--scheduler", "simple",
		"--disable-image-metadata",
		"--diffusion-fa",
	}
	cmd := exec.CommandContext(ctx, serverPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("start %s: %w", serverPath, err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	return cmd, done, nil
}

func stopServer(cmd *exec.Cmd, done <-chan error) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
	}
}

func waitForServer(client *http.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/sdapi/v1/samplers", serverPort)
	for time.Now().Before(deadline) {
		response, err := client.Get(url)
		if err == nil {
			_, _ = io.Copy(io.Discard, response.Body)
			_ = response.Body.Close()
			if response.StatusCode == http.StatusOK {
				fmt.Println("Diffusion server ready")
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("diffusion server did not become ready within %s", timeout)
}

func vaeRoundTrip(client *http.Client, inputPNG []byte, width, height int) (*image.NRGBA, error) {
	payload := img2imgRequest{
		InitImages:        []string{base64.StdEncoding.EncodeToString(inputPNG)},
		Prompt:            "faithful reconstruction, preserve the original image",
		NegativePrompt:    "",
		Steps:             9,
		CFGScale:          1,
		DenoisingStrength: 0,
		SamplerName:       "Euler",
		Scheduler:         "Simple",
		Seed:              119704256080775,
		Width:             width,
		Height:            height,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("http://127.0.0.1:%d/sdapi/v1/img2img", serverPort)
	request, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("server returned %s: %s", response.Status, strings.TrimSpace(string(responseBody)))
	}
	var result img2imgResponse
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, fmt.Errorf("decode server response: %w", err)
	}
	if len(result.Images) == 0 {
		return nil, errors.New("server returned no images")
	}
	encoded := result.Images[0]
	if comma := strings.IndexByte(encoded, ','); comma >= 0 && strings.Contains(encoded[:comma], "base64") {
		encoded = encoded[comma+1:]
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("decode generated image: %w", err)
	}
	return decodePNG(decoded)
}

func loadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open input: %w", err)
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("decode input: %w", err)
	}
	return img, nil
}

func decodePNG(data []byte) (*image.NRGBA, error) {
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return toNRGBA(img), nil
}

func encodePNG(img image.Image) ([]byte, error) {
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, img); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

func writePNG(path string, img image.Image) error {
	data, err := encodePNG(img)
	if err != nil {
		return fmt.Errorf("encode output: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

func toNRGBA(src image.Image) *image.NRGBA {
	bounds := src.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			dst.Set(x, y, color.NRGBAModel.Convert(src.At(bounds.Min.X+x, bounds.Min.Y+y)))
		}
	}
	return dst
}

func modelSize(width, height int, megapixels float64) (int, int) {
	scale := sqrt(megapixels * 1_000_000 / float64(width*height))
	return align16(int(float64(width)*scale + 0.5)), align16(int(float64(height)*scale + 0.5))
}

func align16(value int) int {
	if value < 16 {
		return 16
	}
	return (value + 8) / 16 * 16
}

func sqrt(value float64) float64 {
	guess := 1.0
	if value > 1 {
		guess = value
	}
	for i := 0; i < 12; i++ {
		guess = (guess + value/guess) / 2
	}
	return guess
}

func resizeBilinear(src *image.NRGBA, width, height int) *image.NRGBA {
	if src.Bounds().Dx() == width && src.Bounds().Dy() == height {
		return src
	}
	dst := image.NewNRGBA(image.Rect(0, 0, width, height))
	xScale := float64(src.Bounds().Dx()) / float64(width)
	yScale := float64(src.Bounds().Dy()) / float64(height)
	for y := 0; y < height; y++ {
		sy := (float64(y)+0.5)*yScale - 0.5
		y0 := int(sy)
		fy := sy - float64(y0)
		if y0 < 0 {
			y0, fy = 0, 0
		}
		y1 := y0 + 1
		if y1 >= src.Bounds().Dy() {
			y1 = src.Bounds().Dy() - 1
		}
		for x := 0; x < width; x++ {
			sx := (float64(x)+0.5)*xScale - 0.5
			x0 := int(sx)
			fx := sx - float64(x0)
			if x0 < 0 {
				x0, fx = 0, 0
			}
			x1 := x0 + 1
			if x1 >= src.Bounds().Dx() {
				x1 = src.Bounds().Dx() - 1
			}
			p00 := src.NRGBAAt(x0, y0)
			p10 := src.NRGBAAt(x1, y0)
			p01 := src.NRGBAAt(x0, y1)
			p11 := src.NRGBAAt(x1, y1)
			dst.SetNRGBA(x, y, color.NRGBA{
				R: interpolate(p00.R, p10.R, p01.R, p11.R, fx, fy),
				G: interpolate(p00.G, p10.G, p01.G, p11.G, fx, fy),
				B: interpolate(p00.B, p10.B, p01.B, p11.B, fx, fy),
				A: interpolate(p00.A, p10.A, p01.A, p11.A, fx, fy),
			})
		}
	}
	return dst
}

func interpolate(p00, p10, p01, p11 uint8, fx, fy float64) uint8 {
	top := float64(p00)*(1-fx) + float64(p10)*fx
	bottom := float64(p01)*(1-fx) + float64(p11)*fx
	value := top*(1-fy) + bottom*fy
	if value < 0 {
		value = 0
	}
	if value > 255 {
		value = 255
	}
	return uint8(value + 0.5)
}

func newPath(inputPath string) string {
	ext := filepath.Ext(inputPath)
	base := strings.TrimSuffix(inputPath, ext)
	return base + "_desynthed.png"
}
