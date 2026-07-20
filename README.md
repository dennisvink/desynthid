# DesynthID local reconstruction

This Go CLI performs a local VAE-only image round trip using a Metal-enabled
`stable-diffusion.cpp` server. It downsizes automatically, performs one
zero-denoise VAE encode/decode, upscales to the original dimensions, and writes
a fresh PNG without copying input metadata.

## What SynthID looks like

![SynthID visualization](https://raw.githubusercontent.com/dennisvink/desynthid/refs/heads/main/assets/img/synthid_visualised.jpg)

## Requirements

Tested on Apple Silicon macOS. Install Xcode Command Line Tools, Go, CMake,
Git, and curl. With Homebrew:

```bash
xcode-select --install
brew install go cmake git curl
```

Run the commands below from the repository root.

## 1. Put the runtime source in `.third_party`

Clone the complete `stable-diffusion.cpp` repository, including submodules. Do
not copy individual source files:

```bash
mkdir -p .third_party
git clone --recurse-submodules \
  https://github.com/leejet/stable-diffusion.cpp.git \
  .third_party/stable-diffusion.cpp
git -C .third_party/stable-diffusion.cpp checkout ea4e566ccffa10f853ecc3f29e74b1820bc91beb
git -C .third_party/stable-diffusion.cpp submodule update --init --recursive
```

The recursive checkout supplies `ggml`, the server frontend, `libwebm`, and
`libwebp`.

## 2. Build the Metal runtime

```bash
cmake \
  -S .third_party/stable-diffusion.cpp \
  -B .third_party/stable-diffusion.cpp/build \
  -DSD_METAL=ON \
  -DSD_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build .third_party/stable-diffusion.cpp/build \
  --config Release --parallel

mkdir -p bin
cp .third_party/stable-diffusion.cpp/build/bin/sd-server bin/sd-server
cp .third_party/stable-diffusion.cpp/build/bin/sd-cli bin/sd-cli
chmod +x bin/sd-server bin/sd-cli
```

`bin/sd-server` is required by the app. `bin/sd-cli` is optional and can be
used to inspect PNG metadata.

## 3. Download the models

The app expects exactly this layout:

```text
models/
├── ae.safetensors
├── Qwen3-4B-Instruct-2507-Q4_K_M.gguf
└── z_image_turbo-Q4_0.gguf
```

Download the approximately 6.1 GB of model files:

```bash
mkdir -p models

curl -L -f --retry 3 \
  'https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_0.gguf?download=true' \
  -o models/z_image_turbo-Q4_0.gguf

curl -L -f --retry 3 \
  'https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf?download=true' \
  -o models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

curl -L -f --retry 3 \
  'https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors?download=true' \
  -o models/ae.safetensors
```

Do not put model files in `.third_party`; the Go app loads them from `models/`.

## 4. Build the Go app

```bash
go build -o desynthid .
```

The Go app uses only the standard library.

## 5. Process an image

```bash
./desynthid orig.png
```

This writes `orig_desynthed.png`. To choose the output path:

```bash
./desynthid another.png -output another_desynthed.png
```

The working resolution is automatic. A 4096-pixel-long image uses the
2.5-megapixel target from the original workflow; smaller images use a
proportionally smaller target. The result is restored to the input dimensions.

The app hardcodes one pass and zero denoising strength. It does not apply the
earlier random RGB pixel offsets.

## 6. Visualize differences

For two same-sized images:

```bash
./desynthid diff orig.png orig_desynthed.png
```

This writes `orig_desynthed_diff.png`. Matching pixels are black. Green
intensity is the largest absolute RGB-channel difference: offset 1 is dark
green, offset 2 is brighter, and larger offsets continue toward full green.

An explicit output path is supported:

```bash
./desynthid diff orig.png orig_desynthed.png -output comparison.png
```

## Troubleshooting

Run from the repository root so the relative `bin/` and `models/` paths work.

If a required-file error appears, check the filenames and layout above.

If Metal reports a command-queue error or the server returns `EOF`, run the
program from a normal macOS Terminal session so it can bind to localhost and
access the Metal device.

The automatic downscaling is intentional: full 4096×4096 VAE processing can
require an impractically large temporary buffer.
