# Malachite

NVIDIA driver for macOS 26 on OpenCore PCs. Native Metal, no Vulkan.

NullMoth goes from AIR to SPIR-V to Vulkan to NIR to NAK to reach the same GPU, and they've probably added another layer
since you started reading this. Malachite skips all that.

Still experimental. RTX 30 series only for now, support for other cards is planned.

- Download the app: [Malachite.zip](https://nv.aochi.uk/updates/Malachite.zip)
- Problems: press Send logs in the app, and open an issue with the Log ID

| | |
|---|---|
| macOS | 26 Tahoe, x86_64, built on 26.7.1 |
| GPUs | RTX 30 series (GA10x), developed on an RTX 3060 |
| Metal | Metal 3, used by WindowServer and apps |
| OpenGL | Through Apple's GL-on-Metal |
| OpenCL | Through Apple's OpenCL-on-Metal |
| Video | H.264, H.265, AV1, VP9 and VP8 decode through VideoToolbox |
| Displays | Up to four, HDMI and DisplayPort, high refresh rates |

## Benchmarks

Geekbench and Unigine (windowed) on an RTX 3060 in a macOS 26.7.1 VM (GPU passed through), both drivers in the same VM on
the same day, November 2026.

| | Malachite 2026.11.12-2 | NullMoth 2.9 |
|---|---|---|
| Geekbench 7 Metal | [77140](https://browser.geekbench.com/v7/gpu/392279) | [80819](https://browser.geekbench.com/v7/gpu/390474) |
| Geekbench 7 OpenCL | [52534](https://browser.geekbench.com/v7/gpu/392287) | [43784](https://browser.geekbench.com/v7/gpu/390480), Path Tracer scores about 3k |
| Geekbench 6 Metal | [68802](https://browser.geekbench.com/v6/compute/6971619) | [22235](https://browser.geekbench.com/v6/compute/6970997), Face Detection fails validation |
| Geekbench 6 OpenCL | [56734](https://browser.geekbench.com/v6/compute/6971620) | [74223](https://browser.geekbench.com/v6/compute/6970984) |
| Unigine Heaven 4.0, 1920x1080 | 31.3 fps | doesn't finish, OpenGL falls back to software |
| Unigine Valley 1.0, 1920x1080 | 35.0 fps | doesn't finish, OpenGL falls back to software |

NullMoth's compiler still beats ours in Geekbench 7 Metal overall, most of all in Feature Matching and Super
Resolution, and in every Geekbench 6 OpenCL workload. That's being worked on.

## Install

You need a PC that already boots macOS 26 through OpenCore. If yours doesn't, follow the
[OpenCore Install Guide](https://dortania.github.io/OpenCore-Install-Guide/) first.

1. In `config.plist`, set `csr-active-config` to `FF0F0000` and add `amfi_get_out_of_my_way=1 -amfipassbeta` to
   `boot-args`.
2. On AMD boards, set `Booter > Quirks > DisableVariableWrite` to true and restart once. A kernel panic writes to
   NVRAM, which can leave some boards unable to start.
3. Turn on Remote Login, so you can get back in if the screen stays dark.
4. Open Malachite, press Install, and restart. Allow it in Privacy & Security if macOS asks.

The app also updates the driver, runs its checks and uninstalls it. From a terminal it takes `--install`,
`--uninstall`, `--check`, `--collect-logs` and `--send-logs`.

If a build won't boot, add `-nonvidia` to `boot-args`. The driver leaves the card alone and macOS starts without
acceleration.

## Coming from NullMoth

Two drivers can't own the same card, so remove NullMoth first: pick **1401: Remove NVIDIA driver** in the OpenCore
boot picker, or run its `uninstall.sh`, and restart. Take `nvfb=1 nvaccel=1 nvfbheads=4 -nvkmsnosmooth` out of
`boot-args`, then install Malachite as above.

## Resizable BAR

The app sets OpenCore's `ResizeAppleGpuBars` to 8, which shrinks the BAR to 256 MB for macOS only. Set it to -1 to try
the full BAR, which the driver places itself. That works on some boards and not on others.

## Extra resolutions

Malachite can render more pixels than a monitor has and let the display engine scale them down, like DSR on Windows.
Open Resolutions in the app, turn on the sizes you want and press Use, no restart needed. They also show up in Displays.
Sizes go up to twice the panel and 3840x2160, in the panel's shape. Over SSH:
`sudo /Applications/Malachite.app/Contents/MacOS/Malachite --scaled 2560x1440 3840x2160`.

## Building

On macOS 26 with the Xcode command line tools:

```
make
make mtl
make vt
sh updater/build.sh
```

`sudo make install-kext` and `sudo make install-mtl SIGN=1` install what you built. WindowServer refuses a Metal driver
that isn't signed. For the firmware, unpack `gsp`, `bootloader`, `booter_load` and
`booter_unload` `-570.144.bin.zst` from linux-firmware's `nvidia/ga102/gsp` into `fw/570/`, without the version in their
names.

## Not finished

- Sleep: the display doesn't come back after waking, so turn sleep off.
- OpenGL apps stop at about 90 frames a second on a 240 Hz monitor.
- Apps that lean on the GPU may hit unsupported features. They log them, and those logs are what's needed.

## Special thanks

- [RealistikDash](https://github.com/RealistikDash) for lending me his Claude
- [NiceAesth](https://github.com/NiceAesth) for helping get shader compilation right
