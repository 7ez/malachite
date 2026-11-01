# Malachite

NVIDIA driver for macOS 26 on OpenCore PCs. RTX 30 series for now. Everything is our own: the kext boots GSP firmware
and drives the card, and the Metal driver compiles Apple AIR straight to SASS. It only builds on macOS.

## Layout

- `kext/`, `core/`, `accel/`, `fbkext/`: kernel driver
- `mtl/`, `compiler/`: Metal driver and shader compiler
- `vt/`: H.264 decoder
- `updater/`: the app, which also installs the driver
- `sdk/`, `nvrm/`, `hw/`: NVIDIA headers, don't edit them by hand

## Build

```
make
make mtl
make vt
sh updater/build.sh
```

After installing, run `/Applications/Malachite.app/Contents/MacOS/Malachite --check`.

## Rules

- Never write NVRAM, from the driver, the installer or recovery. Keep state in files. NVRAM writes can leave AMD boards
  unable to start.
- Keep the AMD `DisableVariableWrite` check in the installer. `--yes` must not skip it.
- Don't hardcode a GPU. Get chips, classes and limits from the card.
- Replace signed binaries with `mv`, never `cp` over them.
- Sign the Metal driver with `SIGN=1`. WindowServer won't load it unsigned, and crash-loops at the next boot.
- A wrong idle wait or cache invalidate can wedge the GPU until a power cycle. Put changes like that behind a sysctl that
  defaults to off, and test them before turning it on.
- Never put signing keys in the repo.
- `VERSION` holds the date and a build number. The build number only goes up, because installed apps compare it to find
  updates. It stays internal, so show people the date.
- Test anything on screen with the desktop unlocked. macOS throttles windows behind the lock screen, so the numbers
  mean nothing there.

## Style

- Conventional Commits, short, no `Co-Authored-By`.
- British English.
- Comments say why, never what.
- Keep the README short.
