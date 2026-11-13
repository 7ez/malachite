# GPU support

Every test so far ran on an RTX 3060 (GA106), and a 3060 Ti (GA104) runs it too. This page lists what the driver
assumes about the chip, what that is on each family, and what another family would take.

## Where the chip comes from

`Gpu::start()` reads PMC_BOOT_0 and looks the chipset (bits 28:20) up in `core/chip.c`. Each entry covers a range of
chipsets and holds the engine classes, the CTA shared memory limit and the GSP-RM signature section. Families the driver
can't drive yet keep their entry with `missing` saying why, and start() refuses them with that in the log. The kext
matches any NVIDIA display controller (`kext/Info.plist`), so the chip decides, not the PCI device ID. The ID table in
`chip.c` is only there to name the product in the log before GSP-RM is up, and it can't stand in for the chip: the
RTX 3050 A laptop parts carry RTX 30 names on Ada chips.

| Family | Chipsets | Status |
|---|---|---|
| GA10x (GA102, GA103, GA104, GA106, GA107) | 0x172-0x177 | driven, same path as GA106 |
| AD10x (AD102-AD107) | 0x192-0x197 | untested, starts like GA10x |
| TU10x (TU102, TU104, TU106) | 0x162-0x166 | refused: boot path, QMD, compiler |
| TU11x (TU116, TU117) | 0x167-0x168 | refused: as TU10x |
| GA100, Tegra, Hopper, Blackwell | | not in the table |

Already taken from the card, on every family: VRAM size (0x1183a4), BAR1 size and placement, heads and windows
(GSP-RM), engines and runlists (device info table), GPC/TPC/SM counts and warps per SM (GR info), the GPU's name, the
WPR2 heap size (from VRAM), the copy engine with physical engines behind it (PCE mask).

## Inventory

### Firmware files

- Where: `fw/570/`, `tools/install/make-package.sh`, `updater/Installer.swift`, `kext/abi.h`
  (`MALACHITE_FW_NAMES`), `Gpu::autostart()` (`/Library/Application Support/Malachite/570/`), `ctl/malachitectl.c`.
- GA10x: one set. NVIDIA's 570.144 `gsp`, `bootloader`, `booter_load` and `booter_unload` are byte-identical for
  GA102, GA103, GA104, GA106 and GA107, so the files we ship serve the whole family.
- AD10x: same `gsp.bin`, but its own `bootloader`, `booter_load` and `booter_unload` (signed for Ada), byte-identical
  for AD102-AD107: linux-firmware's `nvidia/ad102/gsp/*-570.144.bin.zst`.
- TU10x/TU11x: another `gsp.bin` (`gsp_tu10x`), its own bootloader, and two booter sets (TU102/4/6 and TU116/7).
  FWSEC on Turing also needs the falcon bootloader from the ACR firmware.
- Layout: `570/gsp.bin` for every family; GA10x's set beside it, where it always was, so installs, updates and apps
  from before keep working; every other family's set in a folder named by `chip->fwdir` (`570/ad10x/`). The package
  carries both sets, the installer copies the folders too (an older app's installer copies only the top-level files),
  `autostart()` reads the chip's folder, and the kext publishes it as `MalachiteFirmwareDir` for malachitectl.
- Linux-firmware also has an Ada `scrubber` (SEC2, clears VRAM past what the VBIOS scrubbed). NVIDIA's driver needs it
  when the pre-scrubbed top 256 MB doesn't hold WPR2 and the heaps (its newer releases run it on every Ada boot, as a
  workaround). Ours take about 193 MB on 24 GB, as on GA10x, so it isn't shipped or run. First lead if booter-load or
  GSP-RM's init fails on an Ada card.

### VBIOS and FWSEC (`core/vbios.c`, `core/fwsec.c`)

- PROM window at 0x300000, IFR/NVGI headers, BIT falcon data, FWSEC descriptor v3 with fuse-selected signatures.
- GA10x and AD10x: as is.
- Turing: descriptor v2 (no fuse version, signature patched by the loader), loaded through the falcon bootloader
  rather than straight into IMEM. Code.
- Laptops: the VBIOS reads from the PROM. Some laptop GPUs have the image only in the system firmware (ACPI `_ROM`),
  and FWSEC isn't found there. Code (an ACPI source) once a laptop tester shows it.

### Falcon, booter and GSP boot (`core/falcon.c`, `core/booter.c`, `core/gsp.c`)

- GA102-style: heavy-secure boot through the falcon BROM, HS header v2 booter on SEC2, GSP reset into RISC-V.
- GA10x, AD10x: as is (Ada takes the same functions with its own signature section).
- Turing: booter loaded the older way (signature patched into DMEM), a different GSP reset, the falcon's legacy DMA
  load. Code, the bulk of Turing's boot work.
- WPR2 layout (`fb_layout`): the heap is sized from VRAM. The 22 MB OS carve-out is LIBOS3, GA10x and AD10x. Turing's
  GSP-RM is LIBOS2, carve-out 0. Data once Turing boots.
- WPR2 still up after a failed boot leaves the card needing a reset, which is why a family with the wrong firmware is
  refused before FWSEC runs.

### Engine classes (`core/chip.c`)

| | GA10x | AD10x | Turing |
|---|---|---|---|
| channel | c56f | c56f | c46f |
| usermode | c561 | c561 | c461 |
| 3D | c797 | c997 | c597 |
| compute | c7c0 | c9c0 | c5c0 |
| copy | c7b5 | c7b5 | c5b5 |
| NVDEC | c7b0 | c9b0 | c4b0 |
| display root, core | c670, c67d | c770, c77d | c570, c57d |
| window, immediate, cursor | c67e, c67b, c67a | c67e, c67b, c67a | c57e, c57b, c57a |

Checked against NVIDIA's per-chip class lists: all five GA10x chips have the GA10x set. The front end for the
user-space driver ABI (`core/rmfe/rmfe.c`) still names c56f and c7b5 for allocation parameter sizes; Turing's c46f and
c5b5 need adding there. Data.

### Usermode, doorbell, interrupts (`core/chan.c`, `core/rmfe/rmfe.c`, `core/gsp.c`)

- Usermode at 0xbb0000 (0xb80000 + 0x30000), doorbell token `runlist << 16 | chid`, the CPU interrupt tree at
  0xb81000.
- The same from Turing to Ada under GSP-RM. Nothing to do.

### Copy engines (`core/chan.c`, `core/rmfe/rmfe_chan.c`)

- GSP-RM lists logical copy engines it gave no physical ones; on GA106 that was LCE4. Both the kernel's copy channel
  and the front end now skip such engines by their PCE mask, so a chip floorswept differently picks one that runs.

### Graphics context and local memory (`core/gr.c`, `kext/Device.cpp`)

- Golden context once per GSP session, the unrestricted register access map for admin channels, local memory sized from
  GR info (warps per SM, SMs per TPC, GPCs times TPCs per GPC).
- Same on all three families.
- The fault decoder (`core/gsp.c`) uses GA100's fault client table. Log only.

### Compute (`core/compute.c`)

- QMD V03_00; shared memory carve-outs 0, 8, 16, 32, 64, 100 KB; 99 KB per CTA (`chip->max_shared`).
- GA10x, AD10x: the same.
- Turing: QMD V02_02 (other field layout) and its carve-outs (up to 64 KB). Code: a second QMD builder.

### 3D, shader headers, textures (`core/drawbuild.c`, `core/render.c`, `core/sph.c`, `core/texdesc.c`)

- Turing+ draw methods only (no pipeline prefetch, which needs Ampere B), SPH v4, Maxwell+ texture headers and
  samplers, the MME macro for priv registers.
- Same on all three families. Nothing to do, past testing.

### Memory (`core/mem.c`, `core/bar1.c`, `mtl/MalachiteDevice.mm`)

- GP100-style page tables (5 levels, 4 KB and 64 KB pages), PTE kinds 0x06 generic, 0x01 Z16, 0x04 ZF32_X24S8,
  BAR1 with 64 KB pages from GSP-RM's tables, four 34 MB scanout slots.
- Same format from Turing to Ada. Nothing to do.

### Display (`core/disp.c`, `core/dispmode.c`, `core/dispdp.c`)

- Heads, windows and outputs from GSP-RM; core and window methods from the C67D/C67E headers; HDMI via the CTRL_HDMI
  controls and GCP.
- GA10x: as is.
- AD10x: C77D is C67D plus infoframe methods at new offsets; every method we send sits where it did. Only the class
  changes, which the table has.
- Turing: C57D/C57E are C67D/C67E without the newer additions (DSI, HDMI FRL, semaphore acquire on the window). The
  methods we send look present at the same offsets, which needs checking method by method (`tools/check-methods.py`).
  Code if any differ.

### Video (`core/video.c`, `core/nvdec.c`, `vt/`)

- Codec support is decided by class: AV1 and VP9 from c7b0, VP8 from c5b0.
- GA10x, AD10x: every codec.
- Turing: no AV1 (right), but the thresholds also turn off VP9 and VP8, which Turing's NVDEC decodes. Data: a codec
  mask per family in `struct chip`, after checking c4b0's VP9 methods.
- There is no encoder code.

### Compiler (`compiler/`)

- `sass86.c` encodes sm86 only; the latency table in `back.c` is sm80/sm86's; occupancy assumes 48 warps per SM.
- GA10x: nothing in it is GA106's. GA102-GA107 are all sm86.
- AD10x: sm89 runs sm86 code, and nvcc's sm86 and sm89 output are byte-identical for what we emit (nvdisasm also
  decodes `sass86_test`'s output the same as SM86 and SM89). Nothing to do. It uses no HMMA, LDGSTS, LDSM or REDUX, so
  Ada's FP8 isn't missed. Its limits are sm86's: 48 warps and 64K registers per SM, 100 KB shared memory per SM, 99 KB
  per CTA. Local memory is sized from GR info, so AD102's 12 GPCs need nothing.
- Turing (sm75), decoded with nvdisasm:
  - LDG/STG memory order bits split into scope and order on sm75: today's LDG decodes as the constant (read-only) path
    and STG as invalid. Per-target encodings. Data.
  - ATOMG/RED and SUST strong bits move. Data.
  - I2FP, F2IP.U8 and F2FP.PACK_AB don't exist: I2F, F2I plus a clamp, F2F plus PRMT, all variable latency. Code.
  - Own latency table and 32 warps per SM. Data, measured on a Turing card.
  - `airc_compile` takes no target, and the shader cache key (`mtl/MalachiteDevice.mm`) doesn't hold the SM version, so
    a card swap would load sm86 code. Code.
- Tests: `make -C compiler check` compares against nvcc sm86; nvcc and nvdisasm on the host take sm75 and sm89 too, so a
  `sass75_test` is cheap.

### Metal plugin (`mtl/MalachiteDevice.mm`)

- Threadgroup memory 48 KB, 1024 threads, SIMD 32, argument buffers tier 2, Metal 3 family, VRAM and core count from
  the kext's properties.
- All fit Turing to Ada. 48 KB leaves GA10x/AD10x's 99 KB unused; publishing `chip->max_shared` would let the plugin
  offer more. Data.

### Accelerator, graphics control, installer

- `accel/Accelerator.cpp` and `accel/GraphicsControl.cpp` start on any NVIDIA display controller, whatever the chip.
  On a refused card graphics control still takes the framebuffer match category, which keeps Apple's framebuffer off
  the boot screen, while `Gpu` never publishes its own: likely no screen at all. They should start only once `Gpu`
  has accepted the chip. Code; needs
  care, as reading BAR0 before `Gpu` has enabled memory decode on a secondary card returns all ones. Until then
  `-nonvidia` is the way back for anyone whose card didn't come up.
- `updater/Checks.swift` takes the first NVIDIA device; Turing isn't checked.
- README says RTX 40 is untested; the app says RTX 30.

## Ampere laptops

GA103, GA104, GA106 and GA107 laptop parts get the same path. What isn't covered:

- Muxless laptops have no display wired to the NVIDIA GPU. With no screens WindowServer falls back to software
  compositing and crash-loops (see the push_surface assert). The driver needs a headless mode before a muxless laptop
  can run it.
- The dGPU may be powered off by the system firmware or an SSDT; the driver has no ACPI power control.
- VBIOS from ACPI, as above.

## What each family still needs

- Other GA10x: nothing known in code. Testing on each chip, GA102's 24 GB and GA107's 4 GB at the ends.
- AD10x: wired up (firmware folder, classes), never run. A tester's first boot log says whether booter-load,
  GSP-RM init, the golden context and the display come up; then the 3D, compute and NVDEC classes need a draw, a
  dispatch and a decode each.
- Turing: firmware set and FWSEC v2, the older booter and falcon load, LIBOS2 WPR layout, QMD V02_02, the compiler's
  sm75 target, video codec mask, and checking the C57x display methods.

Everything past GA106 and GA104 is untested.
