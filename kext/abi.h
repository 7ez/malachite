// The GPU service's user client ABI, for nvrm_xnu.c and malachitectl
#pragma once

// what nvrm_xnu.c calls
enum {
    MALACHITE_SEL_ESCAPE = 1,      // scalar: escape number; struct in and out: its parameters
    MALACHITE_SEL_BIND_FD = 3,     // scalar: user space's fd for this connection, -2 to unbind
    MALACHITE_SEL_WAIT_EVENT = 6,  // scalars: nonstall interrupts seen, timeout in us; out: count now
};

// bring-up, malachitectl
enum {
    MALACHITE_LOAD_FIRMWARE = 100,  // scalars: firmware index, user address, length
    MALACHITE_GSP_START,
    MALACHITE_GSP_STOP,
    MALACHITE_DISP_PROBE,  // connected displays, to the kernel log
    MALACHITE_DISP_SHOW,   // test pattern on the first connected display
    MALACHITE_DISP_READ,      // scalars: user address, length; gets struct malachite_shot, then the head's pixels
    MALACHITE_PM_CYCLE,       // suspend and resume the GPU as system sleep would, without sleeping
    // scalars: stream address and length (Annex-B; AV1 OBUs; VP9/VP8 IVF), output address and length, most pictures,
    // codec (0 H.264, 1 H.265, 2 AV1, 3 VP9, 4 VP8); output gets struct malachite_frames, then pictures in decoding order
    MALACHITE_VIDEO_DECODE,
    MALACHITE_END
};

// before MALACHITE_DISP_READ's B8G8R8A8 rows of pitch bytes
struct malachite_shot {
    uint32_t width, height, pitch, reserved;
};

// IOServiceOpen type for VideoUserClient: open to any process, a GPU context per client
#define MALACHITE_VIDEO_CLIENT 1
// IOServiceOpen type for DisplayUserClient: console user only
#define MALACHITE_DISPLAY_CLIENT 2

enum {
    MALACHITE_DISPLAY_SET_SCALED,  // scalars: width, height of each extra resolution; replaces the list
};

enum {
    // scalars: width and height in macroblocks (H.265, VP9: pictures rounded to 64 luma samples; AV1: 128), surfaces,
    // bitstream bytes, codec (0 H.264, 1 H.265, 2 AV1, 3 VP9, 4 VP8), bytes a sample (1, or 2 for 10-bit). Again on a
    // size change; width 0 opens only the GPU context and NVDEC channel.
    NVVIDEO_OPEN,
    // after writing the setup page and slices; scalars: surface, decode index, error concealment index. Returns when
    // NVDEC is done, status in the setup page.
    NVVIDEO_DECODE,
};

// IOConnectMapMemory64 types after NVVIDEO_OPEN; surfaces are read-only, laid out as core/nvdec.h's nvdec_surface
enum { NVVIDEO_MAP_SETUP, NVVIDEO_MAP_BITS, NVVIDEO_MAP_SURFACE };

struct malachite_frames {
    uint32_t count, reserved[3];
};

// before each picture's NV12 (P010 at two bytes a sample)
struct malachite_frame {
    uint32_t index, idr_count;  // decoding order; display order is by idr_count, then poc
    int32_t poc;
    uint32_t width, height, bps;
};

enum { MALACHITE_FW_GSP, MALACHITE_FW_BOOTLOADER, MALACHITE_FW_BOOTER_LOAD, MALACHITE_FW_BOOTER_UNLOAD, MALACHITE_FW_COUNT };

#define MALACHITE_FW_NAMES { "gsp.bin", "bootloader.bin", "booter_load.bin", "booter_unload.bin" }
