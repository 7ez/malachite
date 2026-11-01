// Interface between the kext's GPU service and its user space: NVK's NVRM backend (nvrm_xnu.c) and malachitectl
#pragma once

// what nvrm_xnu.c calls
enum {
    MALACHITE_SEL_ESCAPE = 1,      // scalar: escape number; struct in and out: its parameters
    MALACHITE_SEL_BIND_FD = 3,     // scalar: the file number user space gave this connection, -2 to unbind
    MALACHITE_SEL_WAIT_EVENT = 6,  // scalars: count seen, timeout in us; out: count now (graphics nonstall interrupts)
};

// bring-up, malachitectl
enum {
    MALACHITE_LOAD_FIRMWARE = 100,  // scalars: firmware index, user address, length
    MALACHITE_GSP_START,
    MALACHITE_GSP_STOP,
    MALACHITE_DISP_PROBE,  // what is connected, into the kernel log
    MALACHITE_DISP_SHOW,   // a test pattern on the first connected display
    MALACHITE_DISP_READ,      // scalars: user address, length; gets struct malachite_shot, then what the head shows
    MALACHITE_PM_CYCLE,       // what system sleep does to the GPU, suspend then resume, without sleeping
    // scalars: Annex-B H.264 stream address and length, output address and length, most pictures to decode; the
    // output gets struct malachite_frames, then the pictures that fit, in decoding order
    MALACHITE_H264_DECODE,
    MALACHITE_END
};

// in front of the pixels MALACHITE_DISP_READ copies out: B8G8R8A8 rows of pitch bytes
struct malachite_shot {
    uint32_t width, height, pitch, reserved;
};

// IOServiceOpen type for H.264 decoding (VideoUserClient): open to any process, each client a GPU context of its
// own that goes with it
#define MALACHITE_VIDEO_CLIENT 1

enum {
    // scalars: width and height in macroblocks, surfaces, bitstream bytes; again for a stream of another size.
    // Width 0 makes the GPU context and NVDEC channel alone.
    NVVIDEO_OPEN,
    // once the picture's setup page and slices are written through the mappings; scalars: its surface, its index in
    // decoding order, the error concealment index. Back when NVDEC is done, its status in the setup page.
    NVVIDEO_DECODE,
};

// IOConnectMapMemory64 types on a video client after NVVIDEO_OPEN: the setup page (core/nvdec.h), the bitstream,
// then the surfaces (core/nvdec.h's nvdec_surface layout), read-only
enum { NVVIDEO_MAP_SETUP, NVVIDEO_MAP_BITS, NVVIDEO_MAP_SURFACE };

struct malachite_frames {
    uint32_t count, reserved[3];
};

// in front of each picture's NV12: width * height bytes of luma, then width * height / 2 of interleaved chroma
struct malachite_frame {
    uint32_t index, idr_count;  // decoding order; display order is by idr_count, then poc
    int32_t poc;
    uint32_t width, height, reserved;
};

enum { MALACHITE_FW_GSP, MALACHITE_FW_BOOTLOADER, MALACHITE_FW_BOOTER_LOAD, MALACHITE_FW_BOOTER_UNLOAD, MALACHITE_FW_COUNT };

#define MALACHITE_FW_NAMES { "gsp.bin", "bootloader.bin", "booter_load.bin", "booter_unload.bin" }
