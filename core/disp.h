#pragma once
#include "gsp.h"

#define DISP_MAX_OUTPUTS 16
#define DISP_MAX_MODES 32
#define DISP_MAX_SCALED 8
#define DISP_MAX_HEADS 8  // the most any NVDisplay has (heads says how many this one has); head h drives window 2h
#define DISP_COMMON 0xd1500073u  // NV04_DISPLAY_COMMON's handle in the display client
#define DISP_HOTPLUG 0xd150007eu  // the hotplug event object's
#define DISP_DP_IRQ 0xd150007fu   // the DisplayPort IRQ_HPD event object's

struct disp_mode {
    uint32_t clock;  // kHz
    uint16_t hactive, hblank, hsync_off, hsync_len;
    uint16_t vactive, vblank, vsync_off, vsync_len;
    uint8_t flags;  // EDID detailed timing byte 17
    uint16_t view_w, view_h;  // a bigger image the head scales down to the timing's size, 0 for none
};

// the size of the image a mode shows
static inline uint32_t mode_w(const struct disp_mode *m)
{
    return m->view_w ? m->view_w : m->hactive;
}

static inline uint32_t mode_h(const struct disp_mode *m)
{
    return m->view_h ? m->view_h : m->vactive;
}

struct disp_size {
    uint32_t w, h;
};

struct disp_output {
    uint32_t id;  // display ID (one bit)
    int connected;
    uint32_t or_type, or_index, protocol;
    int dp;  // a DisplayPort output, as GSP-RM's DFP info says
    char name[18];  // EDID manufacturer and monitor name
    struct disp_mode mode;  // the EDID's preferred one
    struct disp_mode modes[DISP_MAX_MODES];  // every detailed timing it lists, preferred first, scaled ones last
    uint32_t nmodes;
    uint8_t edid[512];      // base block and the first three extensions
    uint32_t edid_size;
    // DP: DPCD 0x0-0xf (DP 1.4 extended caps if present), fastest shared link in 0.27 Gbps units; lanes 0 until read
    uint8_t dpcd[16];
    uint32_t dp_bw, dp_lanes;
};

// a mode as the head's raster registers want it (nouveau's nv50_head_atomic_check_mode)
struct disp_raster {
    uint32_t clock;  // kHz
    uint32_t htotal, hsynce, hblanke, hblanks;
    uint32_t vtotal, vsynce, vblanke, vblanks;
    int nhsync, nvsync;
};

struct dchan {
    struct dma pb;
    uint32_t user, cls, handle, cur;  // cur: dwords pushed
};

struct disp_head {
    struct dchan wndw, wimm;  // its window's
    uint64_t luts;            // the window's input LUT, then two output LUTs the head alternates between
    int olut;                 // the output LUT the head shows
    uint32_t wndw_ntfy;       // offset in the notifier memory of the window's last notifier
    int set_up;               // the window channels and LUTs exist
    uint64_t curs_img;        // two cursor images, once disp_cursor_init has run
    int curs_back;            // which of them the next image is written to
    uint32_t curs_w[2], curs_h[2];  // the size of the image each holds
    uint32_t sor;             // the OR the last modeset drove
    uint32_t display;         // and the display on it, 0 before the first
    struct disp_mode timing;  // and the mode it shows
    uint32_t dp_bw, dp_lanes;  // the DisplayPort link it trained, lanes 0 for none
    int dp_ef;
};

struct disp {
    struct gsp *s;
    struct rm_client c;
    uint64_t inst;
    uint32_t heads, windows;
    struct disp_output out[DISP_MAX_OUTPUTS];
    int n;
    struct dchan core;
    int probe_only;
    uint64_t ntfy;  // notifier memory: the core's at 0, each head's window's and flip semaphore after
    int set_up;     // the root object, the core channel and the notifier memory exist
    struct disp_head head[DISP_MAX_HEADS];
    uint32_t hotplug;    // displays plugged or unplugged since the caller last looked (it clears this)
    uint32_t dp_irq;     // DisplayPort displays that raised IRQ_HPD since then (cleared the same way)
    uint32_t connected;  // as of the last disp_read_outputs
    uint32_t fw_display[DISP_MAX_HEADS];  // what the EFI firmware lit on each head, as of the probe
    uint32_t dp_max_bw;  // the GPU's fastest DisplayPort link rate, 0.27 Gbps units; 0 when GSP-RM didn't say
    int dp_wm_increased;  // its SORs take the higher SST watermark limits
    struct disp_size scaled[DISP_MAX_SCALED];  // images to offer on any panel they suit
    uint32_t nscaled;
    int imp;  // GSP-RM's object that checks modes exists
};

int disp_probe(struct gsp *s, struct disp *d, const struct disp_size *scaled, uint32_t nscaled);
// refills out[] from what GSP-RM says is connected, in the board's display order
void disp_read_outputs(struct disp *d);
// rebuilds every output's scaled modes for these sizes; other modes[] keep their index
void disp_set_scaled(struct disp *d, const struct disp_size *sizes, uint32_t n);
int disp_mode_possible(struct disp *d, int head, const struct disp_mode *t);
void disp_fini(struct disp *d);
// disp_fini without RPCs, for when GSP-RM and its display objects are already gone
void disp_forget(struct disp *d);

// head shows fb (A8R8G8B8, VRAM) on output out in mode t (one of its modes[]). DISP_NO_OUTPUT when the output
// can't be driven (no SOR or DP link), channels untouched; -1 when the engine may need disp_reset()
int disp_modeset(struct disp *d, int head, int out, const struct disp_mode *t, uint64_t fb, uint32_t pitch);
#define DISP_NO_OUTPUT (-2)
// the display channels' memory, notifier, LUTs and cursor images freed; their objects in GSP-RM go separately
void disp_modeset_fini(struct disp *d);
// the display channels torn down after a modeset the engine refused; the next disp_modeset() starts over
void disp_reset(struct disp *d);
// The engine keeps channels allocated after GSP-RM frees them (core IDLE, not DEALLOC), even across an unload, and
// a fresh GSP-RM then hangs allocating the core channel; this deallocates them all, as at power-on.
void disp_release_channels(struct gpu *g);
// three flip semaphores per head at d->ntfy + DISP_FLIP_SEMA, used in turn. The window releases one when the next
// flip takes over, so a flip's copy may only write the word of the flip three before.
#define DISP_FLIP_SEMA(head, k) (0x400 + (head) * 0x30 + ((k) % 3) * 0x10)
// fb shown from the first vblank after semaphore k reads `acquire`; returns once the window channel has the
// methods, which it takes when the previous flip is through
int disp_flip_queue(struct disp *d, int head, uint64_t fb, uint32_t acquire, unsigned k);
// whether the head's last queued flip has been taken up (its notifier written)
int disp_flip_taken(struct disp *d, int head);
// head's output LUT from rgb: 1025 red, green, blue triples, 0x10000 = 1.0 (as lut_identity writes them)
int disp_gamma(struct disp *d, int head, const uint16_t *rgb);
int disp_blank(struct disp *d, int head, int out);
void disp_vblank(struct disp *d, int head, int on);
int disp_cursor_init(struct disp *d, int head);
int disp_cursor_image(struct disp *d, int head, const uint32_t *argb, uint32_t w, uint32_t h);
void disp_cursor_move(struct disp *d, int head, int x, int y);
int disp_copy_firmware(struct gpu *g, uint64_t fb, uint32_t pitch, uint32_t w, uint32_t h);

// DisplayPort (dispdp.c)
int dp_aux_read(struct disp *d, uint32_t id, uint32_t addr, uint8_t *buf, uint32_t size);
int dp_aux_write(struct disp *d, uint32_t id, uint32_t addr, const uint8_t *buf, uint32_t size);
// o's DPCD capabilities and the fastest link it and the GPU share; -1 when the sink doesn't answer over AUX
int disp_dp_caps(struct disp *d, struct disp_output *o, int quiet);
// EDID over I2C-over-AUX into buf, base block plus whatever extensions fit; returns bytes read
uint32_t disp_dp_edid(struct disp *d, uint32_t id, uint8_t *buf, uint32_t size);
// whether the fastest link o can train carries mode t at 8 bpc
int disp_dp_fits(const struct disp *d, const struct disp_output *o, const struct disp_mode *t);
// trains the smallest link that carries t for o on the SOR and sets up the head's SST stream on it
int disp_dp_link(struct disp *d, int head, uint32_t sor, struct disp_output *o, const struct disp_mode *t);
// after o raised IRQ_HPD: its service requests acknowledged, the head's link trained again if it dropped
void disp_dp_service(struct disp *d, int head, const struct disp_output *o);
