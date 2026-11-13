// Turing+ shader program headers (SPHV4, the sdk's clc797sph.h): 128 bytes in front of a vertex or pixel
// shader's code, telling the 3D pipeline what the shader reads and writes.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPH_BYTES 128

enum sph_interp { SPH_FLAT = 1, SPH_PERSPECTIVE = 2, SPH_LINEAR = 3 };

// system values a vertex shader can read
#define SPH_VERTEX_ID 1
#define SPH_INSTANCE_ID 2
#define SPH_POINT_SIZE 4  // written, not read
#define SPH_LAYER 8       // the render target array index, written
#define SPH_VIEWPORT 16   // the viewport index, written

// a vertex shader reading the generic attributes set in the inputs mask and the SPH_* system values in sysvals,
// writing the position and generic outputs 0..outputs-1, xyzw each
void sph_vertex(uint32_t sph[32], uint32_t inputs, unsigned outputs, unsigned sysvals);

// tessellation init: reads the inputs mask of its patch's vertices, writes the outputs mask per control point (a
// thread each, threads per patch) and, once per patch, the levels and patch_outputs generic patch attributes
void sph_tess_init(uint32_t sph[32], uint32_t inputs, uint32_t outputs, unsigned patch_outputs, unsigned threads);

// a tessellation shader reading the generic attributes in the inputs mask of its patch's control points, writing
// what sph_vertex's would (it reads no vertex or instance id)
void sph_tess_eval(uint32_t sph[32], uint32_t inputs, unsigned outputs, unsigned sysvals);

// system values a pixel shader can read
#define SPH_PS_POSITION 1     // window position x, y
#define SPH_PS_POINT_COORD 2  // the point sprite coordinate
#define SPH_PS_KILLS 4        // not a value: the shader discards, and its discarded threads live on as helpers
#define SPH_PS_POSITION_Z 8   // window position z
#define SPH_PS_SAMPLE_MASK 16  // not a value: the shader writes the sample mask, in the register after the colours
#define SPH_PS_LAYER 32        // the render target array index
#define SPH_PS_VIEWPORT 64     // the viewport index
#define SPH_PS_DEPTH 128      // not a value: the shader writes the depth, in the register after its colours' and one

// a pixel shader reading generic inputs 0..inputs-1 interpolated per interp[i] and the SPH_PS_* sysvals; targets has
// a bit per colour target written, its RGBA from R0 on, a vec4 per target in target order
void sph_pixel(uint32_t sph[32], const uint8_t *interp, unsigned inputs, unsigned sysvals, unsigned targets);

#ifdef __cplusplus
}
#endif
