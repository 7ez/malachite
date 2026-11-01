#include <string.h>
#include "sph.h"

#define TYPE_VTG 1
#define TYPE_PS 2
#define SHADER_VERTEX 1
#define SHADER_TESS_INIT 2
#define SHADER_TESS 3
#define SHADER_PIXEL 5

// word 0 holds type (4:0), version (9:5), shader type (13:10), MRT enable (14), SASS version (20:17)
void sph_vertex(uint32_t sph[32], uint32_t inputs, unsigned outputs, unsigned sysvals)
{
    memset(sph, 0, SPH_BYTES);
    sph[0] = TYPE_VTG | 4 << 5 | SHADER_VERTEX << 10 | 1 << 17;
    sph[4] = 0xffu << 12;  // STORE_REQ_START 0xff (140:147), END 0: no output is read back
    for (unsigned i = 0; i < 32; i++)
        if (inputs >> i & 1)
            sph[6 + i / 8] |= 0xfu << (i % 8 * 4);  // generic input i, xyzw (192 + 4i)
    if (sysvals & SPH_INSTANCE_ID)
        sph[10] |= 1u << 30;  // instance id (350)
    if (sysvals & SPH_VERTEX_ID)
        sph[10] |= 1u << 31;  // vertex id (351)
    sph[13] = 0xfu << 12;     // position xyzw out (428:431)
    if (sysvals & SPH_POINT_SIZE)
        sph[13] |= 1u << 11;  // point size out (427)
    if (sysvals & SPH_LAYER)
        sph[13] |= 1u << 9;  // render target array index out (425)
    if (sysvals & SPH_VIEWPORT)
        sph[13] |= 1u << 10;  // viewport index out (426)
    for (unsigned i = 0; i < outputs; i++)
        sph[13 + (i + 4) / 8] |= 0xfu << ((i + 4) % 8 * 4);  // generic output i, xyzw (432 + 4i)
}

void sph_tess_init(uint32_t sph[32], uint32_t inputs, uint32_t outputs, unsigned patch_outputs, unsigned threads)
{
    memset(sph, 0, SPH_BYTES);
    sph[0] = TYPE_VTG | 4 << 5 | SHADER_TESS_INIT << 10 | 1 << 17 | 1 << 26;  // DOES_LOAD_OR_STORE (26), as NAK has it
    // PER_PATCH_ATTRIBUTE_COUNT (56:63), in words: the levels and the padding after them (0x00..0x1f), then four a
    // generic; its low nibble again at 124:127, its high one at 148:151
    uint32_t words = patch_outputs ? 8 + 4 * patch_outputs : 6;
    sph[1] = words << 24;
    sph[3] = (words & 0xf) << 28;
    sph[2] = threads << 24;  // THREADS_PER_INPUT_PRIMITIVE (88:95)
    // STORE_REQ_START (140:147) and END (152:159): the words of the control points' outputs, as NAK sets them
    uint32_t start = 0xff, end = 0;
    for (unsigned i = 0; i < 32; i++)
        if (outputs >> i & 1) {
            if (start == 0xff)
                start = 0x20 + 4 * i;
            end = 0x23 + 4 * i;
        }
    sph[4] = start << 12 | (words >> 4) << 20 | end << 24;
    for (unsigned i = 0; i < 32; i++) {
        if (inputs >> i & 1)
            sph[6 + i / 8] |= 0xfu << (i % 8 * 4);  // generic input i, xyzw (192 + 4i)
        if (outputs >> i & 1)
            sph[13 + (i + 4) / 8] |= 0xfu << ((i + 4) % 8 * 4);  // generic output i, xyzw (432 + 4i)
    }
}

void sph_tess_eval(uint32_t sph[32], uint32_t inputs, unsigned outputs, unsigned sysvals)
{
    sph_vertex(sph, inputs, outputs, sysvals & (SPH_POINT_SIZE | SPH_LAYER | SPH_VIEWPORT));
    sph[0] = (sph[0] & ~(0xfu << 10)) | SHADER_TESS << 10;
    sph[18] |= 3u << 12;  // tessellation coordinate u, v (588, 589): unclaimed, a draw traps with an SPH error
}

void sph_pixel(uint32_t sph[32], const uint8_t *interp, unsigned inputs, unsigned sysvals, unsigned targets)
{
    memset(sph, 0, SPH_BYTES);
    sph[0] = TYPE_PS | 4 << 5 | SHADER_PIXEL << 10 | 1 << 14 | 1 << 17 | (sysvals & SPH_PS_KILLS ? 1u << 15 : 0);
    // position w in (191) whether used or not: without it a draw traps with SPH error 11
    // z (190) only when read: an IPA of an attribute the SPH leaves out faults the warp (Out Of Range Address)
    sph[5] = 1u << 31 | (sysvals & SPH_PS_POSITION ? 3u << 28 : 0) | (sysvals & SPH_PS_POSITION_Z ? 1u << 30 : 0);
    if (sysvals & SPH_PS_LAYER)
        sph[5] |= 1u << 25;  // render target array index (185)
    if (sysvals & SPH_PS_VIEWPORT)
        sph[5] |= 1u << 26;  // viewport index (186)
    if (sysvals & SPH_PS_POINT_COORD)
        sph[14] |= 3u << 24;  // point sprite s, t (472, 473: system values C from 464)
    for (unsigned i = 0; i < inputs; i++) {
        uint32_t m = interp[i];
        sph[6 + i / 4] |= (m | m << 2 | m << 4 | m << 6) << (i % 4 * 8);  // generic input i, 2 bits per component
    }
    for (unsigned t = 0; t < 8; t++)
        if (targets >> t & 1)
            sph[18] |= 0xfu << (t * 4);  // colour target t RGBA out (576 + 4t)
    if (sysvals & SPH_PS_SAMPLE_MASK)
        sph[19] |= 1;  // sample mask out (608)
    if (sysvals & SPH_PS_DEPTH)
        sph[19] |= 1u << 1;  // OMAP_DEPTH (609)
}
