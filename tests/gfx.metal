#include <metal_stdlib>
using namespace metal;

struct v2f {
    float4 pos [[position]];
    float4 colour;
};

vertex v2f vmain(const device float4 *pos [[buffer(0)]], const device float4 *colour [[buffer(1)]],
                 uint vid [[vertex_id]])
{
    v2f o;
    o.pos = pos[vid];
    o.colour = colour[vid];
    return o;
}

fragment float4 fmain(v2f in [[stage_in]])
{
    return in.colour;
}

struct vin {
    float2 pos [[attribute(0)]];
    float4 colour [[attribute(1)]];
};

// vmain's triangle again, its vertices read through a vertex descriptor
vertex v2f vattr(vin v [[stage_in]])
{
    v2f o;
    o.pos = float4(v.pos, 0, 1);
    o.colour = v.colour;
    return o;
}

struct full {
    float4 pos [[position]];
};

struct inst {
    float4 pos [[position]];
    float4 colour [[flat]];
};

// a unit quad from four corners, one instance per quadrant: red, green, blue, black from the bottom left
vertex inst vquad(const device float2 *corner [[buffer(0)]], uint vid [[vertex_id]], uint iid [[instance_id]])
{
    inst o;
    float2 origin = float2(iid & 1, iid >> 1) - 1;
    o.pos = float4(origin + corner[vid], 0, 1);
    o.colour = float4(iid == 0, iid == 1, iid == 2, 1);
    return o;
}

fragment float4 fflat(inst in [[stage_in]])
{
    return in.colour;
}

// one triangle covering the target: (-1,-1) (3,-1) (-1,3)
vertex full vfull(uint vid [[vertex_id]])
{
    full o;
    o.pos = float4(float((vid & 1) << 2) - 1, float((vid & 2) << 1) - 1, 0, 1);
    return o;
}

// the copy through a sampler declared in the shader
fragment float4 fcopy_cx(full in [[stage_in]], texture2d<float> src [[texture(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    return src.sample(s, in.pos.xy);
}

// screen-space derivatives of linear functions of the position: constant everywhere
fragment float4 fderiv(full in [[stage_in]])
{
    return float4(dfdx(in.pos.x * 0.25f), dfdy(in.pos.y * 0.5f), dfdx(in.pos.y), 1);
}

// framebuffer fetch: invert what the target holds
fragment float4 finvert(full in [[stage_in]], float4 dst [[color(0)]])
{
    return 1 - dst;
}

fragment float4 fconst(full in [[stage_in]])
{
    return float4(0.2f, 0.4f, 0.6f, 0.8f);
}

// a 16x16 texture's texels, as the shader reads them
kernel void readback(device float4 *o [[buffer(0)]], texture2d<float, access::read> t [[texture(0)]],
                     uint2 g [[thread_position_in_grid]])
{
    o[g.y * 16 + g.x] = t.read(g);
}

fragment float4 fhalf(full in [[stage_in]])
{
    return float4(1, 0, 0, 0.5f);
}

// copy the source texel under each pixel, through the sampler too, and average them (they agree)
fragment float4 fcopy(full in [[stage_in]], texture2d<float> src [[texture(0)]], sampler s [[sampler(0)]])
{
    float4 a = src.read(uint2(in.pos.xy));
    float4 b = src.sample(s, in.pos.xy);
    return (a + b) * 0.5f;
}
