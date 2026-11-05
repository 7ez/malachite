#include "../nvrm/perf.h"
#include "pwr.h"

void pwr_read(struct gsp *s, struct pwr *out)
{
    const struct rm_client *in = gsp_internal(s);
    *out = (struct pwr){ .pstate = ~0u, .gpc_khz = ~0u, .dram_khz = ~0u, .nvd_khz = ~0u };

    NV2080_CTRL_PERF_GET_CURRENT_PSTATE_PARAMS ps = { 0 };
    const NV2080_CTRL_PERF_GET_CURRENT_PSTATE_PARAMS *pr =
        rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_PERF_GET_CURRENT_PSTATE, &ps, sizeof(ps));
    out->pstate_status = gsp_ctrl_status(s);
    if (pr && pr->currPstate)
        out->pstate = __builtin_ctz(pr->currPstate);

    NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS *cp = mem_alloc(sizeof(*cp));
    cp->super.retrievalMode = NV2080_CTRL_PMUMON_RETRIEVAL_MODE_LATEST_SAMPLE_ONLY;
    const NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS *cr =
        rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES, cp, sizeof(*cp));
    out->clk_status = gsp_ctrl_status(s);
    if (cr && cr->super.numSamples) {
        const NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE *c = &cr->samples[0];
        out->gpc_khz = c->gpcClkFreqKHz;
        out->dram_khz = c->dramClkFreqKHz;
        out->nvd_khz = c->nvdClkFreqKHz;
    } else if (cr) {
        out->clk_status = ~0u - 1;  // answered, but with no sample yet
    }
    mem_free(cp);
}
