// Class names share one registry across all kexts and a clash stops the load, so ours register with a prefix:
// malachite::rm::Gpu is "MalachiteGpu" to IOClass and serviceMatching.
#pragma once
#include <IOKit/IOLib.h>
#include <libkern/c++/OSMetaClass.h>
#include <libkern/c++/OSSymbol.h>

// for the framebuffer, whose name needs an NVDA prefix: CoreDisplay reads the vendor from it and turns IOPresentment
// off for an unknown one
#define MalachiteDefineClassNamed(cls, super, name)                                 \
    OSDefineMetaClassWithInit(cls, super, )                                         \
    cls ::MetaClass::MetaClass() : OSMetaClass(name, cls ::superClass, sizeof(cls)) {} \
    OSDefineDefaultStructors(cls, super)

#define MalachiteDefineClass(cls, super) MalachiteDefineClassNamed(cls, super, "Malachite" #cls)

// NVBring's core registers as NVRM. Mid-upgrade both drivers can be installed and would fight over the GPU, so
// Malachite stands down while it is loaded; NVBring's NVAccel and NVFB are inert without it.
inline bool oldDriverLoaded()
{
    const OSSymbol *name = OSSymbol::withCString("NVRM");
    bool loaded = name && OSMetaClass::getMetaClassWithName(name);
    OSSafeReleaseNULL(name);
    if (loaded)
        IOLog("Malachite: the earlier NVBring driver is loaded, standing down\n");
    return loaded;
}
