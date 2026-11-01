// OSDefineMetaClassAndStructors registers the class under its bare token in the one registry every kext shares, and
// a clash there stops the kext loading. Ours live in namespaces under short names, so the name registered (which
// IOClass and serviceMatching use) gets the prefix here instead: Gpu in malachite::rm is "MalachiteGpu".
#pragma once
#include <IOKit/IOLib.h>
#include <libkern/c++/OSMetaClass.h>
#include <libkern/c++/OSSymbol.h>

// The framebuffer is the exception: CoreDisplay takes the GPU's vendor from the framebuffer's class name ("NVDA",
// "AMD", "Intel"...) and without a known one turns IOPresentment off, so its name keeps an NVDA prefix.
#define MalachiteDefineClassNamed(cls, super, name)                                 \
    OSDefineMetaClassWithInit(cls, super, )                                         \
    cls ::MetaClass::MetaClass() : OSMetaClass(name, cls ::superClass, sizeof(cls)) {} \
    OSDefineDefaultStructors(cls, super)

#define MalachiteDefineClass(cls, super) MalachiteDefineClassNamed(cls, super, "Malachite" #cls)

// The driver this one replaced registers its core as NVRM (NVBring.kext, which OpenCore loads before any collection
// does) and runs it with its own NVAccel and NVFB. Two of them on one GPU fight over it, and a restart half way
// through the move to Malachite has both installed, so Malachite stands down while the old core is loaded and the
// old driver, as it always did, keeps the GPU. The old NVAccel and NVFB alone are inert without it: NVFB only
// attaches to a nub that Malachite doesn't publish under that name, and Malachite's accelerator and device control
// outscore the old ones.
inline bool oldDriverLoaded()
{
    const OSSymbol *name = OSSymbol::withCString("NVRM");
    bool loaded = name && OSMetaClass::getMetaClassWithName(name);
    OSSafeReleaseNULL(name);
    if (loaded)
        IOLog("Malachite: the earlier NVBring driver is loaded, standing down\n");
    return loaded;
}
