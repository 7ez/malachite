# macOS only: needs Xcode's SDK (xcrun)
SDK  := $(shell xcrun --show-sdk-path)
OUT  := build/Malachite.kext/Contents

KFLAGS := -arch x86_64 -mkernel -nostdinc -I$(SDK)/System/Library/Frameworks/Kernel.framework/Headers \
	-DKERNEL -DKERNEL_PRIVATE -DDRIVER_PRIVATE -DAPPLE -DNeXT -fno-builtin -fno-common -Wall -O2
CXXFLAGS := $(KFLAGS) -fapple-kext -std=c++17 -fno-rtti -fno-exceptions \
	-Wno-deprecated-declarations -Wno-inconsistent-missing-override
CFLAGS := $(KFLAGS) -std=c11

CORE := chip falcon vbios fwsec booter gsp pwr mem bar1 chan gr compute render drawbuild sph texdesc disp dispmode dispdp video bitreader h264 hevc av1 av1cdf av1grain vp9 vp9prob vp8 vp8prob nvdec
# the RM front end speaks NVK's side of the ABI, from the 610 headers in sdk/
RMFE := rmfe rmfe_ctrl rmfe_mem rmfe_chan
SDKINC := -Isdk/610/inc -Isdk/610/unix
OBJS := $(addprefix build/,Device.o plat.o barplace.o $(addsuffix .o,$(CORE) $(RMFE)))

all: $(OUT)/MacOS/Malachite $(OUT)/Info.plist build/malachitectl

build/%.o: kext/%.cpp kext/*.h core/*.h fbkext/surface.h gpu_bridge.h registry.h
	@mkdir -p build
	xcrun clang++ $(CXXFLAGS) -c $< -o $@

build/%.o: core/%.c core/*.h nvrm/*.h
	@mkdir -p build
	xcrun clang $(CFLAGS) -c $< -o $@

build/%.o: core/rmfe/%.c core/rmfe/*.h core/*.h
	@mkdir -p build
	xcrun clang $(CFLAGS) $(SDKINC) -c $< -o $@

# the framebuffer and the accelerator are part of the one kext
build/fb_%.o: fbkext/%.cpp fbkext/*.h core/cursor.h registry.h
	@mkdir -p build
	xcrun clang++ $(CXXFLAGS) -c $< -o $@

FBOBJS := build/fb_Framebuffer.o

build/acc_%.o: accel/%.cpp accel/*.h gpu_bridge.h registry.h
	@mkdir -p build
	xcrun clang++ $(CXXFLAGS) -c $< -o $@

ACCOBJS := build/acc_Accelerator.o build/acc_EventMachine.o build/acc_DisplayMachine.o build/acc_Task.o \
	build/acc_CommandQueue.o build/acc_SysMemory.o build/acc_VidMemory.o build/acc_DisplayPipe.o build/acc_GraphicsControl.o \
	build/acc_Resource.o build/acc_MemoryMap.o

$(OUT)/MacOS/Malachite: $(OBJS) $(FBOBJS) $(ACCOBJS)
	@mkdir -p $(dir $@)
	xcrun clang++ -arch x86_64 -fapple-kext -nostdlib -Xlinker -kext -lkmodc++ -lkmod -lcc_kext $^ -o $@

$(OUT)/Info.plist: kext/Info.plist
	cp $< $@

build/malachitectl: ctl/malachitectl.c kext/abi.h
	xcrun clang -O2 -Wall $< -framework IOKit -lz -o $@

# boots GSP-RM at startup: firmware, malachitectl and a LaunchDaemon
FWDIR := /Library/Application Support/Malachite/570
install-fw:
	sudo mkdir -p "$(FWDIR)"
	sudo cp fw/570/*.bin "$(FWDIR)/"

install: install-fw build/malachitectl
	sudo mkdir -p /usr/local/bin
	sudo cp build/malachitectl /usr/local/bin/malachitectl.new && sudo mv -f /usr/local/bin/malachitectl.new /usr/local/bin/malachitectl
	sudo cp ctl/pl.aochi.malachite.plist /Library/LaunchDaemons/
	sudo chown root:wheel /Library/LaunchDaemons/pl.aochi.malachite.plist

# the kext into the auxiliary collection: it links IOGraphicsFamily and IOAcceleratorFamily2, which only the system
# collection has, so it can't ride in with OpenCore. Takes effect at the next boot.
install-kext: $(OUT)/MacOS/Malachite $(OUT)/Info.plist
	sudo rm -rf /Library/Extensions/Malachite.kext.new
	sudo cp -R build/Malachite.kext /Library/Extensions/Malachite.kext.new
	sudo chown -R root:wheel /Library/Extensions/Malachite.kext.new
	sudo rm -rf /Library/Extensions/Malachite.kext && sudo mv /Library/Extensions/Malachite.kext.new /Library/Extensions/Malachite.kext
	# approve once in System Settings; the kext_policy row is keyed by bundle id, so it covers every later build.
	# the AuxKC must be rebuilt by hand to pick up a changed build (no KDK installed, hence --allow-missing-kdk);
	# kmutil load -p / "kmutil install --update-all" don't do it here
	sudo kmutil create -n aux --allow-missing-kdk --volume-root / \
		--boot-path /System/Library/KernelCollections/BootKernelExtensions.kc \
		--system-path /System/Library/KernelCollections/SystemKernelExtensions.kc \
		--auxiliary-path /Library/KernelCollections/AuxiliaryKernelExtensions.kc

# the user-space Metal driver: a loadable bundle Metal.framework pulls in via the kext's MetalPluginName
MTLBUNDLE := build/MalachiteMetal
AIRC_SRC := compiler/airc.c compiler/front.c compiler/opt.c compiler/back.c compiler/ir.c compiler/llvm_lite.c \
	compiler/sass86.c compiler/tessgen.c core/texdesc.c core/sph.c core/drawbuild.c
$(MTLBUNDLE)/Contents/MacOS/MalachiteMetal: mtl/MalachiteDevice.mm mtl/Info.plist $(AIRC_SRC) compiler/airc.h \
		compiler/llvm_lite.h compiler/nvk.h compiler/sass86.h core/texdesc.h core/sph.h core/drawbuild.h gpu_bridge.h \
		accel/commands.h
	@mkdir -p $(MTLBUNDLE)/Contents/MacOS
	cp mtl/Info.plist $(MTLBUNDLE)/Contents/Info.plist
	xcrun clang -bundle -fobjc-arc -fmodules -Wall -Wno-protocol -O2 \
		-framework Foundation -framework Metal -framework IOKit -framework IOSurface \
		-Xlinker -undefined -Xlinker dynamic_lookup \
		mtl/MalachiteDevice.mm $(AIRC_SRC) -o $@

mtl: $(MTLBUNDLE)/Contents/MacOS/MalachiteMetal

# install as <name>.bundle in a standard GPU bundle dir; gpu_bundle_find_trusted() stats "<dir>/MalachiteMetal.bundle"
install-mtl: $(MTLBUNDLE)/Contents/MacOS/MalachiteMetal
	sudo mkdir -p /Library/GPUBundles
	sudo rm -rf /Library/GPUBundles/MalachiteMetal.bundle.new
	sudo cp -R $(MTLBUNDLE) /Library/GPUBundles/MalachiteMetal.bundle.new
	# WindowServer only loads signed code (ad-hoc is enough). Signed only with SIGN=1 
	# while WindowServer can't composite through it yet: unsigned, it composites on the CPU.
	if [ "$(SIGN)" = 1 ]; then sudo codesign --force --sign - /Library/GPUBundles/MalachiteMetal.bundle.new; fi
	sudo chown -R root:wheel /Library/GPUBundles/MalachiteMetal.bundle.new
	sudo rm -rf /Library/GPUBundles/MalachiteMetal.bundle && sudo mv /Library/GPUBundles/MalachiteMetal.bundle.new /Library/GPUBundles/MalachiteMetal.bundle

# the VideoToolbox H.264, H.265, AV1, VP9 and VP8 decoder, which MalachiteMetal registers from /Library/Video/Plug-Ins
VTBUNDLE := build/NVVideo.bundle
VTSRC := core/bitreader.c core/h264.c core/hevc.c core/av1.c core/av1grain.c core/vp9.c core/vp9prob.c core/vp8.c \
	core/nvdec.c
VTDEPS := $(VTSRC) core/bitreader.h core/h264.h core/hevc.h core/av1.h core/av1cdf.h core/av1grain.h core/vp9.h \
	core/vp9prob.h core/vp8.h core/nvdec.h core/plat.h kext/abi.h
$(VTBUNDLE)/Contents/MacOS/NVVideo: vt/decoder.c vt/Info.plist $(VTDEPS)
	@mkdir -p $(VTBUNDLE)/Contents/MacOS
	cp vt/Info.plist $(VTBUNDLE)/Contents/Info.plist
	# the decoder SPI (CMDerivedObjectCreate, VTDecoderSession*) is resolved at load
	xcrun clang -bundle -arch x86_64 -std=c11 -Wall -O2 \
		-framework CoreFoundation -framework CoreMedia -framework CoreVideo -framework VideoToolbox -framework IOKit \
		-Xlinker -undefined -Xlinker dynamic_lookup \
		vt/decoder.c $(VTSRC) -o $@
	codesign --force --sign - $(VTBUNDLE)

vt: $(VTBUNDLE)/Contents/MacOS/NVVideo

install-vt: $(VTBUNDLE)/Contents/MacOS/NVVideo
	sudo mkdir -p /Library/Video/Plug-Ins
	sudo rm -rf /Library/Video/Plug-Ins/NVVideo.bundle.new
	sudo cp -R $(VTBUNDLE) /Library/Video/Plug-Ins/NVVideo.bundle.new
	sudo chown -R root:wheel /Library/Video/Plug-Ins/NVVideo.bundle.new
	sudo rm -rf /Library/Video/Plug-Ins/NVVideo.bundle && sudo mv /Library/Video/Plug-Ins/NVVideo.bundle.new /Library/Video/Plug-Ins/NVVideo.bundle

clean:
	rm -rf build

.PHONY: all clean install install-fw install-kext mtl install-mtl vt install-vt

-include tools/test.mk
