import Foundation

func isAMD() -> Bool {
    capture("/usr/sbin/sysctl", ["-n", "machdep.cpu.vendor"]).text.contains("AMD")
}

private func lines(_ out: Output) -> [String] {
    out.text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
}

private func keep(_ text: String, containing words: [String]) -> [String] {
    text.split(separator: "\n").map(String.init).filter { line in
        words.contains { line.contains($0) }
    }
}

// its kexts or Metal plugin, installed or loaded: its uninstaller leaves its folder in /Library behind
private func otherDriver() -> Bool {
    let ext = "/Library/Extensions"
    let kexts = (try? FileManager.default.contentsOfDirectory(atPath: ext)) ?? []
    let ids = kexts.compactMap { bundleID("\(ext)/\($0)")?.lowercased() }
    let loaded = capture("/usr/sbin/kextstat", []).text.lowercased()
    let metal = "/Library/GPUBundles/NVMTLDriver.bundle"
    return kexts.contains("NVDAStartup.kext") || loaded.contains("com.nvidia.web")
        || ids.contains { $0.hasPrefix("com.nullmoth.") } || loaded.contains("com.nullmoth.")
        || exists("/Library/GPUBundles/nvmtl") || (exists(metal) && !ours(metal))
}

// the checks before installing: false when it can't go on or the user declines
func checkMac(setsVariableWriteOff: Bool, _ o: InstallOptions, say: Say, ask: Ask) -> Bool {
    var warned = false
    func note(_ text: String) {
        say("  ! \(text)")
        warned = true
    }
    say("Checking this Mac")
    let ver = capture("/usr/bin/sw_vers", ["-productVersion"]).text.trimmingCharacters(in: .whitespacesAndNewlines)
    if ver.hasPrefix("26.7") {
        say("  macOS \(ver)")
    } else if ver.hasPrefix("26.") {
        note("macOS \(ver) is untested (built on 26.7.1), WindowServer may crash at login")
    } else {
        say("  macOS \(ver) isn't supported, needs 26")
        return false
    }
    let arch = capture("/usr/bin/uname", ["-m"]).text.trimmingCharacters(in: .whitespacesAndNewlines)
    if arch != "x86_64" {
        say("  \(arch) isn't supported, needs x86_64")
        return false
    }
    // an NVIDIA GA10x (RTX 30) display controller: vendor 0x10de, device 0x22xx-0x25xx
    var vendor = false
    var gpu: String?
    for line in lines(capture("/usr/sbin/ioreg", ["-r", "-c", "IOPCIDevice", "-d", "1", "-l"], timeout: 60)) {
        if line.contains("\"vendor-id\" = <de100000>") {
            vendor = true
        }
        if vendor, line.contains("\"device-id\" = <") {
            gpu = line.trimmingCharacters(in: CharacterSet(charactersIn: " |"))
            break
        }
    }
    guard let gpu else {
        say("  No NVIDIA GPU found")
        return false
    }
    say("  NVIDIA GPU: \(gpu)")
    if capture("/usr/bin/csrutil", ["status"]).text.contains("disabled") {
        say("  SIP disabled")
    } else {
        note("SIP is on, set csr-active-config to FF0F0000 in OpenCore")
    }
    // nvram prints the name, a tab, then the value
    let args = capture("/usr/sbin/nvram", ["boot-args"]).text.split(separator: "\t", maxSplits: 1).dropFirst().first
        .map { $0.trimmingCharacters(in: .newlines) } ?? ""
    for want in ["amfi_get_out_of_my_way=1", "-amfipassbeta"] where !" \(args) ".contains(" \(want) ") {
        note("boot-args are missing \(want)")
    }
    // On AMD boards (AM4/AM5) whose OpenCore config rewrites the firmware memory map, a write to NVRAM from macOS (a
    // kernel panic report is one, and an experimental driver may panic) can corrupt the firmware's variables until the
    // board no longer starts. OpenCore's DisableVariableWrite stops macOS writing NVRAM at all. Neither --yes nor a
    // missing terminal answers for the user here.
    if isAMD(), !setsVariableWriteOff, !o.variableWriteOff {
        say("  On AMD, a kernel panic can write to NVRAM and leave the board unable to start.")
        say("  Set Booter > Quirks > DisableVariableWrite to true in config.plist and restart first, or use --efi.")
        if !ask("  Is it on? [y/N] ") {
            return false
        }
    }
    if otherDriver() {
        note("another NVIDIA driver is installed; remove it first")
    }
    if warned, !o.yes {
        return ask("Install anyway? [y/N] ")
    }
    return true
}

// a quick look at whether the driver is up: the GPU Metal sees, and a few of the driver's own tests on it
func runChecks(say: Say) -> Bool {
    let bin = "\(supportDir)/bin"
    guard FileManager.default.fileExists(atPath: bin) else {
        say("Driver not installed.")
        return false
    }
    say("Metal device:")
    let shown = ["Chipset Model", "VRAM", "Metal", "Resolution", "UI Looks like"]
    // a wedged GPU can leave system_profiler waiting for good
    let displays = capture("/usr/sbin/system_profiler", ["SPDisplaysDataType"], timeout: 60)
    if displays.status != 0 {
        say("system_profiler failed or took over a minute, the GPU may be stuck: send logs, then restart")
    }
    for line in keep(displays.text, containing: shown) {
        say(line)
    }
    say(rootPatchSummary())
    say("")
    let tests = [
        ["mtldraw", "tests/gfx.metal"], ["mtlsimd"], ["mtlsoftmax"], ["mtltess"], ["mtlderiv"],
        ["mtlkernels", "tests/kernels.metal"],
    ]
    for t in tests where FileManager.default.isExecutableFile(atPath: "\(bin)/\(t[0])") {
        say("== \(t[0])")
        let out = capture("\(bin)/\(t[0])", Array(t.dropFirst()), cwd: supportDir, timeout: 60)
        let shownLines = lines(out).filter { !$0.hasPrefix("frame ") }
        // a trailing newline leaves one empty last element, which tail ignores
        let body = shownLines.last == "" ? shownLines.dropLast() : shownLines[...]
        for line in body.suffix(12) {
            say(line)
        }
    }
    return true
}
