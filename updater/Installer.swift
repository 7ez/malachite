import Foundation

struct InstallOptions {
    var package: String
    var editEFI = false
    var efiPath: String?
    var fullBar = false
    var yes = false
    var dryRun = false
    // the GUI has asked about DisableVariableWrite itself, as it has no terminal to ask on
    var variableWriteOff = false
}

// stopped: nothing is broken, and running it again later finishes the job
enum Outcome {
    case done, stopped, failed
}

let stoppedStatus: Int32 = 3

let kmutilLog = "/tmp/malachite-kmutil.log"
let kmutil = "/usr/bin/kmutil"
let kmutilArgs = [
    "create", "-n", "aux", "--allow-missing-kdk", "--volume-root", "/",
    "--boot-path", "/System/Library/KernelCollections/BootKernelExtensions.kc",
    "--system-path", "/System/Library/KernelCollections/SystemKernelExtensions.kc",
    "--auxiliary-path", "/Library/KernelCollections/AuxiliaryKernelExtensions.kc",
]
let configBackup = "\(supportDir)/config.plist.backup"

// every change goes through here, so a dry run prints each one instead of making it
final class Steps {
    let dry: Bool
    let say: Say

    init(dry: Bool, say: @escaping Say) {
        self.dry = dry
        self.say = say
    }

    private func shown(_ path: String, _ args: [String]) -> String {
        let words = args.map { arg in
            arg.contains { " '\"".contains($0) } ? quote(arg) : arg
        }
        return ([URL(fileURLWithPath: path).lastPathComponent] + words).joined(separator: " ")
    }

    // a log is opened fresh, never through a link left at its path: this runs as root and the path is in /tmp
    private func openLog(_ path: String) -> FileHandle? {
        unlink(path)
        let fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o644)
        return fd < 0 ? nil : FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    }

    @discardableResult
    func tool(_ path: String, _ args: [String], log: String? = nil) -> Output {
        if dry {
            say("    would run \(shown(path, args))" + (log.map { " > \($0)" } ?? ""))
            return Output(status: 0, text: "")
        }
        return capture(path, args, log: log.flatMap(openLog))
    }

    func must(_ path: String, _ args: [String]) throws {
        let out = tool(path, args)
        guard out.status == 0 else {
            throw Failure(message: "\(shown(path, args)) failed:\n\(out.text)")
        }
    }

    func mkdir(_ path: String) throws {
        if dry {
            say("    would mkdir -p \(path)")
        } else {
            try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        }
    }

    func remove(_ path: String) throws {
        if dry {
            say("    would rm -rf \(path)")
        } else if (try? FileManager.default.attributesOfItem(atPath: path)) != nil {
            try FileManager.default.removeItem(atPath: path)
        }
    }

    func write(_ text: String, to path: String) throws {
        if dry {
            say("    would write \(path): \(text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces))")
        } else {
            try Data(text.utf8).write(to: URL(fileURLWithPath: path))
        }
    }

    // a fresh copy beside the old one, then a rename, so a running process never sees a half-written file
    func put(_ src: String, _ dst: String, sign: Bool = false, fat: Bool = false) throws {
        let new = dst + ".new"
        try remove(new)
        try must("/bin/cp", ["-R", src, new])
        // FAT has no owners, and a chown failing there mustn't stop an install halfway
        if !fat {
            try must("/usr/sbin/chown", ["-R", "root:wheel", new])
        }
        if sign {
            // WindowServer loads signed code only
            try must("/usr/bin/codesign", ["--force", "--sign", "-", new])
        }
        try remove(dst)
        if dry {
            say("    would mv \(new) \(dst)")
        } else {
            try FileManager.default.moveItem(atPath: new, toPath: dst)
        }
    }
}

func exists(_ path: String) -> Bool {
    (try? FileManager.default.attributesOfItem(atPath: path)) != nil
}

// the first file or folder a package needs that isn't there
func payloadMissing(_ pkg: String) -> String? {
    let need = ["Kexts/Malachite.kext", "Bundles/MalachiteMetal.bundle", "Bundles/NVVideo.bundle", "bin/malachitectl", "tests",
                "firmware/570"]
    return need.first { !exists("\(pkg)/\($0)") }
}

// the folder the unpacked zip has this app in: App/Malachite.app beside the payload
func defaultPackage() -> String {
    Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent().path
}

func requireRoot(_ flag: String, say: Say) -> Bool {
    guard geteuid() == 0 else {
        say("Run it with sudo: sudo \(quote(selfPath)) \(flag)")
        return false
    }
    return true
}

func install(_ opts: InstallOptions, say: @escaping Say, ask: @escaping Ask) -> Outcome {
    guard requireRoot("--install", say: say) else { return .failed }
    var o = opts
    let pkg = o.package
    if let missing = payloadMissing(pkg) {
        say("\(pkg) isn't a driver package: \(missing) is missing")
        return .failed
    }
    let saved = installOptions()
    if saved["variable-write-off"] == "1" {
        o.variableWriteOff = true
    }
    // the EFI is opened for the options asked for, and to turn off (and clear out) the NVBring.kext an older install
    // put in OpenCore; every other install and update leaves it alone
    let legacy = oldDriverInstalled()
    let found = o.editEFI || legacy ? findEFI(o.efiPath ?? saved["efi"], say: say) : nil
    defer { found?.release() }
    let efi = found?.path
    if legacy, efi == nil, !ocEntryOff() {
        say("Can't find the EFI, nothing changed. Mount it and try again, or use --efi /Volumes/<name>.")
        return .stopped
    }
    let confirm: Ask = o.dryRun ? { say("\($0)y (dry run)"); return true } : ask
    guard checkMac(setsVariableWriteOff: o.editEFI && efi != nil, o, say: say, ask: confirm) else { return .failed }
    let steps = Steps(dry: o.dryRun, say: say)
    do {
        return try copyPayload(pkg, efi: efi, legacy: legacy, saved: saved, o, steps: steps, say: say)
    } catch {
        say((error as? Failure)?.message ?? error.localizedDescription)
        return .failed
    }
}

private func copyPayload(_ pkg: String, efi: String?, legacy: Bool, saved: [String: String], _ o: InstallOptions,
                         steps: Steps, say: Say) throws -> Outcome {
    let fm = FileManager.default
    say("Installing")
    try steps.mkdir("\(supportDir)/570")
    // gsp.bin and GA10x's set, then a folder for each other family's (570/ad10x)
    let firmware = (try? fm.contentsOfDirectory(atPath: "\(pkg)/firmware/570").sorted()) ?? []
    for f in firmware {
        var dir: ObjCBool = false
        if f.hasSuffix(".bin") || fm.fileExists(atPath: "\(pkg)/firmware/570/\(f)", isDirectory: &dir) && dir.boolValue {
            try steps.put("\(pkg)/firmware/570/\(f)", "\(supportDir)/570/\(f)")
        }
    }
    say("  GSP firmware")

    try steps.mkdir("/Library/GPUBundles")
    try steps.put("\(pkg)/Bundles/MalachiteMetal.bundle", "/Library/GPUBundles/MalachiteMetal.bundle", sign: true)
    say("  Metal driver")

    try steps.mkdir("/Library/Video/Plug-Ins")
    try steps.put("\(pkg)/Bundles/NVVideo.bundle", "/Library/Video/Plug-Ins/NVVideo.bundle")
    say("  H.264 video decoder")

    try steps.mkdir("/usr/local/bin")
    try steps.put("\(pkg)/bin/malachitectl", "/usr/local/bin/malachitectl")
    // on a move from the old driver it goes with the rest of that, once Malachite is known to be in the collection
    if !legacy {
        try steps.remove(oldCtl)
    }
    say("  malachitectl")

    try steps.put("\(pkg)/bin", "\(supportDir)/bin")
    try steps.put("\(pkg)/tests", "\(supportDir)/tests")
    // the scripts older packages put here
    for old in ["check.sh", "collect-logs.sh"] where exists("\(supportDir)/\(old)") {
        try steps.remove("\(supportDir)/\(old)")
    }
    say("  test programs")

    // the app changes the extra resolutions as the user, without asking for a password
    if !exists(scaledFile) {
        try steps.write("", to: scaledFile)
    }
    steps.tool("/usr/sbin/chown", ["root:admin", scaledFile])
    steps.tool("/bin/chmod", ["664", scaledFile])

    // tells the app a move from the old install is unfinished
    if legacy {
        try steps.write("", to: "\(supportDir)/migrating")
    }
    try steps.put("\(pkg)/Kexts/Malachite.kext", "/Library/Extensions/Malachite.kext")
    say("  Malachite.kext (rebuilding the kext collection, takes a minute)")
    try rebuildCollection(steps, say: say)

    // the driver works without it, so a failure here doesn't stop the install
    if rootPatchState()["on"] != "0" {
        do {
            try patchRoot(steps: steps, say: say)
        } catch {
            say("  ! System volume not patched, so browsers draw WebGL, video and canvas on the CPU. "
                + ((error as? Failure)?.message ?? error.localizedDescription))
        }
    }

    if let efi, o.editEFI {
        try editOpenCore(efi: efi, o, keepBar: legacy, steps: steps, say: say)
    }

    if !steps.dry, !kextApproved() {
        // asks macOS to show its approval prompt, which a load attempt raises
        steps.tool(kmutil, ["load", "-p", "/Library/Extensions/Malachite.kext"])
        say("Allow Malachite in System Settings > Privacy & Security, then restart if asked.")
        if legacy {
            say("Then update again.")
            return .stopped
        }
    }

    // before the old support folder, where they were, goes
    try writeOptions(efi: efi, o, saved: saved, steps: steps)
    if legacy {
        try moveToMalachite(efi: efi, steps: steps, say: say)
    }

    try installApp(pkg, steps: steps, say: say)
    // the app reads this as "installed", so it comes last
    if exists("\(pkg)/VERSION.txt") {
        try steps.put("\(pkg)/VERSION.txt", "\(supportDir)/VERSION.txt")
    }
    try steps.remove("\(supportDir)/migrating")
    try steps.remove("\(supportDir)/oc-off")

    say("Done. Keep Remote Login on in case the screen stays dark.")
    return .done
}

func rebuildCollection(_ steps: Steps, say: Say) throws {
    let out = steps.tool(kmutil, kmutilArgs, log: kmutilLog)
    if out.status != 0 {
        throw Failure(message: "kmutil failed, see \(kmutilLog)")
    }
}

private func installApp(_ pkg: String, steps: Steps, say: Say) throws {
    let app = "/Applications/Malachite.app"
    if exists("\(pkg)/App/Malachite.app") {
        try steps.put("\(pkg)/App/Malachite.app", app)
        try steps.must("/usr/bin/xattr", ["-cr", app])
        say("  Malachite app")
    }
    // the app this one replaces: it may be the one running, which keeps going on its open files
    try steps.remove("/Applications/NVIDIA Driver.app")
}

// what the app reads on the next update
private func writeOptions(efi: String?, _ o: InstallOptions, saved: [String: String], steps: Steps) throws {
    let path = efi ?? saved["efi"] ?? ""
    let bar = o.fullBar || saved["full-bar"] == "1"
    let off = o.variableWriteOff || (o.editEFI && efi != nil && isAMD())
    try steps.write("efi=\(path)\nfull-bar=\(bar ? 1 : 0)\nvariable-write-off=\(off ? 1 : 0)\n", to: "\(supportDir)/install-options")
}

// the edits to OpenCore's config.plist, the first of which copies the file aside (one copy, outside the small EFI)
private final class ConfigEdit {
    let cfg: String
    let steps: Steps
    var saved = false

    init(cfg: String, steps: Steps) {
        self.cfg = cfg
        self.steps = steps
    }

    func run(_ commands: [String]) throws {
        if !saved {
            try steps.must("/bin/cp", [cfg, configBackup])
            saved = true
        }
        try steps.must(plistBuddy, commands.flatMap { ["-c", $0] } + [cfg])
    }
}

// the options asked for with --efi: the BAR quirk, and DisableVariableWrite on AMD. A move from the old install keeps
// the quirk: it may have been set by hand, which install-options doesn't know about
private func editOpenCore(efi: String, _ o: InstallOptions, keepBar: Bool, steps: Steps, say: Say) throws {
    let cfg = "\(efi)/EFI/OC/config.plist"
    let edit = ConfigEdit(cfg: cfg, steps: steps)

    // macOS leaves a GPU BAR over 1 GB unassigned. The driver then places BAR1 itself at full size (all of VRAM
    // CPU-visible); ResizeAppleGpuBars 8 instead has OpenCore shrink it to 256 MB for macOS alone (Windows keeps
    // Resizable BAR). --full-bar takes the quirk away again.
    let key = ":Booter:Quirks:ResizeAppleGpuBars"
    let bars = plistValue(key, cfg) ?? ""
    if keepBar {
        say("  ResizeAppleGpuBars kept at \(bars.isEmpty ? "the default" : bars)")
    } else if o.fullBar {
        if !bars.isEmpty, bars != "-1" {
            try edit.run(["Set \(key) -1"])
            say("  ResizeAppleGpuBars -1")
        }
    } else if bars.isEmpty {
        try edit.run(["Add \(key) integer 8"])
        say("  ResizeAppleGpuBars 8")
    } else if let n = Int(bars), n < 0 || n > 8 {
        try edit.run(["Set \(key) 8"])
        say("  ResizeAppleGpuBars 8")
    }
    if isAMD() {
        let quirk = ":Booter:Quirks:DisableVariableWrite"
        if plistValue(quirk, cfg) != nil {
            try edit.run(["Set \(quirk) true"])
        } else {
            try edit.run(["Add \(quirk) bool true"])
        }
        say("  DisableVariableWrite on")
    }
    if edit.saved {
        say("  config.plist backed up to \(configBackup)")
    }
}

struct UninstallOptions {
    var editEFI = false
    var efiPath: String?
    var dryRun = false
}

func uninstall(_ o: UninstallOptions, say: @escaping Say) -> Bool {
    guard requireRoot("--uninstall", say: say) else { return false }
    let found = o.editEFI ? findEFI(o.efiPath, say: say) : nil
    defer { found?.release() }
    let efi = found?.path
    let steps = Steps(dry: o.dryRun, say: say)
    do {
        // first, so a failure leaves everything in place to try again
        if exists(systemBundle) || rootPatchState()["patched"] != nil {
            try unpatchRoot(steps: steps, say: say)
        }
        var paths = [
            "/Library/GPUBundles/MalachiteMetal.bundle", "/Library/GPUBundles/MalachiteMetal.bundle.new",
            "/Library/Video/Plug-Ins/NVVideo.bundle", "/Library/Video/Plug-Ins/NVVideo.bundle.new",
            supportDir, "/usr/local/bin/malachitectl", oldCtl, "/Applications/Malachite.app", "/Library/Extensions/Malachite.kext",
        ]
        // an install from before the rename, if the move to Malachite never finished
        let old = oldDriverInstalled()
        if old {
            paths += [oldSupportDir, "/Applications/NVIDIA Driver.app"] + oldBundles()
        }
        for path in paths {
            try steps.remove(path)
        }
        say("Rebuilding the kext collection")
        if steps.tool(kmutil, kmutilArgs, log: kmutilLog).status != 0 {
            say("kmutil failed, see \(kmutilLog). Normal with no other kexts.")
        }
        if let efi {
            let cfg = "\(efi)/EFI/OC/config.plist"
            for i in kextEntries(cfg).nvbring {
                try steps.must(plistBuddy, ["-c", "Set :Kernel:Add:\(i):Enabled false", cfg])
                say("NVBring.kext off in OpenCore")
            }
        } else if old && !ocEntryOff() {
            say("NVBring.kext is still on in OpenCore. Turn it off, or run with --efi /Volumes/EFI.")
        }
    } catch {
        say((error as? Failure)?.message ?? error.localizedDescription)
        return false
    }
    say("Restart to finish.")
    return true
}
