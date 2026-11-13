import Foundation

// The install from before the rename: NVBring.kext in OpenCore, NVFB.kext and NVAccel.kext in /Library/Extensions
let oldSupportDir = "/Library/Application Support/NVBring"
// the control tool of the Malachite builds before malachitectl, and of the old driver
let oldCtl = "/usr/local/bin/nvbringctl"
let oldKexts = ["NVFB", "NVAccel", "NVBring"]
private let kextID = "pl.aochi.malachite"
private let policyDB = "/var/db/SystemPolicyConfiguration/KextPolicy"
private let sqlite = "/usr/bin/sqlite3"

// read from the file each time: Bundle caches what it read first
func bundleID(_ path: String) -> String? {
    NSDictionary(contentsOfFile: "\(path)/Contents/Info.plist")?["CFBundleIdentifier"] as? String
}

// a kext or bundle of this driver, from any build, not another driver's of the same name
func ours(_ path: String) -> Bool {
    let id = bundleID(path) ?? ""
    return id.hasPrefix("pl.aochi.") || id.hasPrefix("uk.aochi.")
}

// the driver itself, not just its app
func oldDriverInstalled() -> Bool {
    let paths = ["/Library/GPUBundles/NVMTLDriver.bundle", oldSupportDir] + oldKexts.map { "/Library/Extensions/\($0).kext" }
    return paths.contains(where: exists)
}

// NVBring.kext is turned off in OpenCore; set once that is done, so a rerun doesn't need the EFI again for it
func ocEntryOff() -> Bool {
    exists("\(supportDir)/oc-off")
}

// the choices of the last install, from the support folder or, before the move, the old one
func installOptions() -> [String: String] {
    parseFields([supportDir, oldSupportDir].lazy.compactMap {
        try? String(contentsOfFile: "\($0)/install-options", encoding: .utf8)
    }.first ?? "")
}

func parseFields(_ text: String) -> [String: String] {
    Dictionary(text.split(separator: "\n").compactMap { line in
        let kv = line.split(separator: "=", maxSplits: 1).map(String.init)
        return kv.count == 2 ? (kv[0], kv[1]) : nil
    }, uniquingKeysWith: { _, last in last })
}

// what System Settings > Privacy & Security records when the user presses Allow
func kextApproved() -> Bool {
    // nothing is recorded when consent is off, as it is on some Hackintoshes
    if capture("/usr/sbin/spctl", ["kext-consent", "status"]).text.contains("DISABLED") {
        return true
    }
    let query = "select count(*) from kext_policy where bundle_id = '\(kextID)' and allowed = 1"
    let out = capture(sqlite, [policyDB, query])
    // an unreadable database isn't a refusal, and would stop a move for good; the collection check still has its say
    return out.status != 0 || Int(out.text.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0 > 0
}

private func inCollection() -> Bool {
    capture(kmutil, ["inspect", "-V", "release", "-B", "/Library/KernelCollections/AuxiliaryKernelExtensions.kc"]).text
        .contains(kextID)
}

// Called with Malachite installed and the collection rebuilt, the old driver still in place. Nothing of the old
// driver goes before the new kext is approved and in the collection, and the old OpenCore entry goes off before the
// old kexts do, so a restart in between loads one driver or the other, never both.
func moveToMalachite(efi: String?, steps: Steps, say: Say) throws {
    if !steps.dry, !inCollection() {
        throw Failure(message: "Malachite.kext isn't in the kext collection, see \(kmutilLog). Old driver untouched.")
    }
    say("Moving from NVBring to Malachite")
    if !ocEntryOff() {
        guard let efi else { throw Failure(message: "EFI not mounted. Old driver untouched.") }
        let cfg = "\(efi)/EFI/OC/config.plist"
        let entries = kextEntries(cfg).nvbring
        if !entries.isEmpty {
            // a rerun, or this install's own OpenCore edit, already has the untouched file there
            if !exists(configBackup) {
                try steps.must("/bin/cp", [cfg, configBackup])
            }
            for i in entries {
                try steps.must(plistBuddy, ["-c", "Set :Kernel:Add:\(i):Enabled false", cfg])
                // PlistBuddy says nothing when the volume is read-only
                if !steps.dry, plistValue(":Kernel:Add:\(i):Enabled", cfg) != "false" {
                    throw Failure(message: "Can't write \(cfg), is the EFI read-only? Old driver untouched.")
                }
            }
        }
        try steps.write("", to: "\(supportDir)/oc-off")
        say("  NVBring.kext off in OpenCore (config.plist backed up to \(configBackup))")
    }

    for k in oldKexts {
        try steps.remove("/Library/Extensions/\(k).kext")
    }
    try steps.remove(oldCtl)
    try steps.remove("/Library/GPUBundles/NVMTLDriver.bundle")
    try steps.remove("/Library/GPUBundles/NVMTLDriver.bundle.new")
    if let efi {
        try steps.remove("\(efi)/EFI/OC/Kexts/NVBring.kext")
    }
    say("  Old driver removed, rebuilding the kext collection")
    try rebuildCollection(steps, say: say)

    if exists(oldSupportDir) {
        // the firmware is in the new folder already, from this package, and the install options are written to it
        try steps.remove(oldSupportDir)
    }
}
