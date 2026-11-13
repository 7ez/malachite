import Foundation

// the install from before the rename had NVBring.kext in OpenCore and NVFB/NVAccel.kext in /Library/Extensions
let oldSupportDir = "/Library/Application Support/NVBring"
// used by the old driver and by Malachite builds before malachitectl
let oldCtl = "/usr/local/bin/nvbringctl"
let oldKexts = ["NVFB", "NVAccel", "NVBring"]
private let kextID = "pl.aochi.malachite"
private let policyDB = "/var/db/SystemPolicyConfiguration/KextPolicy"
private let sqlite = "/usr/bin/sqlite3"

// read from the file each time: Bundle caches what it read first
func bundleID(_ path: String) -> String? {
    NSDictionary(contentsOfFile: "\(path)/Contents/Info.plist")?["CFBundleIdentifier"] as? String
}

// any build of ours, not another driver's bundle of the same name
func ours(_ path: String) -> Bool {
    let id = bundleID(path) ?? ""
    return id.hasPrefix("pl.aochi.") || id.hasPrefix("uk.aochi.")
}

// the driver itself, not just its app
func oldDriverInstalled() -> Bool {
    exists(oldSupportDir) || oldBundles().contains(where: exists)
}

// another driver uses some of the same names
func oldBundles() -> [String] {
    (["/Library/GPUBundles/NVMTLDriver.bundle"] + oldKexts.map { "/Library/Extensions/\($0).kext" }).filter(ours)
}

// set once NVBring.kext is off in OpenCore, so a rerun doesn't need the EFI for it
func ocEntryOff() -> Bool {
    exists("\(supportDir)/oc-off")
}

// before the move they are in the old support folder
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

// what Privacy & Security records when the user presses Allow
func kextApproved() -> Bool {
    // nothing is recorded when consent is off, as it is on some Hackintoshes
    if capture("/usr/sbin/spctl", ["kext-consent", "status"]).text.contains("DISABLED") {
        return true
    }
    let query = "select count(*) from kext_policy where bundle_id = '\(kextID)' and allowed = 1"
    let out = capture(sqlite, [policyDB, query])
    // an unreadable database isn't a refusal and would block a move for good; the collection check still runs
    return out.status != 0 || Int(out.text.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0 > 0
}

private func inCollection() -> Bool {
    capture(kmutil, ["inspect", "-V", "release", "-B", "/Library/KernelCollections/AuxiliaryKernelExtensions.kc"]).text
        .contains(kextID)
}

// nothing old goes before the new kext is approved and in the collection, and NVBring goes off in OpenCore before
// the old kexts go, so a restart in between loads one driver, never both
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
            // a rerun or this install's own OpenCore edit has saved the untouched file already
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

    for path in oldBundles() {
        try steps.remove(path)
    }
    try steps.remove(oldCtl)
    try steps.remove("/Library/GPUBundles/NVMTLDriver.bundle.new")
    if let efi {
        try steps.remove("\(efi)/EFI/OC/Kexts/NVBring.kext")
    }
    say("  Old driver removed, rebuilding the kext collection")
    try rebuildCollection(steps, say: say)

    if exists(oldSupportDir) {
        // this package's firmware and the install options are in the new folder already
        try steps.remove(oldSupportDir)
    }
}
