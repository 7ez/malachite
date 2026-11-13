import Foundation

// A second copy of the Metal driver, on the system volume, for the sandboxed GPU processes of browsers: they may map
// code from there but not from /Library/GPUBundles. As other root patchers do, it is copied onto the volume under the
// sealed snapshot macOS boots from, and a new snapshot of that is booted from then on. A macOS update boots a fresh
// sealed snapshot without it. The kext names this copy to Metal only when it is the same build as the one in
// /Library/GPUBundles.
let gpuBundle = "/Library/GPUBundles/MalachiteMetal.bundle"
let systemBundle = "/System/Library/Extensions/MalachiteMetal.bundle"
let rootPatchFile = "\(supportDir)/root-patch"
private let bless = "/usr/sbin/bless"
// left on the volume by OpenCore Legacy Patcher's own root patches
private let oclpMarker = "/System/Library/CoreServices/OpenCore-Legacy-Patcher.plist"

// on=0: turned off with --remove-root-patch. sealed=1: macOS's sealed snapshot was booted when the copy first went on,
// so going back to it loses nothing. patched: the booted snapshot when the last patch was made, which tells a patch
// waiting for a restart from one a macOS update took away. made: the snapshots bless made for the patches, by UUID
func rootPatchState() -> [String: String] {
    parseFields((try? String(contentsOfFile: rootPatchFile, encoding: .utf8)) ?? "")
}

func authenticatedRootOff() -> Bool {
    capture("/usr/bin/csrutil", ["authenticated-root", "status"]).text.contains("disabled")
}

// the booted snapshot, and the volume it was taken of
struct RootVolume {
    let device: String
    let snapshot: String
    let sealed: Bool
}

func rootVolume() -> RootVolume? {
    let out = capture("/usr/sbin/diskutil", ["info", "-plist", "/"])
    guard let info = try? PropertyListSerialization.propertyList(from: out.data, format: nil) as? [String: Any],
          info["APFSSnapshot"] as? Bool == true, let id = info["DeviceIdentifier"] as? String,
          let suffix = id.range(of: "s[0-9]+$", options: .regularExpression) else { return nil }
    return RootVolume(device: String(id[..<suffix.lowerBound]), snapshot: info["APFSSnapshotUUID"] as? String ?? "",
                      sealed: info["Sealed"] as? String == "Yes")
}

// name by UUID, of the volume's snapshots
private func snapshots(_ device: String) -> [String: String] {
    let out = capture("/usr/sbin/diskutil", ["apfs", "listSnapshots", "-plist", device], timeout: 60)
    let info = try? PropertyListSerialization.propertyList(from: out.data, format: nil) as? [String: Any]
    var names: [String: String] = [:]
    for s in info?["Snapshots"] as? [[String: Any]] ?? [] {
        if let uuid = s["SnapshotUUID"] as? String, let name = s["SnapshotName"] as? String {
            names[uuid] = name
        }
    }
    return names
}

// bless with a new snapshot to boot from; returns the UUIDs of the snapshots that made
private func blessSnapshot(_ root: RootVolume, mnt: String, steps: Steps) throws -> [String] {
    let before = snapshots(root.device)
    try steps.must(bless, ["--mount", mnt, "--bootefi", "--create-snapshot"])
    // the booted snapshot is always listed: an empty list failed, and every snapshot would pass for ours
    guard !before.isEmpty else { return [] }
    return snapshots(root.device).keys.filter { before[$0] == nil }.sorted()
}

// each patch leaves the snapshot before it behind: deletes those of ours that are neither booted nor to be booted next,
// never macOS's sealed one; returns the ones still there
private func dropOldSnapshots(_ root: RootVolume, made: [String], keep: Set<String>, steps: Steps, say: Say) -> [String] {
    let names = snapshots(root.device)
    guard !names.isEmpty else { return made }
    var kept: [String] = []
    var dropped = 0
    for uuid in made {
        guard let name = names[uuid] else { continue }
        if keep.contains(uuid) || !name.hasPrefix("com.apple.bless.") {
            kept.append(uuid)
            continue
        }
        let out = steps.tool("/usr/sbin/diskutil", ["apfs", "deleteSnapshot", root.device, "-uuid", uuid])
        if out.status == 0 {
            dropped += 1
        } else {
            kept.append(uuid)
            say("  ! Old system snapshot \(name) not deleted, it keeps using disk space until it is: "
                + out.text.trimmingCharacters(in: .whitespacesAndNewlines))
            say("    The next root patch tries again")
        }
    }
    if dropped > 0 {
        say("  \(dropped) old system snapshot\(dropped == 1 ? "" : "s") deleted")
    }
    return kept
}

private func madeList(_ state: [String: String]) -> [String] {
    (state["made"] ?? "").split(separator: ",").map(String.init)
}

private func sameBundle(_ a: String, _ b: String) -> Bool {
    capture("/usr/bin/diff", ["-rq", a, b]).status == 0
}

// the volume under the booted snapshot, mounted writable for `body`
private func onSystemVolume(_ root: RootVolume, steps: Steps, _ body: (String) throws -> Void) throws {
    var template = Array("/private/tmp/malachite-root.XXXXXX".utf8CString)
    guard steps.dry || mkdtemp(&template) != nil else {
        throw Failure(message: "Can't make a folder to mount the system volume on")
    }
    let mnt = template.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
    defer {
        if !steps.dry {
            rmdir(mnt)
        }
    }
    try steps.must("/sbin/mount", ["-o", "nobrowse", "-t", "apfs", "/dev/\(root.device)", mnt])
    defer { steps.tool("/usr/sbin/diskutil", ["unmount", mnt]) }
    try body(mnt)
}

// copies /Library/GPUBundles's Metal driver onto the system volume, unless it's there already; true when a restart
// is needed for it
@discardableResult
func patchRoot(steps: Steps, say: Say) throws -> Bool {
    guard authenticatedRootOff() else {
        throw Failure(message: "Authenticated root is on: turn it off in OpenCore (csr-active-config FF0F0000), then "
                      + "install again.")
    }
    guard let root = rootVolume() else { throw Failure(message: "/ isn't an APFS snapshot.") }
    guard steps.dry || exists(gpuBundle) else { throw Failure(message: "Driver not installed.") }
    let state = rootPatchState()
    var changed = false
    try onSystemVolume(root, steps: steps) { mnt in
        let dst = "\(mnt)\(systemBundle)"
        let current = sameBundle(gpuBundle, dst)
        if current, sameBundle(gpuBundle, systemBundle) || state["patched"] == root.snapshot {
            say("  System volume patched already")
            return
        }
        if !current {
            try steps.put(gpuBundle, dst)
            try steps.must("/bin/chmod", ["-R", "u=rwX,go=rX", dst])
        }
        let new = try blessSnapshot(root, mnt: mnt, steps: steps)
        let made = dropOldSnapshots(root, made: madeList(state) + new, keep: Set(new + [root.snapshot]), steps: steps,
                                    say: say)
        // the seal is ours to have broken if the booted snapshot has our copy, or our patch of it waits for a restart
        let ours = exists(systemBundle) || state["patched"] == root.snapshot
        let sealed = root.sealed || (ours && state["sealed"] == "1")
        try steps.write("on=1\nsealed=\(sealed ? 1 : 0)\npatched=\(root.snapshot)\nmade=\(made.joined(separator: ","))\n",
                        to: rootPatchFile)
        say("  Metal driver on the system volume, for browsers (new snapshot, takes effect at the next restart)")
        changed = true
    }
    return changed
}

// takes the system volume's copy away again: back to macOS's sealed snapshot when nothing else could be on the volume,
// otherwise a new snapshot without it, which keeps other patchers' changes
@discardableResult
func unpatchRoot(steps: Steps, say: Say) throws -> Bool {
    let state = rootPatchState()
    guard let root = rootVolume() else {
        say("/ isn't an APFS snapshot, nothing to remove")
        return false
    }
    var changed = false
    var made = madeList(state)
    try onSystemVolume(root, steps: steps) { mnt in
        let dst = "\(mnt)\(systemBundle)"
        guard steps.dry || exists(dst) || exists(systemBundle) else {
            say("The system volume has no Metal driver")
            return
        }
        try steps.remove(dst)
        try steps.remove("\(dst).new")
        let alone = state["sealed"] == "1" && !exists("\(mnt)\(oclpMarker)")
        var new: [String] = []
        if alone, steps.tool(bless, ["--mount", mnt, "--bootefi", "--last-sealed-snapshot"]).status == 0 {
            say("System volume back on macOS's sealed snapshot")
        } else {
            new = try blessSnapshot(root, mnt: mnt, steps: steps)
            say("Metal driver removed from the system volume (new snapshot, other patches kept)")
        }
        made = dropOldSnapshots(root, made: made + new, keep: Set(new + [root.snapshot]), steps: steps, say: say)
        changed = true
    }
    if exists(supportDir) {
        try steps.write("on=0\nsealed=\(state["sealed"] ?? "0")\nmade=\(made.joined(separator: ","))\n", to: rootPatchFile)
    }
    return changed
}

// wanted and possible, but not in the booted snapshot, and not just made and waiting for a restart: what a macOS
// update leaves
func rootPatchLost() -> Bool {
    let state = rootPatchState()
    guard state["on"] != "0", exists(gpuBundle), !exists(systemBundle), authenticatedRootOff(),
          let root = rootVolume() else { return false }
    return state["patched"] != root.snapshot
}

func rootPatchSummary() -> String {
    let state = rootPatchState()
    if state["on"] == "0" {
        return exists(systemBundle) ? "System volume patch off, taken away at the next restart"
            : "System volume patch off: browsers draw on the CPU"
    }
    if exists(systemBundle) {
        return sameBundle(gpuBundle, systemBundle) ? "System volume patched: browsers draw on the GPU"
            : "System volume has another build of the Metal driver, browsers draw on the CPU until it's patched again"
    }
    if !authenticatedRootOff() {
        return "System volume not patched, authenticated root is on: browsers draw on the CPU"
    }
    if let root = rootVolume(), state["patched"] == root.snapshot {
        return "System volume patched, takes effect at the next restart"
    }
    return "System volume not patched: browsers draw on the CPU until it is (Malachite --root-patch)"
}
