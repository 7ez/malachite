import Foundation

typealias Say = (String) -> Void
typealias Ask = (String) -> Bool

struct Failure: Error {
    let message: String
}

let plistBuddy = "/usr/libexec/PlistBuddy"
private let diskutil = "/usr/sbin/diskutil"

func hasConfig(_ efi: String) -> Bool {
    FileManager.default.fileExists(atPath: "\(efi)/EFI/OC/config.plist")
}

private func mountPoint(_ part: String) -> String? {
    for line in capture(diskutil, ["info", part]).text.split(separator: "\n") {
        let field = line.drop { $0 == " " }
        if field.hasPrefix("Mount Point:") {
            let mp = field.dropFirst("Mount Point:".count).drop { $0 == " " }
            return mp.isEmpty ? nil : String(mp)
        }
    }
    return nil
}

// the GUID of the EFI partition OpenCore booted from, which it names in its boot-path variable (only read here)
private func bootPartition() -> String? {
    let text = capture("/usr/sbin/nvram", ["4D1FDA02-38C7-4A6A-9CC6-4BCCA8B30102:boot-path"]).text
    guard let hd = text.range(of: "HD(") else { return nil }
    let parts = text[hd.upperBound...].split(separator: ",", maxSplits: 3)
    guard parts.count >= 3, parts[1] == "GPT" else { return nil }
    return String(parts[2])
}

// a mounted EFI that holds OpenCore, and the partition to unmount when done if it was mounted for this
struct EFI {
    let path: String
    var mounted: String?

    // a FAT volume left mounted is marked dirty if the Mac is shut down before it is let go
    func release() {
        if let mounted {
            _ = capture(diskutil, ["unmount", mounted])
        }
    }
}

// the mounted EFI that holds OpenCore: the one given, then the one OpenCore booted from, then the only one of the
// EFI partitions that does. The partitions mounted only to look are unmounted again.
func findEFI(_ given: String?, say: Say) -> EFI? {
    if let given, !given.isEmpty, hasConfig(given) {
        return EFI(path: given)
    }
    if let guid = bootPartition() {
        let mounted = mountPoint(guid) == nil && capture(diskutil, ["mount", guid]).status == 0
        if let mp = mountPoint(guid), hasConfig(mp) {
            say("EFI: \(mp)")
            return EFI(path: mp, mounted: mounted ? guid : nil)
        }
        if mounted {
            _ = capture(diskutil, ["unmount", guid])
        }
    }
    var mounted: [String] = []
    for line in capture(diskutil, ["list"]).text.split(separator: "\n") {
        let field = line.split(whereSeparator: \.isWhitespace)
        guard field.count > 1, field[1] == "EFI", let part = field.last.map(String.init) else { continue }
        if mountPoint(part) == nil, capture(diskutil, ["mount", part]).status == 0 {
            mounted.append(part)
        }
    }
    let volumes = (try? FileManager.default.contentsOfDirectory(atPath: "/Volumes").sorted()) ?? []
    let found = volumes.map { "/Volumes/\($0)" }.filter(hasConfig)
    let path = found.count == 1 ? found[0] : nil
    if let path {
        say("EFI: \(path)")
    } else if found.isEmpty {
        say("No OpenCore EFI found")
    } else {
        say("More than one OpenCore EFI, pick one with --efi /Volumes/<name>")
    }
    for part in mounted where mountPoint(part) != path {
        _ = capture(diskutil, ["unmount", part])
    }
    guard let path else { return nil }
    return EFI(path: path, mounted: mounted.first { mountPoint($0) == path })
}

func plistValue(_ key: String, _ cfg: String) -> String? {
    let out = capture(plistBuddy, ["-c", "Print \(key)", cfg])
    return out.status == 0 ? out.text.trimmingCharacters(in: .newlines) : nil
}

// how many Kernel > Add entries there are, and which of them are NVBring.kext
func kextEntries(_ cfg: String) -> (count: Int, nvbring: [Int]) {
    var nvbring: [Int] = []
    var i = 0
    while let path = plistValue(":Kernel:Add:\(i):BundlePath", cfg) {
        if path == "NVBring.kext" {
            nvbring.append(i)
        }
        i += 1
    }
    return (i, nvbring)
}
