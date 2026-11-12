import Foundation

private let logLimit = 10 << 20
private let traceLimit = 8_000_000

// a wedged GPU can leave system_profiler or ioreg waiting for good: a report with a gap beats no report
private func logText(_ path: String, _ args: [String]) -> String {
    capture(path, args, timeout: 60).text
}

// grep for any of the words (every line with none given), or without them, then head
private func keep(_ text: String, containing words: [String] = [], skipping: Bool = false, first: Int? = nil) -> String {
    // blank lines stay, as grep -v leaves them
    let all = text.split(separator: "\n", omittingEmptySubsequences: false).dropLast(text.hasSuffix("\n") ? 1 : 0)
    let kept = all.filter { line in
        (words.isEmpty || words.contains { line.contains($0) }) != skipping
    }
    return kept.prefix(first ?? kept.count).map { "\($0)\n" }.joined()
}

private func kernelLog(_ predicate: String, to file: String, last: String = "boot") {
    FileManager.default.createFile(atPath: file, contents: nil)
    if let handle = FileHandle(forWritingAtPath: file) {
        // a busy system's log takes minutes to search: what it found by then is kept
        capture("/usr/bin/log", ["show", "--last", last, "--style", "compact", "--predicate", predicate], timeout: 180,
                log: handle)
        try? handle.close()
    }
}

// A file per process, each its first traceLimit bytes: one game's trace in a shared file pushes out the one before it.
private func splitTraces(_ file: String, into folder: String) {
    guard let input = FileHandle(forReadingAtPath: file) else { return }
    defer { try? input.close() }
    var outputs: [String: FileHandle] = [:]
    var written: [String: Int] = [:]
    var rest = Data()
    while true {
        let chunk = input.readData(ofLength: 1 << 20)
        rest.append(chunk)
        guard let end = chunk.isEmpty ? rest.indices.last : rest.lastIndex(of: 10) else { break }
        for line in String(decoding: rest[...end], as: UTF8.self).split(separator: "\n") {
            // date, time, type, then "name[pid:thread] ..."
            let fields = line.split(separator: " ", maxSplits: 3)
            guard fields.count == 4, let bracket = fields[3].firstIndex(of: "[") else { continue }
            let name = String(fields[3][..<bracket]).replacingOccurrences(of: "/", with: "_")
            guard written[name, default: 0] < traceLimit else { continue }
            if outputs[name] == nil {
                let path = "\(folder)/trace-\(name).txt"
                FileManager.default.createFile(atPath: path, contents: nil)
                outputs[name] = FileHandle(forWritingAtPath: path)
            }
            let bytes = Data((line + "\n").utf8)
            outputs[name]?.write(bytes)
            written[name, default: 0] += bytes.count
        }
        if chunk.isEmpty {
            break
        }
        rest = Data(rest[(end + 1)...])
    }
    outputs.values.forEach { try? $0.close() }
}

// the last `limit` bytes of a file, as tail -c does
private func trim(_ file: String, to limit: Int) throws {
    let url = URL(fileURLWithPath: file)
    let handle = try FileHandle(forUpdating: url)
    defer { try? handle.close() }
    let size = Int(try handle.seekToEnd())
    guard size > limit else { return }
    try handle.seek(toOffset: UInt64(size - limit))
    let tail = try handle.readToEnd() ?? Data()
    try handle.truncate(atOffset: 0)
    try handle.seek(toOffset: 0)
    try handle.write(contentsOf: tail)
}

// the driver's own state and logs in one archive, no personal files; returns the archive's path
func collectLogs(into dir: String, say: Say) throws -> String {
    guard geteuid() == 0 else {
        throw Failure(message: "Run it with sudo: sudo \(quote(selfPath)) --collect-logs")
    }
    let fm = FileManager.default
    let dest = URL(fileURLWithPath: dir).standardizedFileURL.path
    guard fm.fileExists(atPath: dest) else { throw Failure(message: "\(dir) doesn't exist") }
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyyMMdd-HHmmss"
    let name = "malachite-logs-\(f.string(from: Date()))"
    let out = "\(dest)/\(name)"
    try fm.createDirectory(atPath: out, withIntermediateDirectories: true)

    var system = ""
    system += logText("/usr/bin/sw_vers", [])
    system += logText("/usr/bin/uname", ["-m"])
    system += logText("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"])
    system += logText("/usr/sbin/nvram", ["boot-args"])
    system += logText("/usr/bin/csrutil", ["status"])
    system += "\n"
    system += logText("/usr/bin/csrutil", ["authenticated-root", "status"])
    system += keep(logText("/usr/sbin/diskutil", ["info", "/"]), containing: ["Device Identifier", "Snapshot", "Sealed"])
    system += logText("/bin/ls", ["-ld", gpuBundle, systemBundle])
    system += "\(rootPatchFile):\n" + ((try? String(contentsOfFile: rootPatchFile, encoding: .utf8)) ?? "none\n")
    system += keep(logText("/usr/sbin/ioreg", ["-l", "-w0", "-r", "-c", "IOAccelerator"]), containing: ["MetalPluginName"])
    system += rootPatchSummary() + "\n\n"
    system += keep(logText("/usr/sbin/kextstat", []), containing: ["pl.aochi", "IOAccelerator", "IOGraphics"])
    system += "\n"
    system += keep(logText("/usr/bin/kmutil", ["showloaded", "--collection", "aux"]), first: 40)
    system += logText("/bin/ls", ["-la", "/Library/Extensions/Malachite.kext", "/Library/KernelCollections"])
    let framebuffer = logText("/usr/sbin/ioreg", ["-l", "-w0", "-r", "-c", "IOFramebuffer"])
    system += keep(framebuffer, containing: ["class", "IOFBMemorySize", "Malachite", "NVFB"], first: 20)
    system += "\n"
    system += keep(logText("/usr/sbin/system_profiler", ["SPDisplaysDataType"]), containing: ["Serial", "UUID"], skipping: true)
    let pci = logText("/usr/sbin/ioreg", ["-l", "-w0", "-r", "-c", "IOPCIDevice"])
    system += keep(pci, containing: ["\"vendor-id\"", "\"device-id\"", "\"model\"", "\"VRAM,totalMB\"", "\"IOName\"",
                                     "\"assigned-addresses\"", "\"reg\"", "\"ranges\"", "\"IODeviceMemory\""])
    system += keep(logText("/usr/sbin/ioreg", ["-l", "-w0"]), containing: ["acpi-address-spaces"])
    try Data(system.utf8).write(to: URL(fileURLWithPath: "\(out)/system.txt"))

    // everything the driver logged since it loaded, from its own buffer (macOS's logs keep little of it)
    // the old driver's sysctl has its old name
    var driverLog = capture("/usr/sbin/sysctl", ["-b", "kern.malachite_log"])
    if driverLog.data.isEmpty {
        driverLog = capture("/usr/sbin/sysctl", ["-b", "kern.nvrm_log"])
    }
    try driverLog.data.write(to: URL(fileURLWithPath: "\(out)/driver.txt"))
    // the driver's lines (Malachite:, gsp:, disp:, video:, fb:, accel:...), the old driver's (NVRM:, NVFB:...) and
    // WindowServer's view of the plugin
    let words = ["Malachite", "gsp:", "disp:", "video:", "fb:", "accel:", "queue:", "pipe:", "agdc:", "NVRM", "NVFB", "NVAccel",
                 "NVBring", "fwsec", "booter", "panic"]
    kernelLog("process == \"kernel\" AND (\(words.map { "eventMessage CONTAINS \"\($0)\"" }.joined(separator: " OR ")))",
              to: "\(out)/kernel.txt")
    kernelLog("(eventMessage CONTAINS \"Malachite\" OR eventMessage CONTAINS \"NVMTL\") "
              + "AND NOT eventMessage CONTAINS \"getMemoryAllocParameter\" "
              + "AND NOT eventMessage CONTAINS \"Malachite trace\"", to: "\(out)/metal.txt")
    kernelLog("eventMessage CONTAINS \"Malachite trace\"", to: "\(out)/traces.tmp")
    splitTraces("\(out)/traces.tmp", into: out)
    try? fm.removeItem(atPath: "\(out)/traces.tmp")
    // what the GPU spent its time on, which a report of "runs badly" needs
    for name in ["stats", "power"] {
        try capture("/usr/sbin/sysctl", ["-b", "kern.malachite_\(name)"]).data.write(to: URL(fileURLWithPath: "\(out)/\(name).txt"))
    }
    // why a kext of the driver's did or didn't load (the auxiliary collection's policy, missing approval, dependencies)
    kernelLog("eventMessage CONTAINS \"pl.aochi\" OR eventMessage CONTAINS \"AuxKC\" OR eventMessage CONTAINS \"NVFB\" "
              + "OR (process == \"kernelmanagerd\" AND eventMessage CONTAINS[c] \"aochi\")", to: "\(out)/kexts.txt")
    // a boot that failed is over by the time someone can send logs
    kernelLog("process == \"kernel\" AND eventMessage CONTAINS \"Malachite: \"", to: "\(out)/earlier-boots.txt", last: "3d")
    if fm.fileExists(atPath: kmutilLog) {
        try? fm.copyItem(atPath: kmutilLog, toPath: "\(out)/malachite-kmutil.log")
    }
    try trim("\(out)/metal.txt", to: 20_000_000)
    try trim("\(out)/earlier-boots.txt", to: 5_000_000)

    try fm.createDirectory(atPath: "\(out)/reports", withIntermediateDirectories: true)
    let home = ProcessInfo.processInfo.environment["HOME"] ?? ""
    let reports = [
        ("/Library/Logs/DiagnosticReports", { (n: String) in n.hasSuffix(".panic") }),
        ("/Library/Logs/DiagnosticReports", { (n: String) in n.hasPrefix("Kernel") }),
        ("\(home)/Library/Logs/DiagnosticReports", { (n: String) in n.hasPrefix("WindowServer") }),
        ("/Library/Logs/DiagnosticReports", { (n: String) in n.hasPrefix("WindowServer") }),
    ]
    for (folder, matches) in reports {
        for file in (try? fm.contentsOfDirectory(atPath: folder).sorted()) ?? [] where matches(file) && !file.hasPrefix(".") {
            let path = "\(folder)/\(file)"
            let attrs = try? fm.attributesOfItem(atPath: path)
            guard attrs?[.type] as? FileAttributeType == .typeRegular,
                  let modified = attrs?[.modificationDate] as? Date, modified.timeIntervalSinceNow > -3 * 86400 else { continue }
            let target = "\(out)/reports/\(file)"
            try? fm.removeItem(atPath: target)
            try? fm.copyItem(atPath: path, toPath: target)
        }
    }

    if fm.fileExists(atPath: "\(supportDir)/bin/mtldraw") {
        var check = ""
        _ = runChecks { check += $0 + "\n" }
        try Data(check.utf8).write(to: URL(fileURLWithPath: "\(out)/check.txt"))
    }

    let archive = "\(out).tar.gz"
    let tar = capture("/usr/bin/tar", ["czf", archive, "-C", dest, name])
    guard tar.status == 0 else { throw Failure(message: "tar failed:\n\(tar.text)") }
    try fm.removeItem(atPath: out)
    // made by root, and the app that asked for it has to read it
    try fm.setAttributes([.posixPermissions: 0o644], ofItemAtPath: archive)
    say("Send back: \(archive)")
    return archive
}

// uploads an archive under a random ID and returns the ID
func uploadLogs(_ file: URL) async throws -> String {
    let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size <= logLimit else {
        throw UpdateError.failed("Logs are \(size >> 20) MB, the limit is 10 MB. Save them and send the file.")
    }
    let id = String((0..<32).map { _ in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".randomElement()! })
    var req = URLRequest(url: logsBase.appending(path: "\(id).tar.gz"))
    req.httpMethod = "PUT"
    let (_, resp) = try await URLSession(configuration: .ephemeral).upload(for: req, fromFile: file)
    let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
    guard (200..<300).contains(code) else {
        throw UpdateError.failed("Upload failed (HTTP \(code)). Try again in a minute.")
    }
    return id
}
