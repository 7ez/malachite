import Foundation
import ServiceManagement

// the app's own path, which the GUI runs as root
let selfPath = Bundle.main.executablePath ?? CommandLine.arguments[0]

private func say(_ line: String) {
    print(line)
    fflush(stdout)
}

private func ask(_ prompt: String) -> Bool {
    print(prompt, terminator: "")
    fflush(stdout)
    guard let answer = readLine() else {
        say("\nNo answer, stopping.")
        return false
    }
    return answer == "y" || answer == "Y"
}

private func usage(_ text: String) -> Int32 {
    say(text)
    return 2
}

private func installMode(_ args: [String]) -> Int32 {
    var o = InstallOptions(package: defaultPackage())
    var i = 0
    while i < args.count {
        let arg = args[i]
        i += 1
        switch arg {
        case "--package":
            guard i < args.count else { return usage("--package needs a folder") }
            o.package = args[i]
            i += 1
        case "--efi":
            o.editEFI = true
            if i < args.count, !args[i].hasPrefix("--") {
                o.efiPath = args[i]
                i += 1
            }
        case "--full-bar": o.fullBar = true
        case "--yes": o.yes = true
        case "--dry-run": o.dryRun = true
        case "--variable-write-off": o.variableWriteOff = true
        default:
            // the rest of an EFI path with a space in it, typed without quotes (/Volumes/NO NAME)
            guard let path = o.efiPath else { return usage("unknown option \(arg)") }
            o.efiPath = "\(path) \(arg)"
        }
    }
    switch install(o, say: say, ask: ask) {
    case .done: return 0
    case .failed: return 1
    // not an error: the old driver is still in place, and the app tells the two apart
    case .stopped: return stoppedStatus
    }
}

private func uninstallMode(_ args: [String]) -> Int32 {
    var o = UninstallOptions()
    var i = 0
    while i < args.count {
        let arg = args[i]
        i += 1
        switch arg {
        case "--efi":
            o.editEFI = true
            if i < args.count, !args[i].hasPrefix("--") {
                o.efiPath = args[i]
                i += 1
            }
        case "--dry-run": o.dryRun = true
        default:
            guard let path = o.efiPath else { return usage("unknown option \(arg)") }
            o.efiPath = "\(path) \(arg)"
        }
    }
    return uninstall(o, say: say) ? 0 : 1
}

// --root-patch puts the Metal driver back on the system volume (for browsers) after a macOS update or removal;
// --remove-root-patch removes it, and later installs leave it off
private func rootPatchMode(_ flag: String, _ args: [String]) -> Int32 {
    guard requireRoot(flag, say: say) else { return 1 }
    var dry = false
    for arg in args {
        guard arg == "--dry-run" else { return usage("unknown option \(arg)") }
        dry = true
    }
    let steps = Steps(dry: dry, say: say)
    do {
        let changed = flag == "--root-patch" ? try patchRoot(steps: steps, say: say) : try unpatchRoot(steps: steps, say: say)
        if changed {
            say("Restart to finish.")
        }
        return 0
    } catch {
        say((error as? Failure)?.message ?? error.localizedDescription)
        return 1
    }
}

private func collectMode(_ dir: String) -> Int32 {
    do {
        _ = try collectLogs(into: dir, say: say)
        return 0
    } catch {
        say((error as? Failure)?.message ?? error.localizedDescription)
        return 1
    }
}

// collects, uploads and prints the ID on a line of its own
private func sendMode() async -> Int32 {
    let dir = FileManager.default.temporaryDirectory.appending(path: "nvlogs-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    do {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let archive = try collectLogs(into: dir.path, say: say)
        say("Uploading")
        let id = try await uploadLogs(URL(fileURLWithPath: archive))
        say("Uploaded. Log ID:\n\n\(id)")
        return 0
    } catch {
        say((error as? Failure)?.message ?? error.localizedDescription)
        return 1
    }
}

private func scaledMode(_ args: [String]) -> Int32 {
    let sizes = args.compactMap(Size.init)
    guard sizes.count == args.count, sizes.count <= scaledLimit else {
        return usage("usage: --scaled [2560x1440 ...], \(scaledLimit) at most, none to clear them")
    }
    let text = sizes.map { "\($0.text)\n" }.joined()
    guard (try? text.write(toFile: scaledFile, atomically: false, encoding: .utf8)) != nil else {
        say("Couldn't write \(scaledFile)")
        return 1
    }
    guard sendScaled(sizes) else {
        say("Saved, but the driver didn't take them: update it, or restart")
        return 1
    }
    return 0
}

// for recovery over SSH or with a dark screen; nil to open the window
func cli(_ args: [String]) -> Int32? {
    guard args.count > 1 else { return nil }
    setvbuf(stdout, nil, _IOLBF, 0)
    let rest = Array(args.dropFirst(2))
    switch args[1] {
    case "--install": return installMode(rest)
    case "--uninstall": return uninstallMode(rest)
    case "--root-patch", "--remove-root-patch": return rootPatchMode(args[1], rest)
    case "--check": return runChecks(say: say) ? 0 : 1
    // the installing app is often a copy in Downloads, and a login item must be registered by the app it launches
    case "--login-item": return (try? SMAppService.mainApp.register()) == nil ? 1 : 0
    case "--collect-logs": return collectMode(rest.first ?? ".")
    case "--scaled": return scaledMode(rest)
    case "--send-logs":
        guard requireRoot("--send-logs", say: say) else { return 1 }
        Task {
            exit(await sendMode())
        }
        dispatchMain()
    default: return args[1].hasPrefix("--") ? usage("unknown mode \(args[1])") : nil
    }
}
