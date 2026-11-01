import AppKit
import Foundation
import ServiceManagement
import UserNotifications

struct Report {
    var title: String
    var text: String
    var logID: String?
    var logFile: URL?
    var canRestart = false
}

struct Installed {
    let version: String
    let build: Int
}

@MainActor
final class Model: ObservableObject {
    @Published var installed: Installed?
    @Published var loaded = false
    @Published var update: Manifest?
    @Published var status = ""
    @Published var busy: String?
    @Published var report = Report(title: "", text: "")
    // a first install's choices
    @Published var useEFI = true
    @Published var fullBar = false
    // an install from before the name Malachite, which the next install moves over
    @Published var legacy = false
    @Published var startAtLogin = UserDefaults.standard.object(forKey: "startAtLogin") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(startAtLogin, forKey: "startAtLogin")
            applyLoginItem()
        }
    }

    init() {
        refresh()
        Task {
            while true {
                // a login can get here before the network is up
                let delay = await check() ? 6 * 3600 : 60
                try? await Task.sleep(for: .seconds(delay))
            }
        }
    }

    func refresh() {
        let version = fields(of: "VERSION.txt")
        installed = version["version"].map { Installed(version: $0, build: Int(version["build"] ?? "") ?? 0) }
        legacy = oldDriverInstalled()
        Task {
            loaded = await !run("/usr/sbin/kextstat", ["-l", "-b", "pl.aochi.malachite"]).text.isEmpty
        }
        applyLoginItem()
    }

    @discardableResult
    func check() async -> Bool {
        guard busy == nil else { return false }
        busy = "Checking"
        defer { busy = nil }
        do {
            let m = try await fetchManifest()
            let needs = ProcessInfo.processInfo.operatingSystemVersion
            let have = "\(needs.majorVersion).\(needs.minorVersion)"
            if installed == nil && have.compare(m.minMacos, options: .numeric) != .orderedAscending {
                update = m
                status = "Latest: \(m.version)"
            } else if m.build <= installed?.build ?? 0 {
                update = nil
                status = "Up to date"
            } else if have.compare(m.minMacos, options: .numeric) == .orderedAscending {
                update = nil
                status = "\(m.version) needs macOS \(m.minMacos)"
            } else {
                update = m
                status = "Update: \(m.version)"
            }
            if let update, legacy {
                status = exists("\(supportDir)/migrating") ? "Finish the move to Malachite \(update.version)"
                    : "Move to Malachite \(update.version)"
            }
            // not waited for: the first time, it sits there until the user answers macOS's permission prompt
            if let update, installed != nil || legacy {
                Task { await announce(update) }
            }
            return true
        } catch {
            update = nil
            status = error.localizedDescription
            return false
        }
    }

    // one notification per version; if it can't be shown, the menu bar item still says an update is there
    private func announce(_ m: Manifest) async {
        guard UserDefaults.standard.integer(forKey: "notifiedBuild") < m.build else { return }
        let center = UNUserNotificationCenter.current()
        guard (try? await center.requestAuthorization(options: [.alert])) == true else { return }
        let content = UNMutableNotificationContent()
        content.title = "Malachite \(m.version) available"
        let request = UNNotificationRequest(identifier: "update-\(m.build)", content: content, trigger: nil)
        if (try? await center.add(request)) != nil {
            UserDefaults.standard.set(m.build, forKey: "notifiedBuild")
        }
    }

    // from /Applications only: a copy run from Downloads would register a path that goes away
    func applyLoginItem() {
        guard Bundle.main.bundlePath.hasPrefix("/Applications/") else { return }
        let service = SMAppService.mainApp
        if startAtLogin, service.status != .enabled {
            try? service.register()
        } else if !startAtLogin, service.status == .enabled {
            try? service.unregister()
        }
    }

    func start(_ title: String, _ work: @escaping @MainActor () async -> Report) {
        guard busy == nil else { return }
        busy = title
        report = Report(title: title, text: "")
        Task {
            report = await work()
            busy = nil
            refresh()
        }
    }

    func install() {
        guard let m = update, busy == nil else { return }
        var args = installArgs()
        // the installer has no terminal here to ask on, so the app asks and says so
        if !args.contains("--efi"), isAMD(), installOptions()["variable-write-off"] != "1" {
            guard confirm("Is DisableVariableWrite on?",
                          "On AMD, a kernel panic can write to NVRAM and leave the board unable to start. Set Booter > Quirks > DisableVariableWrite to true in config.plist and restart first.",
                          yes: "It's on") else { return }
            args.append("--variable-write-off")
        }
        start("Installing \(m.version)") { await self.install(m, args) }
    }

    private func confirm(_ title: String, _ text: String, yes: String) -> Bool {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = text
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: yes)
        return alert.runModal() == .alertSecondButtonReturn
    }

    private func install(_ m: Manifest, _ args: [String]) async -> Report {
        let title = "Installing \(m.version)"
        let work = FileManager.default.temporaryDirectory.appending(path: "nvupdate-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: work) }
        var log = ""
        func step(_ line: String) {
            log += line + "\n"
            report.text = log
        }
        do {
            guard m.url.scheme == "https" else { throw UpdateError.failed("Update URL isn't HTTPS.") }
            step("Downloading \(m.url.lastPathComponent) (\(m.size >> 20) MB)")
            let watch = Download()
            let progress = Task { @MainActor in
                while !Task.isCancelled {
                    report.text = log + "  \(watch.received >> 20) of \(m.size >> 20) MB\n"
                    try? await Task.sleep(for: .milliseconds(250))
                }
            }
            let (file, _) = try await URLSession(configuration: .ephemeral).download(from: m.url, delegate: watch)
            progress.cancel()
            step("Checking download")
            let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize
            guard size == m.size, try sha256(of: file) == m.sha256 else {
                throw UpdateError.failed("Download is corrupt, not installed.")
            }
            step("Unpacking")
            try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
            let unzip = await run("/usr/bin/ditto", ["-x", "-k", file.path, work.path])
            guard unzip.status == 0 else { throw UpdateError.failed("Unpacking failed:\n\(unzip.text)") }
            let dirs = try FileManager.default.contentsOfDirectory(at: work, includingPropertiesForKeys: nil)
            guard let pkg = dirs.first(where: { payloadMissing($0.path) == nil }) else {
                throw UpdateError.failed("No driver files in the package.")
            }
            step("Running the installer\n")
            let head = log
            // the package is only data: the one program that runs is this app's own
            let command = [quote(selfPath), "--install", "--package", quote(pkg.path), "--yes"] + args.map(quote)
            let out = await admin(command.joined(separator: " ")) { text in
                self.report.text = head + text
            }
            let ok = out.status == 0
            if ok, startAtLogin {
                _ = await run("/Applications/Malachite.app/Contents/MacOS/Malachite", ["--login-item"])
            }
            let end = ok ? "\nRestart to load the driver.\n"
                : out.status == stoppedStatus ? "\nNot finished. Your driver still works.\n" : "\nInstall failed.\n"
            return Report(title: title, text: head + out.text + end, canRestart: ok)
        } catch {
            return Report(title: title, text: log + error.localizedDescription + "\n")
        }
    }

    // a first install's choices. An update leaves OpenCore alone, except to finish moving an older install over:
    // that needs the EFI the last install used
    private func installArgs() -> [String] {
        var args: [String] = []
        if legacy {
            if let efi = installOptions()["efi"], !efi.isEmpty {
                args += ["--efi", efi]
            }
        } else if installed == nil {
            if useEFI {
                args.append("--efi")
            }
            if fullBar {
                args.append("--full-bar")
            }
        }
        return args
    }

    func uninstall() {
        guard confirm("Uninstall Malachite?", "Removes the driver and this app.", yes: "Uninstall") else { return }
        start("Uninstalling") {
            var command = [quote(selfPath), "--uninstall"]
            // the EFI an older install edited, to turn NVBring.kext off there
            if self.legacy, let efi = installOptions()["efi"], !efi.isEmpty {
                command += ["--efi", quote(efi)]
            }
            let out = await admin(command.joined(separator: " ")) { text in
                self.report.text = text
            }
            let ok = out.status == 0
            return Report(title: "Uninstalling", text: out.text + (ok ? "" : "\nUninstall failed.\n"),
                          canRestart: ok)
        }
    }

    private func fields(of name: String) -> [String: String] {
        parseFields((try? String(contentsOfFile: "\(supportDir)/\(name)", encoding: .utf8)) ?? "")
    }

    func runChecks() {
        start("Checks") {
            Report(title: "Checks", text: await run(selfPath, ["--check"]).text)
        }
    }

    func sendLogs() {
        start("Send logs") {
            do {
                let file = try await self.collectLogs()
                do {
                    let id = try await uploadLogs(file)
                    return Report(title: "Send logs", text: "Uploaded. Log ID:\n\n\(id)\n", logID: id, logFile: file)
                } catch {
                    return Report(title: "Send logs", text: error.localizedDescription + "\n", logFile: file)
                }
            } catch {
                return Report(title: "Send logs", text: error.localizedDescription)
            }
        }
    }

    func saveLogs() {
        start("Save logs") {
            do {
                let file = try await self.collectLogs()
                let dest = try self.saveToDesktop(file)
                return Report(title: "Save logs", text: "Saved to \(dest.path)\n")
            } catch {
                return Report(title: "Save logs", text: error.localizedDescription)
            }
        }
    }

    func saveToDesktop(_ file: URL) throws -> URL {
        let desktop = try FileManager.default.url(for: .desktopDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let dest = desktop.appending(path: file.lastPathComponent)
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.copyItem(at: file, to: dest)
        return dest
    }

    private func collectLogs() async throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: "nvlogs-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let out = await admin("\(quote(selfPath)) --collect-logs \(quote(dir.path))")
        guard out.status == 0, let line = out.text.split(separator: "\n").last(where: { $0.hasPrefix("Send back: ") }) else {
            throw UpdateError.failed("Collecting logs failed:\n\(out.text)")
        }
        return URL(fileURLWithPath: String(line.dropFirst("Send back: ".count)))
    }

    func restart() {
        Task {
            _ = await run("/usr/bin/osascript", ["-e", "tell application \"System Events\" to restart"])
        }
    }
}
