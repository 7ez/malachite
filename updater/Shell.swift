import Foundation

struct Output {
    let status: Int32
    let data: Data

    var text: String {
        String(decoding: data, as: UTF8.self)
    }

    init(status: Int32, data: Data) {
        self.status = status
        self.data = data
    }

    init(status: Int32, text: String) {
        self.init(status: status, data: Data(text.utf8))
    }
}

func run(_ path: String, _ args: [String]) async -> Output {
    await withCheckedContinuation { done in
        DispatchQueue.global().async {
            done.resume(returning: capture(path, args))
        }
    }
}

// stdout and stderr together, or into `log`; killed after `timeout` seconds
@discardableResult
func capture(_ path: String, _ args: [String], cwd: String? = nil, timeout: Double? = nil, log: FileHandle? = nil) -> Output {
    let p = Process()
    let pipe = Pipe()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    p.currentDirectoryURL = cwd.map { URL(fileURLWithPath: $0) }
    p.standardOutput = log ?? pipe
    p.standardError = log ?? pipe
    do {
        try p.run()
    } catch {
        return Output(status: 127, text: "\(path): \(error.localizedDescription)")
    }
    if let timeout {
        let pid = p.processIdentifier
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout) {
            if p.isRunning {
                kill(pid, SIGKILL)
            }
        }
    }
    let data = log == nil ? pipe.fileHandleForReading.readDataToEndOfFile() : Data()
    p.waitUntilExit()
    return Output(status: p.terminationStatus, data: data)
}

func quote(_ s: String) -> String {
    "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

// do shell script drops a failing command's output, so the status is echoed and parsed back
func admin(_ command: String) async -> Output {
    let shell = "{ \(command); } 2>&1; echo nvstatus=$?"
    let literal = shell.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    let out = await run("/usr/bin/osascript", ["-e", "do shell script \"\(literal)\" with administrator privileges"])
    if out.status != 0 {
        let cancelled = out.text.contains("-128")
        return Output(status: 1, text: cancelled ? "Cancelled.\n" : out.text)
    }
    var lines = out.text.replacingOccurrences(of: "\r", with: "\n").split(separator: "\n", omittingEmptySubsequences: false)
    while lines.last == "" {
        lines.removeLast()
    }
    let status = Int32(lines.last?.dropFirst("nvstatus=".count) ?? "") ?? 1
    return Output(status: status, text: lines.dropLast().joined(separator: "\n") + "\n")
}

// output tailed from a file for a live log, as do shell script returns only when the command is done
func admin(_ command: String, live: @escaping @MainActor (String) -> Void) async -> Output {
    let log = FileManager.default.temporaryDirectory.appending(path: "nvadmin-\(UUID().uuidString).log")
    FileManager.default.createFile(atPath: log.path, contents: nil)
    defer { try? FileManager.default.removeItem(at: log) }
    let tail = Task { @MainActor in
        while !Task.isCancelled {
            if let data = FileManager.default.contents(atPath: log.path) {
                live(String(decoding: data, as: UTF8.self))
            }
            try? await Task.sleep(for: .milliseconds(300))
        }
    }
    let out = await admin("(\(command)) >> \(quote(log.path)) 2>&1")
    tail.cancel()
    let text = String(decoding: FileManager.default.contents(atPath: log.path) ?? Data(), as: UTF8.self)
    return Output(status: out.status, text: text + out.text.trimmingCharacters(in: .newlines))
}

final class Download: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private var task: URLSessionTask?

    func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
        self.task = task
    }

    var received: Int64 { task?.countOfBytesReceived ?? 0 }
}
