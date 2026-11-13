import AppKit
import IOKit
import SwiftUI

let scaledLimit = 8
private let framebufferBytes = 34 << 20
private let kDisplayModeNativeFlag: UInt32 = 0x0200_0000

struct Size: Hashable {
    let width, height: Int

    init(_ width: Int, _ height: Int) {
        self.width = width
        self.height = height
    }

    init?(_ text: String) {
        let parts = text.split(separator: "x").compactMap { Int($0) }
        guard parts.count == 2 else { return nil }
        self.init(parts[0], parts[1])
    }

    var text: String { "\(width)x\(height)" }

    // rows padded to 256 bytes, as the driver lays them out
    var fits: Bool { ((width * 4 + 255) & ~255) * height <= framebufferBytes }

    func suits(_ native: Size) -> Bool {
        width > native.width && width <= native.width * 2 && width * native.height == height * native.width && fits
    }
}

private func gcd(_ a: Int, _ b: Int) -> Int {
    b == 0 ? a : gcd(b, a % b)
}

// sizes of exactly the panel's shape
func presets(for native: Size) -> [Size] {
    let unit = gcd(native.width, native.height)
    let step = Size(native.width / unit, native.height / unit)
    let sizes = [1.25, 4.0 / 3, 1.5, 2].map { factor -> Size in
        let k = (Double(native.width) * factor / Double(step.width)).rounded()
        return Size(Int(k) * step.width, Int(k) * step.height)
    }
    return sizes.filter { $0.suits(native) && $0.width % 2 == 0 && $0.height % 2 == 0 }
}

func readScaled() -> [Size] {
    ((try? String(contentsOfFile: scaledFile, encoding: .utf8)) ?? "").split(separator: "\n").compactMap { Size(String($0)) }
}

// the running driver's modes change at once; macOS sees a hotplug
func sendScaled(_ sizes: [Size]) -> Bool {
    let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("MalachiteGpu"))
    guard service != 0 else { return false }
    defer { IOObjectRelease(service) }
    var connection: io_connect_t = 0
    guard IOServiceOpen(service, mach_task_self_, 2, &connection) == KERN_SUCCESS else { return false }
    defer { IOServiceClose(connection) }
    let values = sizes.flatMap { [UInt64($0.width), UInt64($0.height)] }
    return IOConnectCallScalarMethod(connection, 0, values, UInt32(values.count), nil, nil) == KERN_SUCCESS
}

struct Monitor: Identifiable {
    let id: CGDirectDisplayID
    let name: String
    let native: Size
    let showing: Size
}

private func displayModes(_ id: CGDirectDisplayID) -> [CGDisplayMode] {
    (CGDisplayCopyAllDisplayModes(id, nil) as? [CGDisplayMode]) ?? []
}

private func connectedMonitors() -> [Monitor] {
    NSScreen.screens.compactMap { screen in
        guard let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber,
              let current = CGDisplayCopyDisplayMode(number.uint32Value) else { return nil }
        let id = number.uint32Value
        let native = displayModes(id).first { $0.ioFlags & kDisplayModeNativeFlag != 0 }
        return Monitor(id: id, name: screen.localizedName,
                       native: Size(native?.pixelWidth ?? current.pixelWidth, native?.pixelHeight ?? current.pixelHeight),
                       showing: Size(current.pixelWidth, current.pixelHeight))
    }
}

@MainActor
final class Resolutions: ObservableObject {
    @Published var sizes = readScaled()
    @Published var monitors: [Monitor] = []
    @Published var custom = ""
    @Published var note = ""

    func refresh() {
        monitors = connectedMonitors()
    }

    func set(_ size: Size, on: Bool) {
        var next = sizes.filter { $0 != size }
        if on {
            next.append(size)
        }
        Task { await apply(next) }
    }

    func addCustom() {
        guard let size = Size(custom), monitors.contains(where: { size.suits($0.native) }) else {
            note = "A size needs a monitor's shape, bigger than it, up to twice its size and at most 3840x2160"
            return
        }
        custom = ""
        set(size, on: true)
    }

    // the driver only lists rates the display engine can do
    func topRate(_ size: Size, on monitor: Monitor) -> Int? {
        displayModes(monitor.id).filter { $0.pixelWidth == size.width && $0.width == size.width }
            .map { Int($0.refreshRate.rounded()) }.max()
    }

    private func apply(_ next: [Size]) async {
        guard next.count <= scaledLimit else {
            note = "\(scaledLimit) at most"
            return
        }
        let text = next.map(\.text).joined(separator: "\n") + "\n"
        var saved = (try? text.write(toFile: scaledFile, atomically: false, encoding: .utf8)) != nil
        if !saved {
            // older apps' installs left the file owned by root
            let file = quote(scaledFile)
            saved = await admin("printf '%s' \(quote(text)) > \(file) && chown root:admin \(file) && chmod 664 \(file)").status == 0
        }
        guard saved else {
            note = "Couldn't save the list"
            return
        }
        sizes = next
        note = sendScaled(next) ? "" : "Saved. Update the driver or restart to use it"
        for _ in 0 ..< 10 {
            try? await Task.sleep(for: .milliseconds(100))
            refresh()
        }
    }

    // a just-added size shows up a moment after the driver is told
    func use(_ size: Size, on monitor: Monitor) {
        Task {
            let rate = CGDisplayCopyDisplayMode(monitor.id)?.refreshRate ?? 0
            for _ in 0 ..< 30 {
                let matching = displayModes(monitor.id).filter { $0.pixelWidth == size.width && $0.width == size.width && $0.pixelHeight == size.height }
                if let mode = matching.min(by: { abs($0.refreshRate - rate) < abs($1.refreshRate - rate) }) {
                    var config: CGDisplayConfigRef?
                    CGBeginDisplayConfiguration(&config)
                    CGConfigureDisplayWithDisplayMode(config, monitor.id, mode, nil)
                    CGCompleteDisplayConfiguration(config, .permanently)
                    try? await Task.sleep(for: .milliseconds(300))
                    refresh()
                    return
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
            note = "macOS didn't list \(size.text)"
        }
    }
}

struct ResolutionsView: View {
    @StateObject private var model = Resolutions()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Resolutions")
                .font(.title3.bold())
            Text("Render more pixels than the monitor has and scale them down. Sharper, but slower.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(model.monitors) { monitor in
                monitorView(monitor)
            }
            HStack {
                TextField("Custom size, like 2560x1440", text: $model.custom)
                Button("Add") {
                    model.addCustom()
                }
            }
            if !model.note.isEmpty {
                Text(model.note)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Done") {
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(width: 460)
        .onAppear(perform: model.refresh)
    }

    private func monitorView(_ monitor: Monitor) -> some View {
        let listed = model.sizes.filter { $0.suits(monitor.native) }
        let offered = presets(for: monitor.native) + listed.filter { !presets(for: monitor.native).contains($0) }
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(monitor.name)
                    .font(.headline)
                Text(monitor.native.text)
                    .foregroundStyle(.secondary)
                Spacer()
                if monitor.showing != monitor.native {
                    Button("Back to native") {
                        model.use(monitor.native, on: monitor)
                    }
                }
            }
            ForEach(offered, id: \.self) { size in
                row(size, on: monitor)
            }
        }
    }

    private func row(_ size: Size, on monitor: Monitor) -> some View {
        let enabled = model.sizes.contains(size)
        let shown = monitor.showing == size
        return HStack {
            Toggle(isOn: Binding(get: { enabled }, set: { model.set(size, on: $0) })) {
                Text(size.text)
                    .monospacedDigit()
            }
            .disabled(shown)
            Text(String(format: "%.2f×", Double(size.width) / Double(monitor.native.width)))
                .foregroundStyle(.secondary)
            if enabled, let rate = model.topRate(size, on: monitor) {
                Text("up to \(rate) Hz")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if shown {
                Text("In use")
                    .foregroundStyle(.secondary)
            } else {
                Button("Use") {
                    if !enabled {
                        model.set(size, on: true)
                    }
                    model.use(size, on: monitor)
                }
            }
        }
    }
}
