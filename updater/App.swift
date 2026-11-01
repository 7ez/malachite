import AppKit
import SwiftUI
import UserNotifications

let showWindow = Notification.Name("pl.aochi.malachite.showWindow")

@MainActor let appModel = Model()

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    private var window: NSWindow?

    // the window is made here and not as a SwiftUI scene, as a scene can't be told to stay closed at login
    func applicationDidFinishLaunching(_ note: Notification) {
        UNUserNotificationCenter.current().delegate = self
        NotificationCenter.default.addObserver(forName: showWindow, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { self.open() }
        }
        // the login item launch says so in its open event (which applicationWillFinishLaunching doesn't have yet)
        let event = NSAppleEventManager.shared().currentAppleEvent
        let how = event?.paramDescriptor(forKeyword: AEKeyword(keyAEPropData))?.enumCodeValue
        if how == AEKeyword(keyAELaunchedAsLogInItem) {
            // no Dock icon until the window is wanted
            NSApp.setActivationPolicy(.accessory)
        } else {
            open()
        }
    }

    func applicationShouldHandleReopen(_ app: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if !hasVisibleWindows {
            open()
        }
        return false
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse) async {
        await MainActor.run { open() }
    }

    private func open() {
        if window == nil {
            let w = NSWindow(contentViewController: NSHostingController(rootView: MainView(model: appModel)))
            w.title = "Malachite"
            w.isReleasedWhenClosed = false
            if !w.setFrameUsingName("main") {
                w.center()
            }
            w.setFrameAutosaveName("main")
            window = w
        }
        NSApp.setActivationPolicy(.regular)
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }
}

struct NVApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @StateObject private var state = appModel

    // keeps the app, and its update checks, running with the window closed
    var body: some Scene {
        MenuBarExtra("Malachite", systemImage: state.update == nil ? "display" : "arrow.down.circle") {
            MenuView(model: state)
        }
    }
}

struct MenuView: View {
    @ObservedObject var model: Model

    var body: some View {
        Text(model.status)
        Button("Open") {
            NotificationCenter.default.post(name: showWindow, object: nil)
        }
        Button("Quit") {
            NSApp.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}

struct MainView: View {
    @ObservedObject var model: Model
    @State private var showResolutions = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Malachite")
                .font(.title2.bold())
            Text(summary)
            Text(model.busy.map { "\($0)..." } ?? model.status)
                .foregroundStyle(.secondary)
            if let update = model.update, !update.notes.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("What's new")
                        .font(.headline)
                    ForEach(update.notes, id: \.self) { note in
                        Text("• \(note)")
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            if model.installed == nil, !model.legacy {
                Text("Needs an RTX 30 card, macOS 26 and OpenCore.")
                Toggle("Set BAR in OpenCore", isOn: $model.useEFI)
                Toggle("Full BAR (Resizable BAR on in BIOS)", isOn: $model.fullBar)
                    .disabled(!model.useEFI)
            }
            HStack {
                Toggle("Start at login", isOn: $model.startAtLogin)
                if model.installed != nil {
                    Spacer()
                    Button("Resolutions") {
                        showResolutions = true
                    }
                }
            }
            HStack {
                if let update = model.update {
                    Button(model.legacy ? "Move to Malachite" : "Install \(update.version)") {
                        model.install()
                    }
                    .keyboardShortcut(.defaultAction)
                }
                Button("Check for updates") {
                    Task { await model.check() }
                }
                if model.installed != nil {
                    Button("Run checks") {
                        model.runChecks()
                    }
                    Button("Send logs") {
                        model.sendLogs()
                    }
                    Button("Save logs") {
                        model.saveLogs()
                    }
                }
                if model.installed != nil || model.legacy {
                    Button("Uninstall") {
                        model.uninstall()
                    }
                }
            }
            .disabled(model.busy != nil)
            log
        }
        .padding()
        .frame(minWidth: 620, minHeight: 420, alignment: .topLeading)
        .sheet(isPresented: $showResolutions) {
            ResolutionsView()
        }
    }

    private var summary: String {
        guard let installed = model.installed else {
            return model.legacy ? "Old driver installed" : "Not installed"
        }
        return "Installed: \(installed.version), \(model.loaded ? "loaded" : "not loaded")"
    }

    private var log: some View {
        VStack(alignment: .leading) {
            ScrollViewReader { scroll in
                ScrollView {
                    Text(model.report.text)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Color.clear
                        .frame(height: 1)
                        .id("end")
                }
                // follows a running install
                .onChange(of: model.report.text) {
                    scroll.scrollTo("end", anchor: .bottom)
                }
            }
            .padding(8)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            HStack {
                if let id = model.report.logID {
                    Button("Copy ID") {
                        copy(id)
                    }
                }
                if let file = model.report.logFile {
                    Button("Save to Desktop") {
                        let dest = try? model.saveToDesktop(file)
                        model.report.text += dest.map { "\nSaved to \($0.path)\n" } ?? "\nSave failed.\n"
                    }
                }
                Button("Copy") {
                    copy(model.report.text)
                }
                .disabled(model.report.text.isEmpty)
                Spacer()
                if model.report.canRestart {
                    Button("Restart") {
                        model.restart()
                    }
                }
            }
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
