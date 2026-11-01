import Foundation

// a test mode for the signature check: main --verify manifest.json manifest.sig [base64 public key]
let args = CommandLine.arguments
if args.count >= 4, args[1] == "--verify" {
    let manifest = FileManager.default.contents(atPath: args[2]) ?? Data()
    let sig = FileManager.default.contents(atPath: args[3]) ?? Data()
    let good = args.count > 4 ? verify(manifest, signature: sig, key: args[4]) : verify(manifest, signature: sig)
    print(good ? "good" : "bad")
    exit(good ? 0 : 1)
}
if let code = cli(args) {
    exit(code)
}
NVApp.main()
