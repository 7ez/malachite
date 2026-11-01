import CryptoKit
import Foundation

// the raw public half of the key publish.sh signs with
let publicKey = "P4qTRMSxG50Xj5LdzhAk4pr7UDF4c0h+Hzgzzy4OxDQ="
let updateBase = URL(string: "https://nv.aochi.uk/updates/")!
let logsBase = URL(string: "https://nv.aochi.uk/logs/")!
let supportDir = "/Library/Application Support/Malachite"
let scaledFile = "\(supportDir)/scaled.txt"

struct Manifest: Decodable {
    let version: String
    let build: Int
    let url: URL
    let sha256: String
    let size: Int
    let notes: [String]
    let minMacos: String
}

enum UpdateError: LocalizedError {
    case badSignature
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .badSignature: "Bad update signature, ignored."
        case .failed(let why): why
        }
    }
}

func verify(_ manifest: Data, signature: Data, key: String = publicKey) -> Bool {
    guard let raw = Data(base64Encoded: key), let pub = try? Curve25519.Signing.PublicKey(rawRepresentation: raw) else {
        return false
    }
    return pub.isValidSignature(signature, for: manifest)
}

func fetchManifest() async throws -> Manifest {
    let session = URLSession(configuration: .ephemeral)
    // not manifest.json: apps from before Malachite read that, and can't install this layout
    async let json = session.data(from: updateBase.appending(path: "malachite.json"))
    async let sig = session.data(from: updateBase.appending(path: "malachite.sig"))
    let (body, sigData) = try await (json.0, sig.0)
    guard verify(body, signature: sigData) else { throw UpdateError.badSignature }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(Manifest.self, from: body)
}

func sha256(of file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var hash = SHA256()
    while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
        hash.update(data: chunk)
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
