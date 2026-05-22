import Foundation

public struct InviteToken: Sendable {
    public let value: String

    public init(_ value: String) {
        self.value = value
    }
}

public struct Model: Sendable {
    public let id: String
    public let name: String
}

public struct Status: Sendable {
    public let connected: Bool
    public let peerCount: Int
}

public struct RequestId: Sendable {
    public let value: String
}

public enum Event: Sendable {
    case connecting
    case joined(nodeId: String)
    case modelsUpdated(models: [Model])
    case tokenDelta(requestId: String, delta: String)
    case completed(requestId: String)
    case failed(requestId: String, error: String)
    case disconnected(reason: String)
}

public struct ChatMessage: Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct ChatRequest: Sendable {
    public let model: String
    public let messages: [ChatMessage]

    public init(model: String, messages: [ChatMessage]) {
        self.model = model
        self.messages = messages
    }
}

public struct ResponsesRequest: Sendable {
    public let model: String
    public let input: String

    public init(model: String, input: String) {
        self.model = model
        self.input = input
    }
}

#if canImport(MeshLLMFFI)
public final class Node: @unchecked Sendable {
    private let handle: MeshNodeHandle

    public let inference: Inference
    public let models: Models
    public let serving: Serving

    public init(
        inviteToken: InviteToken,
        ownerKeypairBytesHex: String,
        cacheDir: String? = nil,
        runtimeDir: String? = nil,
        servingEnabled: Bool = false
    ) throws {
        let handle = try createNode(
            ownerKeypairBytesHex: ownerKeypairBytesHex,
            inviteToken: inviteToken.value,
            cacheDir: cacheDir,
            runtimeDir: runtimeDir,
            servingEnabled: servingEnabled
        )
        self.handle = handle
        self.inference = Inference(handle: handle)
        self.models = Models(handle: handle)
        self.serving = Serving(handle: handle)
    }

    public init(handle: MeshNodeHandle) {
        self.handle = handle
        self.inference = Inference(handle: handle)
        self.models = Models(handle: handle)
        self.serving = Serving(handle: handle)
    }

    public static func discoverPublicMeshes(
        _ query: PublicMeshQuery = PublicMeshQuery(
            model: nil,
            minVramGb: nil,
            region: nil,
            targetName: nil,
            relays: []
        )
    ) async throws -> [PublicMesh] {
        try await runBlocking {
            try MeshLLM.discoverPublicMeshes(query: query)
        }
    }

    public static func connectPublic(
        ownerKeypairBytesHex: String,
        query: PublicMeshQuery = PublicMeshQuery(
            model: nil,
            minVramGb: nil,
            region: nil,
            targetName: nil,
            relays: []
        )
    ) async throws -> Node {
        let handle = try await runBlocking {
            try createAutoNode(ownerKeypairBytesHex: ownerKeypairBytesHex, query: query)
        }
        return Node(handle: handle)
    }

    public func start() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.start()
        }
    }

    public func stop() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.stop()
        }
    }

    public func reconnect() async throws {
        let handle = self.handle
        try await runBlocking {
            try handle.reconnect()
        }
    }

    public func status() async -> Status {
        let handle = self.handle
        let status = await runBlocking {
            handle.status()
        }
        return Status(connected: status.connected, peerCount: Int(clamping: status.peerCount))
    }

    public final class Inference: @unchecked Sendable {
        private let handle: MeshNodeHandle

        fileprivate init(handle: MeshNodeHandle) {
            self.handle = handle
        }

        public func listModels() async throws -> [Model] {
            let handle = self.handle
            let models = try await runBlocking {
                try handle.inferenceListModels()
            }
            return models.map(Node.mapModel)
        }

        public func chat(_ request: ChatRequest) -> AsyncThrowingStream<Event, Error> {
            let native = Node.mapChatRequest(request)
            return AsyncThrowingStream { continuation in
                do {
                    let bridge = EventStreamBridge(continuation: continuation) { [handle] requestId in
                        try? handle.cancel(requestId: requestId)
                    }
                    let requestId = try handle.chat(request: native, listener: bridge)
                    bridge.activate(requestId: requestId)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        public func responses(_ request: ResponsesRequest) -> AsyncThrowingStream<Event, Error> {
            let native = Node.mapResponsesRequest(request)
            return AsyncThrowingStream { continuation in
                do {
                    let bridge = EventStreamBridge(continuation: continuation) { [handle] requestId in
                        try? handle.cancel(requestId: requestId)
                    }
                    let requestId = try handle.responses(request: native, listener: bridge)
                    bridge.activate(requestId: requestId)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        public func cancel(_ requestId: RequestId) async throws {
            let handle = self.handle
            try await runBlocking {
                try handle.cancel(requestId: requestId.value)
            }
        }
    }

    public final class Models: @unchecked Sendable {
        private let handle: MeshNodeHandle

        fileprivate init(handle: MeshNodeHandle) {
            self.handle = handle
        }

        public func recommended() async throws -> [ModelSummary] {
            let handle = self.handle
            return try await runBlocking { try handle.recommendedModels() }
        }

        public func search(_ query: ModelSearchQuery) async throws -> [ModelSummary] {
            let handle = self.handle
            return try await runBlocking { try handle.searchModels(query: query) }
        }

        public func show(_ modelRef: String) async throws -> ModelDetails {
            let handle = self.handle
            return try await runBlocking { try handle.showModel(modelRef: modelRef) }
        }

        public func installed() async throws -> [InstalledModel] {
            let handle = self.handle
            return try await runBlocking { try handle.installedModels() }
        }

        public func cacheStatus() async throws -> ModelCacheStatus {
            let handle = self.handle
            return try await runBlocking { try handle.modelCacheStatus() }
        }

        public func download(_ modelRef: String) async throws -> DownloadedModel {
            let handle = self.handle
            return try await runBlocking { try handle.downloadModel(modelRef: modelRef) }
        }

        public func delete(_ modelRef: String, options: DeleteModelOptions) async throws -> DeleteModelResult {
            let handle = self.handle
            return try await runBlocking { try handle.deleteModel(modelRef: modelRef, options: options) }
        }

        public func cleanup(_ policy: CleanupPolicy) async throws -> CleanupResult {
            let handle = self.handle
            return try await runBlocking { try handle.cleanupModels(policy: policy) }
        }

        public func pruneDerivedCache(_ policy: PrunePolicy) async throws -> PruneResult {
            let handle = self.handle
            return try await runBlocking { try handle.pruneDerivedCache(policy: policy) }
        }
    }

    public final class Serving: @unchecked Sendable {
        private let handle: MeshNodeHandle

        fileprivate init(handle: MeshNodeHandle) {
            self.handle = handle
        }

        public func status() async throws -> ServingStatus {
            let handle = self.handle
            return try await runBlocking { try handle.servingStatus() }
        }

        public func servedModels() async throws -> [ServedModel] {
            let handle = self.handle
            return try await runBlocking { try handle.servedModels() }
        }

        public func load(_ modelRef: String, options: LoadModelOptions) async throws -> ServedModel {
            let handle = self.handle
            return try await runBlocking { try handle.loadServingModel(modelRef: modelRef, options: options) }
        }

        public func unload(_ target: UnloadTarget, options: UnloadModelOptions) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.unloadServingModel(target: target, options: options) }
        }

        public func unloadModel(_ modelId: String, options: UnloadModelOptions) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.unloadServingModelById(modelId: modelId, options: options) }
        }

        public func unloadInstance(_ instanceId: String, options: UnloadModelOptions) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.unloadServingInstance(instanceId: instanceId, options: options) }
        }

        public func setDevicePolicy(_ policy: DevicePolicy) async throws {
            let handle = self.handle
            return try await runBlocking { try handle.setDevicePolicy(policy: policy) }
        }
    }

    fileprivate static func mapModel(_ native: ModelNative) -> Model {
        Model(id: native.id, name: native.name)
    }

    static func mapEvent(_ native: ClientEvent) -> Event {
        switch native {
        case .connecting:
            return .connecting
        case .joined(let nodeId):
            return .joined(nodeId: nodeId)
        case .modelsUpdated(let models):
            return .modelsUpdated(models: models.map(mapModel))
        case .tokenDelta(let requestId, let delta):
            return .tokenDelta(requestId: requestId, delta: delta)
        case .completed(let requestId):
            return .completed(requestId: requestId)
        case .failed(let requestId, let error):
            return .failed(requestId: requestId, error: error)
        case .disconnected(let reason):
            return .disconnected(reason: reason)
        }
    }

    private static func mapChatRequest(_ request: ChatRequest) -> ChatRequestNative {
        ChatRequestNative(
            model: request.model,
            messages: request.messages.map {
                ChatMessageNative(role: $0.role, content: $0.content)
            }
        )
    }

    private static func mapResponsesRequest(_ request: ResponsesRequest) -> ResponsesRequestNative {
        ResponsesRequestNative(model: request.model, input: request.input)
    }
}
#elseif MESH_SWIFT_STUB
public func generateOwnerKeypairHex() -> String {
    UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
}

public struct PublicMeshQuery: Sendable, Hashable {
    public var model: String?
    public var minVramGb: Double?
    public var region: String?
    public var targetName: String?
    public var relays: [String]

    public init(model: String?, minVramGb: Double?, region: String?, targetName: String?, relays: [String]) {
        self.model = model
        self.minVramGb = minVramGb
        self.region = region
        self.targetName = targetName
        self.relays = relays
    }
}

public struct PublicMesh: Sendable, Hashable {
    public var inviteToken: String
}

public enum CapabilityLevel: Sendable, Hashable {
    case none
    case likely
    case supported
}

public struct ModelCapabilities: Sendable, Hashable {
    public var multimodal: Bool
    public var vision: CapabilityLevel
    public var audio: CapabilityLevel
    public var reasoning: CapabilityLevel
    public var toolUse: CapabilityLevel
    public var moe: Bool
}

public struct ModelSummary: Sendable, Hashable {
    public var id: String
    public var name: String
    public var sizeLabel: String?
    public var description: String?
    public var capabilities: ModelCapabilities
}

public struct ModelSearchQuery: Sendable, Hashable {
    public var query: String
    public var limit: UInt64?

    public init(query: String, limit: UInt64?) {
        self.query = query
        self.limit = limit
    }
}

public enum ModelSource: Sendable, Hashable {
    case catalog
    case huggingFace
    case local
}

public enum ModelKind: Sendable, Hashable {
    case gguf
    case safetensors
    case layerPackage
    case unknown
}

public struct ModelDetails: Sendable, Hashable {
    public var id: String
    public var name: String
    public var source: ModelSource
    public var kind: ModelKind
    public var modelRef: String
    public var downloadRef: String
    public var path: String?
    public var sizeBytes: UInt64?
    public var sizeLabel: String?
    public var description: String?
    public var draft: String?
    public var installed: Bool
    public var capabilities: ModelCapabilities
}

public struct InstalledModel: Sendable, Hashable {
    public var modelRef: String
    public var path: String
    public var sizeBytes: UInt64?
    public var capabilities: ModelCapabilities
}

public struct ModelCacheStatus: Sendable, Hashable {
    public var cacheDir: String?
}

public struct DownloadedModel: Sendable, Hashable {
    public var modelRef: String
    public var paths: [String]
    public var primaryPath: String?
    public var details: ModelDetails?
}

public struct DeleteModelOptions: Sendable, Hashable {
    public var force: Bool

    public init(force: Bool) {
        self.force = force
    }
}

public struct DeleteModelResult: Sendable, Hashable {
    public var deletedPaths: [String]
    public var reclaimedBytes: UInt64
}

public struct CleanupPolicy: Sendable, Hashable {
    public var removeAll: Bool

    public init(removeAll: Bool) {
        self.removeAll = removeAll
    }
}

public struct CleanupResult: Sendable, Hashable {
    public var deletedPaths: [String]
    public var reclaimedBytes: UInt64
    public var skippedPaths: [String]
}

public struct PrunePolicy: Sendable, Hashable {
    public var removeAll: Bool

    public init(removeAll: Bool) {
        self.removeAll = removeAll
    }
}

public struct PruneResult: Sendable, Hashable {
    public var deletedPaths: [String]
    public var reclaimedBytes: UInt64
}

public enum DevicePolicy: Sendable, Hashable {
    case auto
    case cpu
    case gpu(deviceIds: [String])
}

public struct LoadModelOptions: Sendable, Hashable {
    public var devicePolicy: DevicePolicy

    public init(devicePolicy: DevicePolicy) {
        self.devicePolicy = devicePolicy
    }
}

public enum ServingModelState: Sendable, Hashable {
    case loading
    case ready
    case failed
    case unloading
    case stopped
    case unknown(value: String)
}

public struct ServedModel: Sendable, Hashable {
    public var modelRef: String
    public var modelId: String
    public var instanceId: String?
    public var state: ServingModelState
    public var backend: String?
    public var capabilities: ModelCapabilities
    public var contextLength: UInt32?
    public var error: String?
}

public struct ServingStatus: Sendable, Hashable {
    public var enabled: Bool
    public var models: [ServedModel]
}

public enum UnloadTarget: Sendable, Hashable {
    case model(modelId: String)
    case instance(instanceId: String)
}

public struct UnloadModelOptions: Sendable, Hashable {
    public var drainTimeoutMs: UInt64
    public var force: Bool

    public init(drainTimeoutMs: UInt64, force: Bool) {
        self.drainTimeoutMs = drainTimeoutMs
        self.force = force
    }
}

public final class Node: @unchecked Sendable {
    public let inference: Inference
    public let models: Models
    public let serving: Serving
    private var isConnected: Bool = false

    public init(
        inviteToken _: InviteToken,
        ownerKeypairBytesHex _: String,
        cacheDir _: String? = nil,
        runtimeDir _: String? = nil,
        servingEnabled _: Bool = false
    ) throws {
        self.inference = Inference()
        self.models = Models()
        self.serving = Serving()
    }

    public static func discoverPublicMeshes(
        _ query: PublicMeshQuery = PublicMeshQuery(
            model: nil,
            minVramGb: nil,
            region: nil,
            targetName: nil,
            relays: []
        )
    ) async throws -> [PublicMesh] {
        []
    }

    public static func connectPublic(
        ownerKeypairBytesHex _: String,
        query _: PublicMeshQuery = PublicMeshQuery(
            model: nil,
            minVramGb: nil,
            region: nil,
            targetName: nil,
            relays: []
        )
    ) async throws -> Node {
        try Node(inviteToken: InviteToken(""), ownerKeypairBytesHex: "")
    }

    public func start() async throws {
        isConnected = true
    }

    public func stop() async throws {
        isConnected = false
    }

    public func reconnect() async throws {
        isConnected = true
    }

    public func status() async -> Status {
        Status(connected: isConnected, peerCount: 0)
    }

    public final class Inference: @unchecked Sendable {
        public func listModels() async throws -> [Model] {
            []
        }

        public func chat(_ request: ChatRequest) -> AsyncThrowingStream<Event, Error> {
            let requestId = UUID().uuidString
            return AsyncThrowingStream { continuation in
                continuation.yield(.completed(requestId: requestId))
                continuation.finish()
            }
        }

        public func responses(_ request: ResponsesRequest) -> AsyncThrowingStream<Event, Error> {
            let requestId = UUID().uuidString
            return AsyncThrowingStream { continuation in
                continuation.yield(.completed(requestId: requestId))
                continuation.finish()
            }
        }

        public func cancel(_ requestId: RequestId) async throws {}
    }

    public final class Models: @unchecked Sendable {
        public func recommended() async throws -> [ModelSummary] { [] }
        public func search(_ query: ModelSearchQuery) async throws -> [ModelSummary] { [] }
        public func show(_ modelRef: String) async throws -> ModelDetails {
            ModelDetails(
                id: modelRef,
                name: modelRef,
                source: .local,
                kind: .unknown,
                modelRef: modelRef,
                downloadRef: modelRef,
                path: nil,
                sizeBytes: nil,
                sizeLabel: nil,
                description: nil,
                draft: nil,
                installed: false,
                capabilities: Self.emptyCapabilities
            )
        }
        public func installed() async throws -> [InstalledModel] { [] }
        public func cacheStatus() async throws -> ModelCacheStatus { ModelCacheStatus(cacheDir: nil) }
        public func download(_ modelRef: String) async throws -> DownloadedModel {
            DownloadedModel(modelRef: modelRef, paths: [], primaryPath: nil, details: nil)
        }
        public func delete(_ modelRef: String, options: DeleteModelOptions) async throws -> DeleteModelResult {
            DeleteModelResult(deletedPaths: [], reclaimedBytes: 0)
        }
        public func cleanup(_ policy: CleanupPolicy) async throws -> CleanupResult {
            CleanupResult(deletedPaths: [], reclaimedBytes: 0, skippedPaths: [])
        }
        public func pruneDerivedCache(_ policy: PrunePolicy) async throws -> PruneResult {
            PruneResult(deletedPaths: [], reclaimedBytes: 0)
        }

        private static let emptyCapabilities = ModelCapabilities(
            multimodal: false,
            vision: .none,
            audio: .none,
            reasoning: .none,
            toolUse: .none,
            moe: false
        )
    }

    public final class Serving: @unchecked Sendable {
        public func status() async throws -> ServingStatus { ServingStatus(enabled: false, models: []) }
        public func servedModels() async throws -> [ServedModel] { [] }
        public func load(_ modelRef: String, options: LoadModelOptions) async throws -> ServedModel {
            ServedModel(
                modelRef: modelRef,
                modelId: modelRef,
                instanceId: nil,
                state: .stopped,
                backend: nil,
                capabilities: ModelCapabilities(
                    multimodal: false,
                    vision: .none,
                    audio: .none,
                    reasoning: .none,
                    toolUse: .none,
                    moe: false
                ),
                contextLength: nil,
                error: nil
            )
        }
        public func unload(_ target: UnloadTarget, options: UnloadModelOptions) async throws {}
        public func unloadModel(_ modelId: String, options: UnloadModelOptions) async throws {}
        public func unloadInstance(_ instanceId: String, options: UnloadModelOptions) async throws {}
        public func setDevicePolicy(_ policy: DevicePolicy) async throws {}
    }
}
#else
#error("MeshLLM Swift SDK requires MeshLLMFFI.xcframework. Build it with sdk/swift/scripts/build-xcframework.sh or set MESH_SWIFT_FORCE_STUB=1 for unit tests only.")
#endif

private func runBlocking<T>(_ work: @escaping () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                continuation.resume(returning: try work())
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

private func runBlocking<T>(_ work: @escaping () -> T) async -> T {
    await withCheckedContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async {
            continuation.resume(returning: work())
        }
    }
}
