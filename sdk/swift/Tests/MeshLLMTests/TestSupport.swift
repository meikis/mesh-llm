import XCTest
@testable import MeshLLM

func makeOwnerKeypairBytesHex() -> String {
    #if canImport(MeshLLMFFI) || MESH_SWIFT_STUB
    return generateOwnerKeypairHex()
    #endif
}

func makeTestNode() throws -> Node {
    try Node(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
}
