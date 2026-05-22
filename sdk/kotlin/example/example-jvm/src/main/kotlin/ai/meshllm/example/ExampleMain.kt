package ai.meshllm.example

import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Event
import ai.meshllm.InviteToken
import ai.meshllm.Node
import com.sun.jna.NativeLibrary
import kotlinx.coroutines.runBlocking
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) = runBlocking {
    val inviteToken = args.firstOrNull { !it.startsWith("--") } ?: run {
        System.err.println("Usage: ExampleMain <invite_token>")
        System.err.println("Set jna.library.path to the directory containing libmeshllm_ffi.")
        return@runBlocking
    }

    NativeLibrary.getInstance("meshllm_ffi")
    // Generate an ephemeral owner keypair for the example. In a real app this
    // must be persisted across launches.
    val ownerKeypairHex = uniffi.mesh_ffi.generateOwnerKeypairHex()
    val node = Node(InviteToken(inviteToken), ownerKeypairHex)
    val recommended = node.models.recommended()
    val serving = node.serving.status()
    println("[node] recommended_models=${recommended.size} serving_enabled=${serving.enabled}")

    node.start()
    println("[connected]")

    val models = waitForModels(node)
    println("[models] N=${models.size}")
    check(models.isNotEmpty()) { "mesh reported no models" }

    val selectedModel = System.getenv("MESH_SDK_MODEL_ID") ?: models.first().id

    val chatRequest = ChatRequest(
        model = selectedModel,
        messages = listOf(ChatMessage(role = "user", content = "hello")),
    )

    val latch = CountDownLatch(1)
    var firstTokenEmitted = false
    var completed = false
    var failed: String? = null
    val chatStartMs = System.currentTimeMillis()

    node.inference.chat(chatRequest) { event ->
        when (event) {
            is Event.TokenDelta -> {
                if (!firstTokenEmitted) {
                    firstTokenEmitted = true
                    val elapsedMs = System.currentTimeMillis() - chatStartMs
                    println("[chat] first_token_ms=$elapsedMs")
                }
            }
            is Event.Completed -> {
                completed = true
                latch.countDown()
            }
            is Event.Failed -> {
                failed = event.error
                latch.countDown()
            }
            else -> Unit
        }
    }

    check(latch.await(60, TimeUnit.SECONDS)) { "chat timed out waiting for completion" }
    check(failed == null) { "chat failed: $failed" }
    check(firstTokenEmitted) { "chat emitted no token deltas" }
    check(completed) { "chat never completed" }
    println("[chat] done")

    node.stop()
    println("[disconnect] ok")
}

private fun waitForModels(node: Node): List<ai.meshllm.Model> {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while (System.nanoTime() < deadline) {
        val models = runBlocking { node.inference.listModels() }
        if (models.isNotEmpty()) {
            return models
        }
        Thread.sleep(250)
    }
    return emptyList()
}
