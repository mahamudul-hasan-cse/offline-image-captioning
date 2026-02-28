package com.example.offlinecaptioning

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

class CaptioningViewModel : ViewModel() {

    sealed class UiState {
        object Idle : UiState()
        data class Loading(val message: String = "Analyzing...") : UiState()
        data class Success(val caption: String, val inferenceTime: Long? = null) : UiState()
        data class Error(val message: String) : UiState()
    }

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState

    private var ortEnv: OrtEnvironment? = null
    private var visionSession: OrtSession? = null
    private var decoderSession: OrtSession? = null
    private var vocab: Map<Int, String> = emptyMap()
    private var tokenToId: Map<String, Int> = emptyMap()
    private var appContext: Context? = null

    fun setContext(context: Context) {
        appContext = context.applicationContext
    }

    fun generateCaption(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            val startTime = System.currentTimeMillis()
            try {
                val ctx = appContext ?: throw Exception("Context not set")

                if (ortEnv == null) {
                    _uiState.value = UiState.Loading("Loading models...")
                    initSessions(ctx)
                }

                _uiState.value = UiState.Loading("Processing image...")
                val pixelValues = preprocessImage(bitmap)

                _uiState.value = UiState.Loading("Extracting features...")
                val imageFeatures = runVisionEncoder(pixelValues)

                _uiState.value = UiState.Loading("Generating caption...")
                val caption = generateText(imageFeatures)

                val inferenceTime = System.currentTimeMillis() - startTime
                _uiState.value = UiState.Success(caption, inferenceTime)

            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.message ?: "Unknown error")
            }
        }
    }

    private fun initSessions(context: Context) {
        ortEnv = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        val modelDir = context.getExternalFilesDir(null)?.absolutePath + "/models"

        val visionPath = "$modelDir/blip_vision_encoder.onnx"
        val decoderPath = "$modelDir/blip_text_decoder.onnx"
        val vocabPath = "$modelDir/vocab.txt"

        if (!File(visionPath).exists()) throw Exception("Vision encoder not found")
        if (!File(decoderPath).exists()) throw Exception("Text decoder not found")
        if (!File(vocabPath).exists()) throw Exception("Vocab not found")

        visionSession = ortEnv!!.createSession(visionPath, opts)
        decoderSession = ortEnv!!.createSession(decoderPath, opts)

        // Load vocab
        val vocabMap = mutableMapOf<Int, String>()
        val tokenMap = mutableMapOf<String, Int>()
        File(vocabPath).readLines().forEachIndexed { idx, line ->
            vocabMap[idx] = line.trim()
            tokenMap[line.trim()] = idx
        }
        vocab = vocabMap
        tokenToId = tokenMap
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 384, 384, true)
        val floatArray = FloatArray(3 * 384 * 384)
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        for (y in 0 until 384) {
            for (x in 0 until 384) {
                val pixel = resized.getPixel(x, y)
                floatArray[0 * 384 * 384 + y * 384 + x] = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
                floatArray[1 * 384 * 384 + y * 384 + x] = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
                floatArray[2 * 384 * 384 + y * 384 + x] = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
            }
        }
        return floatArray
    }

    private fun runVisionEncoder(pixelValues: FloatArray): FloatArray {
        val tensor = OnnxTensor.createTensor(
            ortEnv!!,
            FloatBuffer.wrap(pixelValues),
            longArrayOf(1, 3, 384, 384)
        )
        val results = visionSession!!.run(mapOf("pixel_values" to tensor))
        val output = results[0].value as Array<Array<FloatArray>>
        return output[0].flatMap { it.toList() }.toFloatArray()
    }

    private fun generateText(imageFeatures: FloatArray): String {
        val BOS_TOKEN = 30522  // [CLS] token id for BLIP
        val EOS_TOKEN = 102    // [SEP] token id
        val MAX_LEN = 30

        val inputIds = mutableListOf<Long>(BOS_TOKEN.toLong())

        for (step in 0 until MAX_LEN) {
            val seqLen = inputIds.size.toLong()
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                LongBuffer.wrap(inputIds.toLongArray()),
                longArrayOf(1, seqLen)
            )
            val attentionMask = OnnxTensor.createTensor(
                ortEnv!!,
                LongBuffer.wrap(LongArray(inputIds.size) { 1L }),
                longArrayOf(1, seqLen)
            )
            val encoderHidden = OnnxTensor.createTensor(
                ortEnv!!,
                FloatBuffer.wrap(imageFeatures),
                longArrayOf(1, (imageFeatures.size / 768).toLong(), 768L)
            )

            val results = decoderSession!!.run(mapOf(
                "input_ids" to inputTensor,
                "attention_mask" to attentionMask,
                "encoder_hidden_states" to encoderHidden
            ))

            val logits = results[0].value as Array<Array<FloatArray>>
            val lastLogits = logits[0][inputIds.size - 1]

            // Greedy decoding - pick highest probability token
            val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] }?.toLong() ?: break

            if (nextToken == EOS_TOKEN.toLong()) break
            inputIds.add(nextToken)
        }

        // Convert token ids to text
        return inputIds.drop(1).mapNotNull { vocab[it.toInt()] }
            .filter { !it.startsWith("##") || it.length > 2 }
            .joinToString(" ")
            .replace(" ##", "")
            .trim()
            .ifEmpty { "A scene captured by camera" }
    }

    override fun onCleared() {
        super.onCleared()
        visionSession?.close()
        decoderSession?.close()
        ortEnv?.close()
    }
}
