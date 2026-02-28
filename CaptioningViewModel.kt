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
import java.nio.FloatBuffer

sealed class CaptionState {
    object Idle : CaptionState()
    object Loading : CaptionState()
    data class Success(val caption: String, val inferenceTime: Long? = null) : CaptionState()
    data class Error(val message: String) : CaptionState()
}

class CaptioningViewModel : ViewModel() {

    private val _uiState = MutableStateFlow<CaptionState>(CaptionState.Idle)
    val uiState: StateFlow<CaptionState> = _uiState

    private var ortEnvironment: OrtEnvironment? = null
    private var visionSession: OrtSession? = null
    private var qformerSession: OrtSession? = null

    fun generateCaption(context: Context, bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = CaptionState.Loading
            val startTime = System.currentTimeMillis()

            try {
                // Initialize ORT sessions if not already done
                if (ortEnvironment == null) {
                    initializeSessions(context)
                }

                // Step 1: Preprocess image
                val pixelValues = preprocessImage(bitmap)

                // Step 2: Run Vision Encoder
                val imageFeatures = runVisionEncoder(pixelValues)

                // Step 3: Run Q-Former
                val languageFeatures = runQFormer(imageFeatures)

                val inferenceTime = System.currentTimeMillis() - startTime

                // Step 4: Placeholder caption (decoder comes in Week 3)
                val caption = "Image analyzed! Features extracted: " +
                        "${languageFeatures.size} values. " +
                        "Full captions coming in Week 3."

                _uiState.value = CaptionState.Success(
                    caption = caption,
                    inferenceTime = inferenceTime
                )

            } catch (e: Exception) {
                _uiState.value = CaptionState.Error(
                    message = e.message ?: "Unknown error occurred"
                )
            }
        }
    }

    private fun initializeSessions(context: Context) {
        ortEnvironment = OrtEnvironment.getEnvironment()

        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            try {
                addNnapi()
            } catch (e: Exception) {
                // NNAPI not available, fallback to CPU
            }
        }

        // Load Vision Encoder
        val visionModelBytes = context.assets.open(
            "blip2_vision_encoder_int8.onnx"
        ).readBytes()
        visionSession = ortEnvironment!!.createSession(
            visionModelBytes,
            sessionOptions
        )

        // Load Q-Former
        val qformerModelBytes = context.assets.open(
            "blip2_qformer.onnx"
        ).readBytes()
        qformerSession = ortEnvironment!!.createSession(
            qformerModelBytes,
            sessionOptions
        )
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        // Resize to 224x224
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val floatArray = FloatArray(3 * 224 * 224)
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)
                val r = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
                val g = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
                val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]

                floatArray[0 * 224 * 224 + y * 224 + x] = r
                floatArray[1 * 224 * 224 + y * 224 + x] = g
                floatArray[2 * 224 * 224 + y * 224 + x] = b
            }
        }
        return floatArray
    }

    private fun runVisionEncoder(pixelValues: FloatArray): FloatArray {
        val env = ortEnvironment!!
        val session = visionSession!!

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(pixelValues),
            longArrayOf(1, 3, 224, 224)
        )

        val results = session.run(mapOf("pixel_values" to inputTensor))
        val output = results[0].value as Array<Array<FloatArray>>

        return output.flatMap {
            it.flatMap { row -> row.toList() }
        }.toFloatArray()
    }

    private fun runQFormer(imageFeatures: FloatArray): FloatArray {
        val env = ortEnvironment!!
        val session = qformerSession!!

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(imageFeatures),
            longArrayOf(1, 257, 1408)
        )

        val results = session.run(mapOf("image_features" to inputTensor))
        val output = results[0].value as Array<Array<FloatArray>>

        return output.flatMap {
            it.flatMap { row -> row.toList() }
        }.toFloatArray()
    }

    override fun onCleared() {
        super.onCleared()
        visionSession?.close()
        qformerSession?.close()
        ortEnvironment?.close()
    }
}