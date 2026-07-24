package com.strawberryfields.app

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.*
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private lateinit var recordBtn: Button
    private lateinit var statusText: TextView
    private lateinit var resultsText: TextView

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    private var outputStream: ByteArrayOutputStream? = null

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val serverUrl = "http://10.0.2.2:5000"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recordBtn = findViewById(R.id.recordBtn)
        statusText = findViewById(R.id.statusText)
        resultsText = findViewById(R.id.resultsText)

        recordBtn.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                if (checkPermission()) {
                    startRecording()
                } else {
                    requestPermission()
                }
            }
        }

        checkHealth()
    }

    private fun checkPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 100)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording()
        } else {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startRecording() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return
        }

        outputStream = ByteArrayOutputStream()
        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, bufferSize)

        audioRecord?.startRecording()
        isRecording = true

        recordBtn.text = "STOP"
        recordBtn.setBackgroundColor(ContextCompat.getColor(this, R.color.red))
        statusText.text = "Recording..."

        recordingThread = Thread {
            val buffer = ShortArray(bufferSize / 2)
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (read > 0) {
                    val bytes = ByteArray(read * 2)
                    for (i in 0 until read) {
                        bytes[i * 2] = (buffer[i].toInt() and 0xFF).toByte()
                        bytes[i * 2 + 1] = (buffer[i].toInt() shr 8 and 0xFF).toByte()
                    }
                    outputStream?.write(bytes)
                }
            }
        }
        recordingThread?.start()
    }

    private fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        recordingThread?.join()

        recordBtn.text = "RECORD"
        recordBtn.setBackgroundColor(ContextCompat.getColor(this, R.color.primary))
        statusText.text = "Processing..."

        val audioData = outputStream?.toByteArray() ?: return
        sendAudio(audioData)
    }

    private fun sendAudio(pcmData: ByteArray) {
        Thread {
            try {
                val wavBytes = pcmToWav(pcmData, sampleRate, 1, 16)
                val requestBody = wavBytes.toRequestBody("audio/wav".toMediaType())
                val request = Request.Builder()
                    .url("$serverUrl/search")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val body = response.body?.string() ?: "{}"

                val json = com.google.gson.JsonParser.parseString(body).asJsonObject

                runOnUiThread {
                    if (json.has("error")) {
                        statusText.text = "Error: ${json.get("error").asString}"
                        resultsText.text = ""
                    } else {
                        val results = json.getAsJsonArray("results")
                        val duration = json.get("query_duration").asDouble
                        statusText.text = "Found ${results.size()} matches (${duration}s audio)"

                        val sb = StringBuilder()
                        for (i in 0 until results.size()) {
                            val r = results[i].asJsonObject
                            val rank = r.get("rank").asInt
                            val groupId = r.get("group_id").asString
                            val score = r.get("score").asDouble
                            sb.appendLine("#$rank  $groupId  (${String.format("%.1f", score * 100)}%)")
                        }
                        resultsText.text = sb.toString()
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    resultsText.text = ""
                }
            }
        }.start()
    }

    private fun pcmToWav(pcmData: ByteArray, sampleRate: Int, channels: Int, bitsPerSample: Int): ByteArray {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataSize = pcmData.size
        val totalSize = 44 + dataSize

        val baos = ByteArrayOutputStream()
        val dos = DataOutputStream(baos)

        dos.writeBytes("RIFF")
        dos.writeInt(Integer.reverseBytes(totalSize - 8))
        dos.writeBytes("WAVE")
        dos.writeBytes("fmt ")
        dos.writeInt(Integer.reverseBytes(16))
        dos.writeShort(java.lang.Short.reverseBytes(1))
        dos.writeShort(java.lang.Short.reverseBytes(channels.toShort()))
        dos.writeInt(Integer.reverseBytes(sampleRate))
        dos.writeInt(Integer.reverseBytes(byteRate))
        dos.writeShort(java.lang.Short.reverseBytes(blockAlign.toShort()))
        dos.writeShort(java.lang.Short.reverseBytes(bitsPerSample.toShort()))
        dos.writeBytes("data")
        dos.writeInt(Integer.reverseBytes(dataSize))
        dos.write(pcmData)
        dos.flush()

        return baos.toByteArray()
    }

    private fun checkHealth() {
        Thread {
            try {
                val request = Request.Builder()
                    .url("$serverUrl/health")
                    .get()
                    .build()
                val response = client.newCall(request).execute()
                val json = com.google.gson.JsonParser.parseString(response.body?.string() ?: "{}").asJsonObject
                val dbSize = json.get("db_size").asInt

                runOnUiThread {
                    if (dbSize > 0) {
                        statusText.text = "Ready — $dbSize songs indexed"
                    } else {
                        statusText.text = "Warning: No songs in database"
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    statusText.text = "Cannot connect to server"
                }
            }
        }.start()
    }
}
