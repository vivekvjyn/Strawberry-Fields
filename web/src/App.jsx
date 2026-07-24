import { useState, useRef, useCallback, useEffect } from 'react'

const API_URL = '/api'

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [results, setResults] = useState([])
  const [status, setStatus] = useState('Click to start')
  const [serverReady, setServerReady] = useState(false)
  const canvasRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const analyserRef = useRef(null)
  const animFrameRef = useRef(null)
  const audioContextRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(data => {
        if (data.model_loaded && data.db_size > 0) {
          setServerReady(true)
          setStatus(`Ready — ${data.db_size} songs`)
        } else {
          setStatus('Server running but no model/database')
        }
      })
      .catch(() => setStatus('Cannot connect to server'))
  }, [])

  const drawOscilloscope = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !analyserRef.current) return

    const ctx = canvas.getContext('2d')
    const analyser = analyserRef.current
    const dataArray = new Uint8Array(analyser.frequencyBinCount)

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw)
      analyser.getByteTimeDomainData(dataArray)

      ctx.fillStyle = '#1a1a2e'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      ctx.lineWidth = 2
      ctx.strokeStyle = '#00d4ff'
      ctx.beginPath()

      const sliceWidth = canvas.width / dataArray.length
      let x = 0

      for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * canvas.height) / 2

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
        x += sliceWidth
      }

      ctx.lineTo(canvas.width, canvas.height / 2)
      ctx.stroke()
    }

    draw()
  }, [])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      audioContextRef.current = new AudioContext()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 2048
      source.connect(analyserRef.current)

      drawOscilloscope()

      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' })
      const chunks = []

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data)
      }

      mediaRecorderRef.current.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        cancelAnimationFrame(animFrameRef.current)

        const blob = new Blob(chunks, { type: 'audio/webm' })
        await sendAudio(blob)
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)
      setStatus('Recording...')
      setResults([])
    } catch (err) {
      setStatus(`Error: ${err.message}`)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    setIsRecording(false)
    setStatus('Processing...')
  }

  const sendAudio = async (blob) => {
    try {
      const formData = new FormData()
      formData.append('audio', blob, 'hum.wav')

      const res = await fetch(`${API_URL}/search`, { method: 'POST', body: formData })
      const data = await res.json()

      if (data.error) {
        setStatus(`Error: ${data.error}`)
        return
      }

      setResults(data.results)
      setStatus(`${data.results.length} matches (${data.query_duration}s audio)`)
    } catch (err) {
      setStatus(`Error: ${err.message}`)
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0a1a] text-white flex flex-col items-center px-4 py-8">
      <h1 className="text-3xl font-bold mb-1 tracking-tight">
        <span className="text-[#ff4d6d]">Strawberry</span> Fields
      </h1>
      <p className="text-sm text-gray-500 mb-8">Hum to search</p>

      <div className="w-full max-w-lg">
        <div className="rounded-xl overflow-hidden border border-gray-800 mb-6">
          <canvas
            ref={canvasRef}
            width={640}
            height={160}
            className="w-full h-40 bg-[#1a1a2e] block"
          />
        </div>

        <div className="flex flex-col items-center gap-4 mb-8">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!serverReady && !isRecording}
            className={`w-20 h-20 rounded-full flex items-center justify-center text-sm font-bold transition-all
              ${isRecording
                ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                : 'bg-[#00d4ff] hover:bg-[#00b8d9] text-black'
              } disabled:opacity-30 disabled:cursor-not-allowed`}
          >
            {isRecording ? 'STOP' : 'RECORD'}
          </button>

          <p className="text-sm text-gray-400">{status}</p>
        </div>

        {results.length > 0 && (
          <div className="rounded-xl border border-gray-800 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-gray-500">
                  <th className="py-2 px-4 text-left font-medium w-10">#</th>
                  <th className="py-2 px-4 text-left font-medium">Song</th>
                  <th className="py-2 px-4 text-right font-medium">Score</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.rank} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2 px-4 text-gray-500">{r.rank}</td>
                    <td className="py-2 px-4 font-mono text-xs text-gray-300">{r.group_id}</td>
                    <td className="py-2 px-4 text-right">
                      <span className={`font-mono ${r.rank === 1 ? 'text-green-400 font-bold' : 'text-gray-400'}`}>
                        {(r.score * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
