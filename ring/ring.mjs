// Litter Log Ring sidecar: records camera clips (free live view, no Ring Protect needed) when the camera
// sees motion or when a litter robot reports a cat in/out (POST /capture from server.py), extracts
// 1 fps frames, and hands the clip to server.py at /internal/clip.
import { RingApi } from 'ring-client-api'
import { firstValueFrom } from 'rxjs'
import { readFileSync, writeFileSync, mkdirSync, statSync, readdirSync, renameSync, rmSync } from 'node:fs'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import http from 'node:http'
import path from 'node:path'

const HOME = process.env.CATBOX_HOME || path.resolve(import.meta.dirname, '..')
const SECRETS = path.join(HOME, 'secrets')
const MEDIA = path.join(HOME, 'data', 'media')
const SERVER = process.env.CATBOX_SERVER || 'http://127.0.0.1:8902'
const PORT = Number(process.env.CATBOX_RING_PORT || 8903)
// cams.json (shared with server.py): Ring camera name -> { robot it watches, label }
const CAMS = JSON.parse(readFileSync(path.join(HOME, 'cams.json'), 'utf8'))
const CAM_NAMES = Object.keys(CAMS).map(s => s.toLowerCase())
const SECS = { motion: 40, in: 420, out: 30, peek: 8, test: 20, nap: 6 }
// a visit recording runs from the robot's "cat in" until TAIL_SECS after its "cat out" (SECS.in is only the cap):
// the EVO reports a cat up to 90 s into a visit, and the upstairs camera sees the room a cat walks out into, not the box
const TAIL_SECS = 25
const NAP_EVERY_MIN = Number(process.env.CATBOX_NAP_EVERY_MIN || 10)
const NAP_MIN_BATTERY = 25   // a battery camera keeps its charge for real litter-box visits first
// a robot's in/out trigger only wakes the camera that watches that robot
const CAM_ROBOT = Object.fromEntries(Object.entries(CAMS).map(([name, c]) => [name, c.robot]))
const run = promisify(execFile)

const tokenFile = path.join(SECRETS, 'ring-token.json')
const tok = JSON.parse(readFileSync(tokenFile, 'utf8'))
const internalKey = readFileSync(path.join(SECRETS, 'internal-key'), 'utf8').trim()
const log = (...a) => console.log(new Date().toISOString(), ...a)

const api = new RingApi({ refreshToken: tok.refreshToken, controlCenterDisplayName: 'catbox', cameraStatusPollingSeconds: 60 })
api.onRefreshTokenUpdated.subscribe(({ newRefreshToken }) => {
  tok.refreshToken = newRefreshToken
  tok.saved = new Date().toISOString()
  writeFileSync(tokenFile, JSON.stringify(tok) + '\n', { mode: 0o600 })
})

const post = (p, body) => fetch(SERVER + p, {
  method: 'POST', headers: { 'content-type': 'application/json', 'x-catbox-key': internalKey }, body: JSON.stringify(body),
}).then(r => r.json()).catch(e => log('post', p, 'failed:', e.message))

const cams = (await api.getCameras()).filter(c => CAM_NAMES.includes(c.name.toLowerCase()))
if (!cams.length) log('no camera named', CAM_NAMES, '- nothing to record')
const online = c => c.data?.alerts?.connection !== 'offline'
const busy = new Map() // camera id -> { reason, robot, call (the live stream), tail (stop timer), pending: {reason, robot} | null }
let stopping = false   // set on SIGTERM: stop recordings early (they still save), start nothing new
let finishing = 0      // clips recorded and still being processed (the camera is already free to record again)
// ffmpeg's decoder reports a damaged frame as a warning and hides it; the null muxer's dts notes are not damage.
// Says which streams are damaged: the audio decoder's lines name aac (its own, or the aist#0:1/aac input stream).
const damaged = async f => {
  const { stderr } = await run('ffmpeg', ['-hide_banner', '-nostats', '-v', 'warning', '-i', f, '-f', 'null', '-'],
    { timeout: 120000, maxBuffer: 16 << 20 })
  const bad = stderr.split('\n').filter(l => l.trim() && !l.includes('non monotonically'))
  const audio = bad.filter(l => /aac/.test(l))
  return { video: bad.length > audio.length, audio: audio.length > 0 }
}

async function capture(cam, reason, robot) {
  if (stopping) return 'stopping'
  const cur = busy.get(cam.id)
  if (cur) {
    if (cur.reason === 'in' && cur.robot === robot && (reason === 'in' || reason === 'out')) {
      clearTimeout(cur.tail)
      cur.tail = reason === 'out' ? setTimeout(() => cur.call?.stop(), TAIL_SECS * 1000) : null
      log(cam.name, robot, reason === 'out' ? `cat out, recording ${TAIL_SECS} s more` : 'cat back in, still recording')
      return 'following'
    }
    if (reason === 'motion' || reason === 'nap') return 'busy'
    cur.pending = { reason, robot }
    // a visit beats a motion clip or a nap look: cut that short so the visit recording starts now
    if (cur.reason === 'motion' || cur.reason === 'nap') cur.call?.stop()
    return 'busy'
  }
  // no online() gate: Ring's status lags a minute behind (a 2026-10-08 motion event was skipped as
  // "offline"), and a failed attempt on a really-offline camera just yields a tiny clip that is dropped
  const state = { reason, robot, call: null, tail: null, pending: null }
  busy.set(cam.id, state)
  const start = new Date()
  const day = start.toISOString().slice(0, 10)
  const base = `${start.toISOString().slice(11, 19).replace(/:/g, '')}-${reason}-${cam.id}`
  const rel = `${day}/${base}`
  const clip = path.join(MEDIA, rel + '.mp4')
  const part = path.join(MEDIA, rel + '.part.mp4')   // a killed recording leaves no .mp4 (startup removes .part.mp4 files)
  const fixed = path.join(MEDIA, rel + '.fix.part.mp4')
  const clean = path.join(MEDIA, rel + '.clean.part.mp4')
  let counted = false
  const release = () => {   // free the camera for its next recording
    if (busy.get(cam.id) !== state) return
    busy.delete(cam.id)
    if (state.pending) setTimeout(() => capture(cam, state.pending.reason, state.pending.robot), 500)
  }
  try {
    mkdirSync(path.join(MEDIA, rel), { recursive: true })
    // fragmented while recording: a stream stopped early (cat out, or cut short for a visit) gets killed before
    // ffmpeg can write a normal mp4's index, and a fragmented file is readable up to its last fragment
    state.call = await cam.streamVideo({ output: ['-t', String(SECS[reason] ?? 40), '-movflags', 'frag_keyframe+empty_moov', part] })
    if (stopping) state.call.stop()
    await firstValueFrom(state.call.onCallEnded)
    clearTimeout(state.tail)
    await new Promise(r => setTimeout(r, 1500))   // the last fragment's write
    const size = statSync(part, { throwIfNoEntry: false })?.size || 0
    if (size < 20000) throw new Error(`clip only ${size} bytes`)
    release(); finishing++; counted = true
    // a plain mp4 with its index up front, for the page's video player
    await run('ffmpeg', ['-loglevel', 'error', '-y', '-i', part, '-c', 'copy', '-movflags', '+faststart', fixed], { timeout: 60000 })
    // The Ring stream drops data now and then (damaged frames in 34 of 104 clips, 2026-10-09 to 10-12). Chrome on a
    // Mac decodes in hardware and stops at the first one (a 2:47 AM clip played 0.4 s); other players skip it.
    // A cut-off last audio packet (8 of 138 clips by 10-13) stops it with an error about a second before the end.
    const bad = await damaged(fixed)
    if (bad.video || bad.audio) {
      await run('nice', ['-n', '10', 'ffmpeg', '-loglevel', 'error', '-y', '-i', fixed,
        ...(bad.video ? ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26'] : ['-c:v', 'copy']),
        ...(bad.audio ? ['-c:a', 'aac', '-b:a', '128k'] : ['-c:a', 'copy']),
        '-movflags', '+faststart', clean], { timeout: 300000 })
      renameSync(clean, fixed)
      log(cam.name, reason, 'clip', rel, `re-encoded: the stream had damaged ${bad.video ? 'frames' : 'audio'}`)
    }
    renameSync(fixed, clip)
    rmSync(part, { force: true })
    await run('ffmpeg', ['-loglevel', 'error', '-y', '-i', clip, '-vf', 'fps=1,scale=960:-2', '-q:v', '4',
      path.join(MEDIA, rel, '%03d.jpg')], { timeout: 120000 })
    // 1 fps, so frame N is at N-1 s; server.py scores each frame by how much of it differs from the empty room
    const frames = readdirSync(path.join(MEDIA, rel)).filter(f => f.endsWith('.jpg')).sort()
      .map((f, i) => ({ f, t: i, s: 0 }))
    const end = Math.round(start.getTime() / 1000) + (frames.at(-1)?.t ?? SECS[reason] ?? 40)
    const res = await post('/internal/clip', { cam: cam.name, reason, robot, path: rel, frames,
      start_ts: Math.round(start.getTime() / 1000), end_ts: end })
    log(cam.name, reason, robot || '', 'clip', rel, `${frames.length} frames`, res?.look ? JSON.stringify(res.look) : '')
  } catch (e) {
    log(cam.name, reason, 'capture failed:', e.message)
    for (const f of [part, fixed, clean, clip, path.join(MEDIA, rel)]) rmSync(f, { recursive: true, force: true })
  } finally {
    release()
    if (counted) finishing--
  }
  return 'ok'
}

for (const cam of cams) {
  cam.onMotionStarted.subscribe(() => { log(cam.name, 'motion'); capture(cam, 'motion') })
}

// leftovers from a recording that was killed (power cut, crash): never a finished clip, so remove them
mkdirSync(MEDIA, { recursive: true })
for (const day of readdirSync(MEDIA, { withFileTypes: true }).filter(d => d.isDirectory())) {
  for (const f of readdirSync(path.join(MEDIA, day.name)).filter(f => f.endsWith('.part.mp4'))) {
    const base = path.join(MEDIA, day.name, f.slice(0, -'.part.mp4'.length))
    for (const x of [base + '.part.mp4', base]) rmSync(x, { recursive: true, force: true })
    log('removed unfinished recording', `${day.name}/${f}`)
  }
}

// a restart (deploy, update) ends a recording in progress early and saves it instead of losing it
for (const sig of ['SIGTERM', 'SIGINT']) process.on(sig, () => {
  if (stopping) return
  stopping = true
  const t0 = Date.now()
  const wait = () => ((busy.size || finishing) && Date.now() - t0 < 60000) ? setTimeout(wait, 500) : process.exit(0)
  if (busy.size) log('stopping after', busy.size, 'recording(s) finish')
  for (const st of busy.values()) st.call?.stop()
  wait()
})

// nap time: a short look every NAP_EVERY_MIN on cameras with "naps": true in cams.json (server.py keeps only
// the pet crops, and nothing at all when a person is in view)
const batteryOf = c => { const b = Number(c.data?.battery_life); return Number.isFinite(b) ? b : null }
setInterval(() => {
  for (const cam of cams) {
    if (!CAMS[cam.name]?.naps) continue
    const b = batteryOf(cam)
    if (b !== null && cam.data?.settings?.power_mode !== 'wired' && b < NAP_MIN_BATTERY) { log(cam.name, `nap check skipped: battery ${b}%`); continue }
    capture(cam, 'nap')
  }
}, NAP_EVERY_MIN * 60000)

const reportCams = () => post('/internal/cam', { nap_every_min: NAP_EVERY_MIN, cams: cams.map(c => ({ name: c.name, online: online(c), battery: batteryOf(c) })) })
reportCams()
setInterval(reportCams, 120000)

http.createServer((req, res) => {
  // Every route needs the app's key: a web page open in a browser on this machine can reach 127.0.0.1,
  // and a custom header can't be sent cross-site without a CORS preflight, which this server never answers.
  if (req.headers['x-catbox-key'] !== internalKey) { res.writeHead(403); res.end(); return }
  if (req.method === 'POST' && req.url === '/capture') {
    let body = ''
    req.on('data', d => { body += d })
    req.on('end', () => {
      let b = {}
      try { b = JSON.parse(body || '{}') } catch {}
      const reason = SECS[b.reason] ? b.reason : 'motion'
      log('trigger', reason, b.robot || '')
      for (const cam of cams) {
        const watches = CAM_ROBOT[cam.name]
        if (!b.robot || !watches || watches === b.robot) capture(cam, reason, b.robot)
      }
      res.writeHead(202, { 'content-type': 'application/json' }).end('{"ok":true}')
    })
    return
  }
  if (req.method === 'POST' && req.url === '/device-settings') {
    // {cam, settings}: a partial Ring device-settings patch, e.g. {"video_settings": {"dynamic_resolution_enabled": false}}
    let body = ''
    req.on('data', d => { body += d })
    req.on('end', async () => {
      try {
        const b = JSON.parse(body || '{}')
        const cam = cams.find(c => c.name === b.cam)
        if (!cam) return res.writeHead(404).end('no such camera')
        await cam.setDeviceSettings(b.settings)
        const now = await cam.getDeviceSettings()
        log(cam.name, 'device settings patched', JSON.stringify(b.settings))
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true, video_settings: now.video_settings }))
      } catch (e) { res.writeHead(500).end(String(e?.message || e)) }
    })
    return
  }
  if (req.method === 'POST' && req.url === '/settings') {
    // {cam, settings}: a partial doorbot settings patch, e.g. {"motion_detection_enabled": true}
    let body = ''
    req.on('data', d => { body += d })
    req.on('end', async () => {
      try {
        const b = JSON.parse(body || '{}')
        const cam = cams.find(c => c.name === b.cam)
        if (!cam) return res.writeHead(404).end('no such camera')
        await cam.setSettings(b.settings)
        await new Promise(r => setTimeout(r, 3000))
        const d = (await api.getCameras()).find(x => x.id === cam.id)?.data
        log(cam.name, 'settings patched', JSON.stringify(b.settings))
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true, motion_detection_enabled: d?.settings?.motion_detection_enabled }))
      } catch (e) { res.writeHead(500).end(String(e?.message || e)) }
    })
    return
  }
  if (req.method === 'GET' && req.url === '/status') {
    // fresh device data + the video/motion settings that matter here (read-only)
    Promise.all(cams.map(async c => {
      const d = (await api.getCameras().catch(() => [])).find(x => x.id === c.id)?.data || c.data
      const dev = await c.getDeviceSettings?.().catch(e => ({ error: e.message })) ?? {}
      return { name: c.name, id: c.id, kind: d.kind, online: d.alerts?.connection !== 'offline', alerts: d.alerts,
        firmware: d.firmware_version, battery: d.battery_life, power: d.settings?.power_mode,
        motion_detection_enabled: d.settings?.motion_detection_enabled, stream_setting: d.settings?.stream_setting,
        live_view_disabled: d.settings?.live_view_disabled, video_settings: dev.video_settings,
        snapshot_settings: dev.snapshot_settings, keep_alive: dev.keep_alive_settings,
        motion: Object.fromEntries(Object.entries(d.settings || {}).filter(([k]) => /motion|pir|zone|sensitiv|people|human/i.test(k))),
        motion_device: Object.fromEntries(Object.entries(dev || {}).filter(([k]) => /motion|pir|zone|sensitiv|people|human/i.test(k))) }
    })).then(out => res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(out, null, 1)))
      .catch(e => res.writeHead(500).end(String(e)))
    return
  }
  res.writeHead(404).end()
}).listen(PORT, '127.0.0.1', () => log('ring sidecar on', PORT, 'cams', cams.map(c => `${c.name}${online(c) ? '' : ' (offline)'}`)))

process.on('unhandledRejection', e => log('unhandled', e?.message || e))
