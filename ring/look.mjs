import { RingApi } from 'ring-client-api'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
const tf = new URL('../secrets/ring-token.json', import.meta.url).pathname
const tok = JSON.parse(readFileSync(tf, 'utf8'))
const api = new RingApi({ refreshToken: tok.refreshToken, controlCenterDisplayName: 'catbox', ffmpegPath: process.env.FFMPEG || 'ffmpeg' })
api.onRefreshTokenUpdated.subscribe(({ newRefreshToken }) => { tok.refreshToken = newRefreshToken; writeFileSync(tf, JSON.stringify(tok) + '\n', { mode: 0o600 }) })
const out = process.argv[2] || '/tmp/ringlook'; mkdirSync(out, { recursive: true })
const cams = await api.getCameras()
for (const c of cams) {
  const d = c.data
  console.log(JSON.stringify({ id: c.id, name: c.name, kind: d.kind, battery: c.hasBattery, batteryLevel: c.batteryLevel, wired: !c.hasBattery, online: !d.alerts?.connection || d.alerts.connection, features: d.features && Object.keys(d.features).filter(k => d.features[k] === true), motionDetection: d.settings?.motion_detection_enabled, subscribed: c.isSubscribed ?? d.subscribed }))
  try { const img = await c.getSnapshot(); writeFileSync(`${out}/${c.id}-snap.jpg`, img); console.log('  snapshot ok', img.length) }
  catch (e) { console.log('  snapshot failed:', e.message) }
  try { await c.recordToFile(`${out}/${c.id}-clip.mp4`, 6); console.log('  clip ok') } catch (e) { console.log('  clip failed:', e.message) }
}
process.exit(0)
