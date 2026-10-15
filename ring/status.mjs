import { RingApi } from 'ring-client-api'
import { readFileSync, writeFileSync } from 'node:fs'
const tf = new URL('../secrets/ring-token.json', import.meta.url).pathname
const tok = JSON.parse(readFileSync(tf, 'utf8'))
const api = new RingApi({ refreshToken: tok.refreshToken, controlCenterDisplayName: 'catbox' })
api.onRefreshTokenUpdated.subscribe(({ newRefreshToken }) => { tok.refreshToken = newRefreshToken; writeFileSync(tf, JSON.stringify(tok) + '\n', { mode: 0o600 }) })
for (const c of await api.getCameras()) {
  const h = await c.getHealth().catch(e => ({ err: e.message }))
  console.log(c.name, 'conn=', c.data.alerts?.connection, 'wifi=', h.wifi_name ?? h.err, 'rssi=', h.latest_signal_strength, 'last=', h.last_update_time ?? h.updated_at)
}
process.exit(0)
