// One-time Ring sign-in. Writes only the refresh token to ../secrets/ring-token.json (mode 600); never prints secrets.
//   node auth.mjs you@example.com
// Password: RING_PASSWORD env, else macOS Keychain (service "ring"), else a prompt.
// 2FA code: RING_2FA_CMD (a command that prints the texted code, for unattended setups), else a prompt.
import { RingRestClient } from 'ring-client-api/rest-client'
import { execFileSync, execSync } from 'node:child_process'
import { mkdirSync, writeFileSync, chmodSync } from 'node:fs'
import { createInterface } from 'node:readline/promises'

const email = process.argv[2]
if (!email) { console.error('usage: node auth.mjs you@example.com'); process.exit(1) }
const ask = async (q) => { const rl = createInterface({ input: process.stdin, output: process.stdout }); const a = await rl.question(q); rl.close(); return a.trim() }
let password = process.env.RING_PASSWORD || ''
if (!password && process.platform === 'darwin') {
  try { password = execFileSync('security', ['find-generic-password', '-s', 'ring', '-a', email, '-w'], { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() } catch {}
}
if (!password) password = await ask('Ring password: ')
const client = new RingRestClient({ email, password, controlCenterDisplayName: 'litter-log' })
const save = (t) => {
  const f = new URL('../secrets/ring-token.json', import.meta.url).pathname
  mkdirSync(new URL('../secrets/', import.meta.url).pathname, { recursive: true, mode: 0o700 })
  writeFileSync(f, JSON.stringify({ email, refreshToken: t, saved: new Date().toISOString() }) + '\n', { mode: 0o600 })
  chmodSync(f, 0o600)
  console.log('saved refresh token')
}
try {
  const auth = await client.getAuth()
  save(client.refreshToken || auth.refresh_token)
} catch (e) {
  if (!client.promptFor2fa) { console.error('login failed:', e.message); process.exit(1) }
  console.log('2FA:', client.promptFor2fa)
  let code = ''
  if (process.env.RING_2FA_CMD) {
    try { code = execSync(process.env.RING_2FA_CMD, { timeout: 260000 }).toString().trim() } catch { console.error('RING_2FA_CMD found no code'); process.exit(2) }
  } else {
    code = await ask('Code Ring just sent you: ')
  }
  const auth = await client.getAuth(code)
  save(client.refreshToken || auth.refresh_token)
}
