---
name: served-frontend-artifact-verification
description: Prove that a browser is running the frontend bundle you just built (not a stale copy) before debugging UI behaviour; includes robust SPA E2E harness patterns
trigger: When you build a front-end bundle and then verify it through a browser/E2E test against a running server, especially with static-file middleware (whitenoise/nginx/manifest storage) in front of it
---

## Why
A stale or half-collected artifact produces symptoms that look exactly like application bugs
(empty form state, "app never booted", mismatched JS, no console errors, requests that hang).
Debugging logic in that state wastes entire work cycles. Always prove the artifact identity first.

## Step 1 — Prove artifact identity (before any UI assertion)
1. Note the exact build output path (e.g. `<app>/static/frontend/<locale>/main.js`).
2. Fetch the served URL and compare bytes, not just status code:
   `curl -s -o /tmp/served.bin -D /tmp/h.txt <url> && cmp /tmp/served.bin <build-output-file>`
3. Check headers: `Content-Length` must equal the on-disk size; `Last-Modified`/`ETag` must be the
   timestamp/hash of the fresh build.
4. Repeat with `Accept-Encoding: gzip` and `br` if the server can serve precompressed variants, and
   confirm the compressed payload decompresses to the same bytes (`gunzip`/`brotli -d` + `cmp`).
5. Record the served size/hash in your notes and re-check it after every rebuild; sizes changing by a
   few hundred bytes between runs is normal, a size that stops changing after an edit is not.

## Step 2 — Refresh the served copy correctly
1. Static-file middleware often caches file metadata (size/mtime/etag) at process start: after
   collecting static files you must restart the app process, not just re-run the collector.
2. If the collector reuses pre-existing precompressed siblings (`.gz`, `.br`) it can leave a pair whose
   lengths disagree with the uncompressed file; the browser then aborts with
   `ERR_CONTENT_LENGTH_MISMATCH` and the app never boots. Delete the served static subtree
   (e.g. `rm -rf <static-root>/frontend`) before collecting.
3. Some servers also require the process restart after replacing files by the same name.

## Step 3 — Robust SPA E2E harness patterns
- Wait for specific elements: `getByRole(...).waitFor({state:'visible'})`; do not rely on
  `networkidle` for apps that poll — it never settles and times out.
- Login: submit, then poll for the URL change with `page.waitForFunction(() => !location.pathname.includes('/login'))`;
  some driver versions silently never match a `waitForURL(predicate)`.
- Use `waitUntil: 'commit'` for navigations you immediately follow with an element wait.
- Prefer `page.on('response')`/`request` capture around the action that triggers the write, and assert
  the status code, instead of asserting on rendered text.
- Before blaming the app, dump `document.querySelector('<app-root>').innerHTML.length` and the
  `performance.getEntriesByType('resource')` list: an empty app root + missing bundle entry means the
  artifact never executed (Step 1 failure), not a logic bug.
- Clean up records your E2E test created, and re-read the collection counts afterwards so the shared
  environment is left pristine.

## Step 4 — Distinguish flake from failure
Re-run a failing assertion once unchanged. If it flips, fix the wait/locator, not the product code.
