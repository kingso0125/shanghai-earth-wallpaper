import assert from "node:assert/strict";
import test from "node:test";

import worker, { dispatchWatchdog } from "../src/index.mjs";


test("dispatches the watchdog workflow on main without exposing the token", async () => {
  let request;
  await dispatchWatchdog("secret-token", async (url, options) => {
    request = { url, options };
    return new Response(null, { status: 204 });
  });

  assert.match(request.url, /hourly-watchdog\.yml\/dispatches$/);
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers.Authorization, "Bearer secret-token");
  assert.deepEqual(JSON.parse(request.options.body), { ref: "main" });
});

test("fails closed when the GitHub token is missing", async () => {
  await assert.rejects(() => dispatchWatchdog("", async () => new Response()), {
    message: "GITHUB_TOKEN is not configured",
  });
});

test("reports GitHub API failures", async () => {
  await assert.rejects(
    () => dispatchWatchdog("token", async () => new Response("forbidden", { status: 403 })),
    /GitHub dispatch failed \(403\): forbidden/,
  );
});

test("health endpoint exposes no credentials", async () => {
  const response = await worker.fetch(new Request("https://worker.example/health"));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    ok: true,
    service: "shanghai-earth-wallpaper-trigger",
  });
});
