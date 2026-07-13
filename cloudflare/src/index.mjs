const API_URL =
  "https://api.github.com/repos/kingso0125/shanghai-earth-wallpaper/actions/workflows/hourly-watchdog.yml/dispatches";

export async function dispatchWatchdog(token, fetchImpl = fetch) {
  if (!token) {
    throw new Error("GITHUB_TOKEN is not configured");
  }

  const response = await fetchImpl(API_URL, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "User-Agent": "shanghai-earth-wallpaper-trigger",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: JSON.stringify({ ref: "main" }),
  });

  if (response.status !== 204) {
    const detail = (await response.text()).slice(0, 500);
    throw new Error(`GitHub dispatch failed (${response.status}): ${detail}`);
  }
}

export default {
  async scheduled(_controller, env, context) {
    context.waitUntil(dispatchWatchdog(env.GITHUB_TOKEN));
  },

  async fetch(request) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ ok: true, service: "shanghai-earth-wallpaper-trigger" });
    }
    return new Response("Not found", { status: 404 });
  },
};
