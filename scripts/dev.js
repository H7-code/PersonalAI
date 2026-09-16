const http = require("http");
const net = require("net");
const { spawn } = require("child_process");

const ROOT = __dirname.replace(/[\\/]scripts$/, "");
const services = [];
let shuttingDown = false;

function log(service, message) {
  console.log(`[${service}] ${message}`);
}

function isPortOpen(port, host) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ port, host });
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.once("error", () => resolve(false));
    socket.setTimeout(500, () => {
      socket.destroy();
      resolve(false);
    });
  });
}

function getJson(url) {
  return new Promise((resolve, reject) => {
    const request = http.get(url, (response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => { body += chunk; });
      response.on("end", () => {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(`HTTP ${response.statusCode}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    request.setTimeout(1000, () => request.destroy(new Error("timeout")));
    request.on("error", reject);
  });
}

function waitFor(check, timeoutMs, description) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const attempt = async () => {
      try {
        if (await check()) {
          resolve();
          return;
        }
      } catch (_) {
        // The service may still be starting.
      }
      if (Date.now() - started >= timeoutMs) {
        reject(new Error(`${description} did not become ready within ${timeoutMs / 1000}s`));
        return;
      }
      setTimeout(attempt, 250);
    };
    attempt();
  });
}

function startProcess(service, command, args, cwd) {
  const child = spawn(command, args, {
    cwd,
    stdio: "inherit",
    windowsHide: false,
  });
  services.push({ service, child });
  child.once("error", (error) => {
    log(service, `failed to start: ${error.message}`);
    shutdown(1);
  });
  child.once("exit", (code, signal) => {
    if (!shuttingDown && code !== 0) {
      log(service, `exited with code ${code ?? "unknown"}${signal ? ` (${signal})` : ""}`);
      shutdown(code || 1);
    }
  });
  return child;
}

async function ensureOllama() {
  const running = await isPortOpen(11434, "127.0.0.1");
  if (!running) {
    log("ollama", "not running; starting local Ollama service");
    startProcess("ollama", "ollama", ["serve"], ROOT);
    await waitFor(() => isPortOpen(11434, "127.0.0.1"), 20000, "Ollama");
  } else {
    log("ollama", "already running; reusing port 11434");
  }

  let tags;
  await waitFor(async () => {
    try {
      tags = await getJson("http://127.0.0.1:11434/api/tags");
      return true;
    } catch (_) {
      return false;
    }
  }, 30000, "Ollama model API");
  const models = (tags.models || []).map((model) => model.name);
  const model = models.find((name) => name === "deepseek-r1:1.5b" || name.startsWith("deepseek-r1:1.5b-"));
  if (!model) {
    throw new Error(`Required local model deepseek-r1:1.5b is not installed. Available models: ${models.join(", ") || "none"}. No download was attempted.`);
  }
  log("ollama", `ready with local model ${model}`);
}

async function ensureBackend() {
  if (await isPortOpen(8000, "127.0.0.1")) {
    log("backend", "already running; reusing port 8000");
    return;
  }
  log("backend", "starting FastAPI on 127.0.0.1:8000");
  startProcess("backend", "python", ["-m", "uvicorn", "src.ui.server:app", "--host", "127.0.0.1", "--port", "8000"], ROOT);
  await waitFor(async () => {
    try {
      const health = await getJson("http://127.0.0.1:8000/health");
      return health.status === "healthy";
    } catch (_) {
      return false;
    }
  }, 30000, "ARIA backend");
  log("backend", "ready at http://127.0.0.1:8000");
}

async function ensureFrontend() {
  if (await isPortOpen(3000, "127.0.0.1")) {
    log("frontend", "already running; reusing port 3000");
    return;
  }
  log("frontend", "starting Next.js on http://127.0.0.1:3000");
  const frontendCommand = process.platform === "win32" ? "cmd.exe" : "npm";
  const frontendArgs = process.platform === "win32"
    ? ["/d", "/s", "/c", "npm run dev"]
    : ["run", "dev"];
  startProcess("frontend", frontendCommand, frontendArgs, `${ROOT}/web`);
  await waitFor(() => isPortOpen(3000, "127.0.0.1"), 30000, "Next.js frontend");
  log("frontend", "ready at http://127.0.0.1:3000");
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  log("aria", "stopping child services");
  for (const { child } of services) {
    if (child.killed) continue;
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
    } else {
      child.kill();
    }
  }
  setTimeout(() => process.exit(code), 500);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

(async () => {
  try {
    await ensureOllama();
    await ensureBackend();
    await ensureFrontend();
    log("aria", "ready at http://127.0.0.1:3000");
  } catch (error) {
    log("aria", error.message);
    shutdown(1);
  }
})();
