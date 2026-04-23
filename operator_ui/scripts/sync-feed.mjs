import { cpSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const uiRoot = resolve(scriptDir, "..");
const repoRoot = resolve(uiRoot, "..");
const dashboardDir = resolve(repoRoot, "candidate_registry", "dashboard");
const farmDashboardDir = resolve(repoRoot, "candidate_registry", "farm_dashboard");
const args = process.argv.slice(2);
const farmMode = args.includes("--farm");
const positional = args.filter((arg) => arg !== "--farm");
const target = resolve(
  uiRoot,
  "public",
  "data",
  farmMode ? "farm-dashboard-feed.json" : "dashboard-feed.json",
);

function latestDashboardFeed(dir) {
  const entries = readdirSync(dir)
    .filter((entry) => entry.endsWith(".json"))
    .map((entry) => {
      const path = join(dir, entry);
      return {
        entry,
        path,
        mtimeMs: statSync(path).mtimeMs,
      };
    })
    .sort((left, right) => right.mtimeMs - left.mtimeMs);
  if (entries.length === 0) {
    throw new Error(`No dashboard feeds found in ${dir}`);
  }
  return entries[0].path;
}

const requestedSource = positional[0]
  ? (isAbsolute(positional[0]) ? positional[0] : resolve(process.cwd(), positional[0]))
  : null;
const source = requestedSource ?? latestDashboardFeed(farmMode ? farmDashboardDir : dashboardDir);

if (!existsSync(source)) {
  throw new Error(`Dashboard feed not found: ${source}`);
}

mkdirSync(dirname(target), { recursive: true });
cpSync(source, target);
console.log(`Copied ${source} -> ${target}`);
