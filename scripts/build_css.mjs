// Offline build of frontend/static/app.css using the `tailwindcss` v4 package directly.
// Equivalent to: npx @tailwindcss/cli -i frontend/src/input.css -o frontend/static/app.css
import { compile } from "tailwindcss";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const root = path.resolve(import.meta.dirname, "..");
const input = path.join(root, "frontend/src/input.css");
const require = createRequire(import.meta.url);
const twDir = path.dirname(require.resolve("tailwindcss/package.json"));

const compiler = await compile(fs.readFileSync(input, "utf8"), {
  base: path.dirname(input),
  async loadStylesheet(id, base) {
    const file = id === "tailwindcss" ? path.join(twDir, "index.css")
      : id.startsWith("tailwindcss/") ? path.join(twDir, id.slice("tailwindcss/".length) + (id.endsWith(".css") ? "" : ".css"))
      : path.resolve(base, id);
    return { path: file, base: path.dirname(file), content: fs.readFileSync(file, "utf8") };
  },
});

const files = [path.join(root, "frontend/index.html"), path.join(root, "frontend/static/app.js")];
const candidates = new Set();
for (const f of files) for (const tok of fs.readFileSync(f, "utf8").split(/[\s"'`<>=]+/)) {
    if (!tok) continue;
    candidates.add(tok);
    for (const sub of tok.split(/[{}();,]+/)) if (sub) candidates.add(sub); // over-inclusive on purpose; unknown classes are ignored
  }

const css = compiler.build([...candidates]);
fs.writeFileSync(path.join(root, "frontend/static/app.css"), css);
console.log("app.css", css.length, "bytes");
