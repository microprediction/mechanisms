// Node runner for the JavaScript mechanism ports.
//
// Usage:
//   node js_parity_runner.js <function> <json-args-array>
//
// Emits one JSON line: {"result": <value>} or {"error": "..."}.
// `test_js_parity.py` calls this once per case and compares `result` against
// the Python reference implementation.

const path = require("path");
const M = require(path.resolve(__dirname, "../docs/js/mechanisms.js"));

const fn = process.argv[2];
let args;
try {
  args = JSON.parse(process.argv[3] || "[]");
} catch (e) {
  process.stdout.write(JSON.stringify({ error: "bad args json" }) + "\n");
  process.exit(2);
}

if (typeof M[fn] !== "function") {
  process.stdout.write(JSON.stringify({ error: `unknown function ${fn}` }) + "\n");
  process.exit(2);
}

try {
  const result = M[fn].apply(null, args);
  process.stdout.write(JSON.stringify({ result }) + "\n");
} catch (e) {
  process.stdout.write(
    JSON.stringify({ error: String((e && e.message) || e) }) + "\n",
  );
}
