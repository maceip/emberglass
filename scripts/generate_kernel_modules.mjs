/*
 *   ,;
 *  \@@#\:          :/.        .:;;:
 * _@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\@@@*;,\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
 *      ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */

import { Template } from '@huggingface/jinja';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));
const asciiHeader = `/*
 *   ,;
 *  \\@@#\\:          :/.        .:;;:
 * _@@@@@@#+\\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\\@@@*;,\\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\\@@@@+.
 *      ;\\/:,  -@@@@;|@@@\\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \\##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */`;

const modules = [
  {
    outFile: 'src/qwgpu/kernels.js',
    templateDir: 'src/qwgpu/templates/forward',
    description: 'Forward/inference WGSL kernels.',
  },
  {
    outFile: 'src/qwgpu/backward_kernels.js',
    templateDir: 'src/qwgpu/templates/backward',
    description: 'Backward/training WGSL kernels.',
  },
];

const parametricKernels = new Map([
  ['GEMV4_W4A8', { args: 'hasDP4a, wgSize = 64', hasWgSize: true }],
  ['GEMV4_ADD_W4A8', { args: 'hasDP4a, wgSize = 64', hasWgSize: true }],
  ['QKV_GEMV4_W4A8', { args: 'hasDP4a, wgSize = 64', hasWgSize: true }],
  ['GATE_UP_SILU_GEMV4_W4A8', { args: 'hasDP4a, wgSize = 64', hasWgSize: true }],
  ['GEMM4_W4A8', { args: 'hasDP4a', hasWgSize: false }],
  ['GEMM4_ADD_T_W4A8', { args: 'hasDP4a', hasWgSize: false }],
]);

function renderTemplate(source, context = {}) {
  return new Template(source).render(context);
}

function jsString(source) {
  return `\`${escapeTemplateLiteral(source)}\``;
}

function escapeTemplateLiteral(source) {
  return source.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${');
}

function templatePath(dir, index, name) {
  const prefix = `${String(index + 1).padStart(3, '0')}-${name}.wgsl.jinja`;
  const path = join(rootDir, dir, prefix);
  if (!existsSync(path)) throw new Error(`missing kernel template ${path}`);
  return path;
}

function readManifest(dir) {
  const manifestPath = join(rootDir, dir, 'manifest.json');
  if (!existsSync(manifestPath)) {
    throw new Error(`missing kernel template manifest ${manifestPath}`);
  }
  const names = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (!Array.isArray(names) || names.some((name) => typeof name !== 'string')) {
    throw new Error(`bad kernel template manifest ${manifestPath}`);
  }
  return names;
}

function ensureManifestMatchesFiles(dir, names) {
  const expected = new Set(names.map((name, index) => `${String(index + 1).padStart(3, '0')}-${name}.wgsl.jinja`));
  const actual = readdirSync(join(rootDir, dir)).filter((name) => name.endsWith('.wgsl.jinja'));
  for (const file of actual) {
    if (!expected.has(file)) throw new Error(`template file is not listed in manifest: ${join(dir, file)}`);
  }
}

function renderStaticExport(name, source) {
  const rendered = renderTemplate(source);
  return `export const ${name} = ${jsString(rendered)};`;
}

function renderParametricExport(name, source, spec) {
  const compiled = compileParametricSource(source, spec);
  return `export const ${name} = (${spec.args}) => \`${compiled}\`;`;
}

function compileParametricSource(source, spec) {
  const token = /({%\s*if\s+not\s+hasDP4a\s*%}[\s\S]*?{%\s*endif\s*%}|{%\s*if\s+hasDP4a\s*%}[\s\S]*?{%\s*endif\s*%}|{{\s*wgSize\s*}})/g;
  let out = '';
  let pos = 0;
  for (const match of source.matchAll(token)) {
    out += escapeTemplateLiteral(source.slice(pos, match.index));
    const text = match[0];
    if (text.startsWith('{{')) {
      if (!spec.hasWgSize) throw new Error(`unexpected wgSize placeholder in non-wgSize kernel`);
      out += '${wgSize}';
    } else {
      const negated = /{%\s*if\s+not\s+hasDP4a\s*%}/.test(text);
      const body = text
        .replace(/^{%\s*if\s+(not\s+)?hasDP4a\s*%}/, '')
        .replace(/{%\s*endif\s*%}$/, '');
      out += negated
        ? "${hasDP4a ? '' : `" + escapeTemplateLiteral(body) + "`}"
        : "${hasDP4a ? `" + escapeTemplateLiteral(body) + "` : ''}";
    }
    pos = match.index + text.length;
  }
  out += escapeTemplateLiteral(source.slice(pos));
  if (/{%|{{/.test(out)) throw new Error(`unsupported Jinja syntax left in ${source.slice(0, 40)}`);
  return out;
}

function renderModule(config) {
  const names = readManifest(config.templateDir);
  ensureManifestMatchesFiles(config.templateDir, names);
  const exports = names.map((name, index) => {
    const source = readFileSync(templatePath(config.templateDir, index, name), 'utf8');
    const spec = parametricKernels.get(name);
    return spec ? renderParametricExport(name, source, spec) : renderStaticExport(name, source);
  });

  return [
    asciiHeader,
    '',
    `// Generated by scripts/generate_kernel_modules.mjs from ${config.templateDir}/*.wgsl.jinja.`,
    '// Edit the templates, not this file.',
    `// ${config.description}`,
    '',
    ...exports,
    '',
  ].join('\n');
}

export function generateKernelModules({ check = false } = {}) {
  const stale = [];
  for (const config of modules) {
    const outPath = join(rootDir, config.outFile);
    const next = renderModule(config);
    const current = existsSync(outPath) ? readFileSync(outPath, 'utf8') : null;
    if (check) {
      if (current !== next) stale.push(config.outFile);
    } else if (current !== next) {
      writeFileSync(outPath, next, 'utf8');
      console.log(`generated ${config.outFile}`);
    }
  }
  if (check && stale.length) {
    throw new Error(`kernel modules are stale; run npm run kernels:generate (${stale.join(', ')})`);
  }
}

const isCli = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isCli) {
  generateKernelModules({ check: process.argv.includes('--check') });
}
