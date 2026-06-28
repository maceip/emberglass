/*
 * Static file server with HTTP Range support for multi-GB safetensors shards.
 * Used by real-browser evidence runners (benchmark, training, f16 parity).
 */
import { createServer } from 'node:http';
import { open, readFile, stat } from 'node:fs/promises';
import { extname, isAbsolute, relative, resolve } from 'node:path';

const types = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.webp', 'image/webp'],
]);

export function createRangeServer(root) {
  const server = createServer(async (req, res) => {
    try {
      const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://127.0.0.1').pathname);
      const file = resolve(root, pathname === '/' ? 'index.html' : `.${pathname}`);
      const rel = relative(root, file);
      if (rel.startsWith('..') || isAbsolute(rel)) {
        res.writeHead(403);
        res.end('forbidden');
        return;
      }
      const contentType = types.get(extname(file)) ?? 'application/octet-stream';
      const range = req.headers.range;
      if (range) {
        const m = /^bytes=(\d*)-(\d*)$/.exec(range);
        if (!m) {
          res.writeHead(416);
          res.end('invalid range');
          return;
        }
        const { size } = await stat(file);
        const start = m[1] === '' ? Math.max(0, size - Number(m[2] || 0)) : Number(m[1]);
        const end = m[2] === '' ? size - 1 : Math.min(size - 1, Number(m[2]));
        if (start >= size || end < start) {
          res.writeHead(416, { 'content-range': `bytes */${size}` });
          res.end();
          return;
        }
        const len = end - start + 1;
        const fh = await open(file, 'r');
        try {
          const body = Buffer.alloc(len);
          await fh.read(body, 0, len, start);
          res.writeHead(206, {
            'content-type': contentType,
            'content-range': `bytes ${start}-${end}/${size}`,
            'content-length': len,
            'accept-ranges': 'bytes',
          });
          res.end(body);
        } finally {
          await fh.close();
        }
        return;
      }
      const body = await readFile(file);
      res.writeHead(200, { 'content-type': contentType, 'accept-ranges': 'bytes' });
      res.end(body);
    } catch (err) {
      res.writeHead(err.code === 'ENOENT' ? 404 : 500);
      res.end(String(err.message ?? err));
    }
  });
  return server;
}

export function listen(server, host = '127.0.0.1', port = 0) {
  return new Promise((resolveListen) => server.listen(port, host, () => resolveListen(server.address())));
}
