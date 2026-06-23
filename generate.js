// Prefill + incremental decode loop on top of QwenModel, with a transformers.js
// tokenizer. forward() disposes the KV cache it's handed and returns a fresh one,
// so we just thread r.kvCaches forward (no manual double-dispose).
import * as tf from '@tensorflow/tfjs-core';

async function sampleToken(logits, temperature) {
  // logits: [1,1,vocab]. Async readback so the main thread yields between tokens.
  const flat = tf.reshape(logits, [-1]);
  let id;
  if (!temperature || temperature <= 0) {
    const am = tf.argMax(flat);
    id = (await am.data())[0];
    am.dispose();
  } else {
    const probs = tf.softmax(tf.div(flat, temperature));
    const p = await probs.data();
    probs.dispose();
    let r = Math.random(), c = 0; id = p.length - 1;
    for (let i = 0; i < p.length; i++) { c += p[i]; if (r <= c) { id = i; break; } }
  }
  flat.dispose();
  return id;
}

/**
 * @param model      QwenModel (with weights + active LoRA set via setLora)
 * @param tokenizer  transformers.js tokenizer
 * @param messages   [{role,content}, ...]  (chat-templated with Qwen <think>)
 * @param opts       {maxTokens, temperature, stopIds}
 * @param onToken    (deltaText, tokenId) => void   (streaming)
 * @returns full generated text
 */
export async function generate(model, tokenizer, messages, opts = {}, onToken = null) {
  const { maxTokens = 1024, temperature = 0.0, stopIds = [151645, 151643] } = opts;
  // Our tokenizer_config has no chat_template, so build Qwen2.5 ChatML manually.
  let promptText;
  try {
    promptText = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
  } catch {
    promptText = messages.map(m => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join('') + '<|im_start|>assistant\n';
  }
  const ids = tokenizer.encode(promptText); // number[] (special tokens mapped)

  // ---- prefill ----
  const idsT = tf.tensor2d([ids], [1, ids.length], 'int32');
  const emb = model.embed(idsT);
  let { logits, kvCaches } = model.forward(emb, 0, null);
  emb.dispose(); idsT.dispose();
  let pos = ids.length;
  let next = await sampleToken(logits, temperature);
  logits.dispose();

  // ---- decode ----
  const generated = [];
  let prevText = '';
  for (let step = 0; step < maxTokens; step++) {
    if (stopIds.includes(next)) break;
    generated.push(next);
    // decode the whole tail for correct multi-byte merges, emit the delta
    const text = tokenizer.decode(generated, { skip_special_tokens: true });
    const delta = text.slice(prevText.length);
    prevText = text;
    if (delta && onToken) onToken(delta, next);

    const tokT = tf.tensor2d([[next]], [1, 1], 'int32');
    const e = model.embed(tokT);
    const r = model.forward(e, pos, kvCaches); // disposes old kvCaches internally
    e.dispose(); tokT.dispose();
    kvCaches = r.kvCaches;
    pos++;
    next = await sampleToken(r.logits, temperature);
    r.logits.dispose();
    if (step % 8 === 0) await tf.nextFrame?.();
  }
  model.disposeKV(kvCaches);
  return prevText;
}
