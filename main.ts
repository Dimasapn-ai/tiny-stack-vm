// An interpreter for make.fun's tiny stack ISA (v1), written from
// ISA.md: 32-bit wrapping integers, a 256-deep stack, 64 return
// addresses, 256 words of memory, byte input/output, at most
// 1,000,000 steps and 65,536 output bytes. Ten traps end a run; a
// trap's number is the exit code, and output written before it is
// kept.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";

const STACK = 256, CALLS = 64, WORDS = 256;
const STEP_MAX = 1_000_000, OUT_MAX = 65_536;

// Per-opcode facts: immediate length (-1 = no such opcode), pops,
// pushes, for HALT PUSH8 PUSH32 POP DUP SWAP OVER ROT, then JMP JZ
// JNZ CALL RET, then LOAD STORE PRINT PUTC IN EXIT; arithmetic next.
const IMM = new Int8Array(256).fill(-1);
const POP = new Uint8Array(256);
const PUSH = new Uint8Array(256);
const TABLE = [
  [0x00,0,0,0], [0x01,1,0,1], [0x02,4,0,1], [0x03,0,1,0],
  [0x04,0,1,2], [0x05,0,2,2], [0x06,0,2,3], [0x07,0,3,3],
  [0x30,2,0,0], [0x31,2,1,0], [0x32,2,1,0], [0x33,2,0,0], [0x34,0,0,0],
  [0x40,0,1,1], [0x41,0,2,0], [0x50,0,1,0], [0x51,0,1,0], [0x52,0,0,1], [0x53,0,1,0],
];
for (const [op, i, p, q] of TABLE) { IMM[op] = i; POP[op] = p; PUSH[op] = q; }
// ADD..LTU (0x10..0x22, no 0x1f) pop two, push one; NEG/NOT pop one.
for (let op = 0x10; op <= 0x22; op++) if (op !== 0x1f) { IMM[op] = 0; POP[op] = 2; PUSH[op] = 1; }
POP[0x17] = POP[0x1b] = 1;

// The pure binary operations; DIV/MOD are separate (they can trap).
const BINOP = {
  0x10: (a, b) => (a + b) | 0,
  0x11: (a, b) => (a - b) | 0,
  0x12: (a, b) => Math.imul(a, b),
  0x18: (a, b) => a & b,
  0x19: (a, b) => a | b,
  0x1a: (a, b) => a ^ b,
  0x1c: (a, b) => a << (b & 31),
  0x1d: (a, b) => a >> (b & 31),
  0x1e: (a, b) => (a >>> (b & 31)) | 0,
  0x20: (a, b) => (a === b ? 1 : 0),
  0x21: (a, b) => (a < b ? 1 : 0),
  0x22: (a, b) => ((a >>> 0) < (b >>> 0) ? 1 : 0),
};

// Runs one program to its end: the bytes it wrote and its exit code.
function run(prog: Uint8Array, input: Uint8Array): { out: Uint8Array; code: number } {
  const n = prog.length;
  const st = new Int32Array(STACK);
  let sp = 0;
  const cs = new Int32Array(CALLS);
  let csp = 0;
  const mem = new Int32Array(WORDS);
  const buf = new Uint8Array(OUT_MAX);
  let len = 0, ip = 0, pc = 0, steps = 0;
  const end = (code: number) => ({ out: buf.slice(0, len), code });

  for (;;) {
    if (steps === STEP_MAX) return end(10);
    if (pc < 0 || pc >= n) return end(9);
    const op = prog[pc];
    const imm = IMM[op];
    if (imm < 0) return end(5);
    if (pc + imm >= n) return end(9);
    steps++;
    if (sp < POP[op]) return end(2);
    if (sp - POP[op] + PUSH[op] > STACK) return end(3);
    const next = pc + 1 + imm;
    pc = next;
    if (op >= 0x10 && op <= 0x22 && op !== 0x17 && op !== 0x1b) {
      // ADD..LTU: two in, one out; the DIV/MOD family traps on zero.
      const b = st[sp - 1], a = st[sp - 2];
      let r: number;
      if (op >= 0x13 && op <= 0x16) {
        if (b === 0) return end(4);
        r = op === 0x13 ? (a === -2147483648 && b === -1 ? a : (a / b) | 0) // DIV, truncated toward zero
          : op === 0x14 ? (b === -1 ? 0 : (a % b) | 0) // MOD, sign of a
          : op === 0x15 ? Math.floor((a >>> 0) / (b >>> 0)) | 0 // DIVU
          : ((a >>> 0) % (b >>> 0)) | 0; // MODU
      } else {
        r = BINOP[op](a, b);
      }
      st[sp - 2] = r; sp--;
      continue;
    }
    switch (op) {
    case 0x00: return end(0);
    case 0x01: st[sp++] = (prog[next - 1] << 24) >> 24; break; // PUSH8, sign-extended
    case 0x02: // PUSH32, little-endian
      st[sp++] = prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24);
      break;
    case 0x03: sp--; break;
    case 0x04: st[sp] = st[sp - 1]; sp++; break;
    case 0x05: { // SWAP
      const t = st[sp - 1]; st[sp - 1] = st[sp - 2]; st[sp - 2] = t; break;
    }
    case 0x06: st[sp] = st[sp - 2]; sp++; break; // OVER
    case 0x07: { // ROT: ( a b c -- b c a )
      const a = st[sp - 3];
      st[sp - 3] = st[sp - 2]; st[sp - 2] = st[sp - 1]; st[sp - 1] = a;
      break;
    }
    case 0x17: st[sp - 1] = -st[sp - 1] | 0; break; // NEG, wrapping
    case 0x1b: st[sp - 1] = ~st[sp - 1]; break;
    case 0x30: case 0x31: case 0x32: { // JMP, JZ, JNZ: offset from the next instruction
      const rel = ((prog[next - 2] | (prog[next - 1] << 8)) << 16) >> 16;
      let go = op === 0x30;
      if (!go) go = (op === 0x31) === (st[--sp] === 0); // JZ jumps on 0, JNZ on non-0
      if (go) pc = next + rel;
      break;
    }
    case 0x33: // CALL: the immediate is an unsigned address
      if (csp === CALLS) return end(7);
      cs[csp++] = next;
      pc = prog[next - 2] | (prog[next - 1] << 8);
      break;
    case 0x34:
      if (csp === 0) return end(8);
      pc = cs[--csp];
      break;
    case 0x40: { // LOAD
      const addr = st[sp - 1];
      if (addr < 0 || addr >= WORDS) return end(6);
      st[sp - 1] = mem[addr];
      break;
    }
    case 0x41: { // STORE: ( v addr -- )
      const addr = st[sp - 1];
      if (addr < 0 || addr >= WORDS) return end(6);
      mem[addr] = st[sp - 2]; sp -= 2;
      break;
    }
    case 0x50: { // PRINT: decimal ASCII; a write past the limit writes nothing
      const s = String(st[sp - 1]);
      if (len + s.length > OUT_MAX) return end(11);
      for (let i = 0; i < s.length; i++) buf[len++] = s.charCodeAt(i);
      sp--;
      break;
    }
    case 0x51: // PUTC: one byte, the low 8 bits
      if (len + 1 > OUT_MAX) return end(11);
      buf[len++] = st[--sp] & 255;
      break;
    case 0x52: // IN: -1 once the input is used up
      st[sp++] = ip < input.length ? input[ip++] : -1;
      break;
    case 0x53: return end(st[--sp] & 255); // EXIT: the code is the low byte
    }
  }
}

// Answer every instance with its output bytes and exit code; the
// harness names the directories, /input and /output in the sandbox.
const IN = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
for (const id of readdirSync(IN).sort()) {
  const prog = readFileSync(`${IN}/${id}/program.bin`);
  const input = readFileSync(`${IN}/${id}/input.bin`);
  const { out, code } = run(prog, input);
  mkdirSync(`${OUT}/${id}`, { recursive: true });
  writeFileSync(`${OUT}/${id}/stdout.bin`, out);
  writeFileSync(`${OUT}/${id}/exit.txt`, String(code));
}
