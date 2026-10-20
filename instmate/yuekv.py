"""8-bit KV cache for the YuE2 MLX port (E3), patched in at run time; the port's files are not edited.

The AR sampling caches (cond / uncond) and the NAR prefill k/v are stored as 8-bit group-64 affine arrays and
dequantized one layer at a time.  Measured on the fixed 106-bar song: peak 6.39 -> 5.98 GB on an M-series Mac,
6.72 -> 6.53 GB on an RTX 3070 (MLX-CUDA); blind listening found no loss against the plain cache.
"""
import mlx.core as mx

BITS, GROUP = 8, 64


def _q(a):
    return mx.quantize(a, group_size=GROUP, bits=BITS)


def _dq(t):
    return mx.dequantize(*t, group_size=GROUP, bits=BITS)


class QuantizedKVCache:
    """Same interface as the port's KVCache: append-only, grown in 256-slot steps."""

    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def _grow(self, old, new, prev):
        if old is None:
            return new
        if prev % self.step:
            old = [x[..., :prev, :] for x in old]
        return [mx.concatenate([a, b], axis=2) for a, b in zip(old, new)]

    def update(self, k, v):
        prev = self.offset
        B, H, T, D = k.shape
        if self.keys is None or prev + T > self.keys[0].shape[2]:
            n = ((T + self.step - 1) // self.step) * self.step
            self.keys = self._grow(self.keys, _q(mx.zeros((B, H, n, D), k.dtype)), prev)
            self.values = self._grow(self.values, _q(mx.zeros((B, H, n, D), v.dtype)), prev)
        self.offset += T
        for store, new in ((self.keys, _q(k)), (self.values, _q(v))):
            for part, value in zip(store, new):
                part[..., prev:self.offset, :] = value
        cut = lambda t: _dq([x[..., :self.offset, :] for x in t])
        return cut(self.keys), cut(self.values)


def patch(model_module, generate_module=None):
    """Swap the cache into an imported yue2_model (and generate, which holds its own KVCache name)."""
    if getattr(model_module.DecoderLayer, "_instmate_kv8", False):
        raise RuntimeError("the 8-bit KV cache is already patched in")
    plain_nar = model_module.DecoderLayer.nar

    def nar_prefill(self, ar_tokens):
        x = self.model.embed_tokens(mx.array([ar_tokens]))
        cache = []
        for layer in self.model.layers:
            x, (k, v) = layer.ar_prefill_kv(x)
            cache.append((_q(k), _q(v)))
            mx.eval(x, cache[-1])
        return cache

    def nar(self, x, ar_k, ar_v, offset):
        if isinstance(ar_k, (tuple, list)):          # mx.quantize returns a list
            ar_k, ar_v = _dq(ar_k), _dq(ar_v)
        return plain_nar(self, x, ar_k, ar_v, offset)

    model_module.KVCache = QuantizedKVCache
    model_module.Yue2Model.nar_prefill = nar_prefill
    model_module.DecoderLayer.nar = nar
    model_module.DecoderLayer._instmate_kv8 = True
    if generate_module is not None:
        generate_module.KVCache = QuantizedKVCache
