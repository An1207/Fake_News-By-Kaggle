"""CPU ONNX execution with the original tokenizer and validated frozen weights."""
import gc
import json
from pathlib import Path
from threading import Lock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class OnnxEuroBertEncoder:
    def __init__(self, spec: dict):
        from tokenizers import Tokenizer
        from .transformer import sha256
        self._inference_lock = Lock()

        path = ROOT / spec["onnx_path"]
        if not path.is_file() or sha256(path) != spec["onnx_sha256"]:
            raise RuntimeError("Validated ONNX encoder missing or changed; run news-ai export-encoder first.")
        snapshot = ROOT / ".hf-cache" / "models--EuroBERT--EuroBERT-210m" / "snapshots" / spec["revision"]
        self.tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        config = json.loads((snapshot / "config.json").read_text())
        tokenizer_config = json.loads((snapshot / "tokenizer_config.json").read_text())
        self.tokenizer.enable_truncation(max_length=spec["max_length"], direction=tokenizer_config.get("truncation_side", "right"))
        pad_token = tokenizer_config.get("pad_token", config["pad_token"])
        if isinstance(pad_token, dict):
            pad_token = pad_token["content"]
        pad_id = self.tokenizer.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError("Original tokenizer padding token is absent from the vocabulary.")
        self.tokenizer.enable_padding(direction=tokenizer_config.get("padding_side", "right"),
                                      pad_id=pad_id, pad_token=pad_token)
        self.backend = spec.get("backend", "onnx")
        if self.backend == "openvino":
            import openvino as ov
            core = ov.Core()
            requested = spec.get("device", "AUTO")
            self.device = ("GPU" if "GPU" in core.available_devices else "CPU") if requested == "AUTO" else requested
            if self.device not in core.available_devices:
                raise RuntimeError(f"Requested encoder device {self.device} unavailable. Available: {core.available_devices}")
            model = core.read_model(str(path))
            model.reshape({"input_ids": [-1, spec["max_length"]], "attention_mask": [-1, spec["max_length"]]})
            self.tokenizer.enable_padding(length=spec["max_length"], direction=tokenizer_config.get("padding_side", "right"),
                                          pad_id=pad_id, pad_token=pad_token)
            cache = ROOT / ".hf-cache" / "openvino_cache"
            cache.mkdir(parents=True, exist_ok=True)
            options = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(cache)}
            if self.device == "CPU":
                options.update(INFERENCE_NUM_THREADS=spec["threads"], NUM_STREAMS=1, INFERENCE_PRECISION_HINT="f32")
            else:
                options.update(INFERENCE_PRECISION_HINT=spec.get("inference_precision", "f16"))
            self.session = core.compile_model(model, self.device, options)
        else:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = spec["threads"]
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.dimension = config["hidden_size"]

    def tokenize(self, texts: list[str]) -> dict:
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        return {"input_ids": np.array([row.ids for row in encoded], dtype=np.int64),
                "attention_mask": np.array([row.attention_mask for row in encoded], dtype=np.int64)}

    def encode(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenize(texts)
        if self.backend == "openvino":
            # CompiledModel.__call__ uses one internal infer request; copy its
            # output before another Streamlit session can reuse that buffer.
            with self._inference_lock:
                output = np.array(self.session(inputs)[self.session.output(0)], copy=True)
        else:
            output = self.session.run(["embeddings"], inputs)[0]
        if not np.isfinite(output).all():
            raise RuntimeError("Non-finite ONNX encoder output.")
        return output


def export_encoder(revision: str, output_dir: Path) -> dict:
    """Native float32 export with tokenizer/output parity checks and metadata."""
    from .transformer import EuroBertEncoder, sha256
    import torch
    import onnxruntime as ort

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {"model_id": "EuroBERT/EuroBERT-210m", "revision": revision, "max_length": 128,
            "pooling": "masked-mean-l2", "quantization": "none", "threads": 2}
    encoder = EuroBertEncoder(spec)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            mask4 = (1 - attention_mask[:, None, None, :]).float() * torch.finfo(torch.float32).min
            mask4 = mask4.expand(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
            hidden = self.model(input_ids=input_ids, attention_mask=mask4, use_cache=False).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    wrapper = ExportWrapper(encoder.model).eval()
    sample_texts = ["The fictional city council approved a budget for schools.",
                    "A fictional newspaper published a report on schools and hospitals. " * 50]
    sample = encoder.tokenizer(sample_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        expected = encoder.encode(sample_texts)
    np.savez(output_dir / "validation_reference.npz", expected=expected, texts=np.array(sample_texts))
    original = output_dir / "encoder.fp32.onnx"
    print("Exporting original float32 EuroBERT...", flush=True)
    with torch.inference_mode():
        torch.onnx.export(wrapper, (sample["input_ids"], sample["attention_mask"]), original,
                          input_names=["input_ids", "attention_mask"], output_names=["embeddings"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                        "attention_mask": {0: "batch", 1: "sequence"}, "embeddings": {0: "batch"}},
                          opset_version=18, dynamo=False)
    del wrapper, encoder
    gc.collect()
    spec.update(backend="onnx", quantization="none",
                onnx_path=original.resolve().relative_to(ROOT).as_posix(), onnx_sha256=sha256(original))
    runtime = OnnxEuroBertEncoder(spec)
    actual_inputs = runtime.tokenize(sample_texts)
    for key in ["input_ids", "attention_mask"]:
        if not np.array_equal(actual_inputs[key], sample[key].numpy()):
            raise RuntimeError("ONNX tokenizer differs from the reference tokenizer.")
    output = runtime.encode(sample_texts)
    cosine = (output * expected).sum(1)
    if float(cosine.min()) < .999:
        raise RuntimeError(f"Encoder parity failed: minimum cosine={cosine.min():.6f}")
    metadata = {"encoder": spec, "validation_min_cosine": float(cosine.min()),
                "tokenizer_parity": True, "onnxruntime_version": ort.__version__, "precision": "float32"}
    (output_dir / "encoder_export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Validated ONNX encoder: cosine >= {cosine.min():.6f}", flush=True)
    return metadata
