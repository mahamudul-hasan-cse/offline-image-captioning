"""
export_model.py
---------------
Exports the Salesforce BLIP-base model to ONNX format
for deployment on Android via ONNX Runtime.

Usage:
    python scripts/export_model.py

Output:
    models/blip_vision_encoder.onnx
    models/blip_vision_encoder.onnx.data
    models/blip_text_decoder.onnx
    models/blip_text_decoder.onnx.data
    models/tokenizer.json
    models/vocab.txt
"""

import os
import torch
import shutil
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "Salesforce/blip-image-captioning-base"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"[1/5] Loading BLIP-base model from Hugging Face: {MODEL_ID}")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model     = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()

# ── Export Vision Encoder ─────────────────────────────────────────────────────
print("[2/5] Exporting vision encoder to ONNX...")

class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.vision_model

    def forward(self, pixel_values):
        return self.encoder(pixel_values).last_hidden_state

vision_wrapper = VisionEncoderWrapper(model)
dummy_image    = torch.randn(1, 3, 384, 384)

torch.onnx.export(
    vision_wrapper,
    dummy_image,
    str(OUTPUT_DIR / "blip_vision_encoder.onnx"),
    input_names  = ["pixel_values"],
    output_names = ["image_features"],
    dynamic_axes = {"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
    opset_version = 17,
)
print("    ✅ Vision encoder exported.")

# ── Export Text Decoder ───────────────────────────────────────────────────────
print("[3/5] Exporting text decoder to ONNX...")

class TextDecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.text_decoder

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        return self.decoder(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            encoder_hidden_states= encoder_hidden_states,
        ).logits

text_wrapper          = TextDecoderWrapper(model)
seq_len               = 10
dummy_input_ids       = torch.ones(1, seq_len, dtype=torch.long)
dummy_attention_mask  = torch.ones(1, seq_len, dtype=torch.long)
dummy_encoder_states  = torch.randn(1, 577, 768)

torch.onnx.export(
    text_wrapper,
    (dummy_input_ids, dummy_attention_mask, dummy_encoder_states),
    str(OUTPUT_DIR / "blip_text_decoder.onnx"),
    input_names  = ["input_ids", "attention_mask", "encoder_hidden_states"],
    output_names = ["logits"],
    dynamic_axes = {
        "input_ids":             {0: "batch", 1: "seq_len"},
        "attention_mask":        {0: "batch", 1: "seq_len"},
        "encoder_hidden_states": {0: "batch"},
        "logits":                {0: "batch", 1: "seq_len"},
    },
    opset_version = 17,
)
print("    ✅ Text decoder exported.")

# ── Save Tokenizer Files ──────────────────────────────────────────────────────
print("[4/5] Saving tokenizer files...")
processor.tokenizer.save_pretrained(str(OUTPUT_DIR))

# Copy vocab.txt explicitly
vocab_src = OUTPUT_DIR / "vocab.txt"
if not vocab_src.exists():
    shutil.copy(
        Path(processor.tokenizer.vocab_file),
        vocab_src
    )
print("    ✅ Tokenizer files saved.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n[5/5] Export complete! Generated files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"    {f.name:<45} {size_mb:>8.1f} MB")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Next step: push models to your Android device
 Run:  adb push models/ /sdcard/Android/data/com.example.offlinecaptioning/files/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
