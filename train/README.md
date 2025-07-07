EEG Signals (88.56% accuracy)
    ↓
Digit Predictions (0-9)
    ↓
Template Captions ("A handwritten digit X")
    ↓
Pure CLIP (Frozen) → Rich 512-dim Embeddings (similarity 3.7+)
    ↓
CLIP Decoder → Reconstructed 28x28 MNIST Images ✅

🧠 Pipeline Flow:
EEG → Digit Prediction
EEGInferencer loads trained PyTorch EEG model
Predicts digit (0-9) from EEG signals
Digit → Caption
DigitCaptionGenerator creates template captions
Format: "A handwritten digit {number}"
Label → MNIST Image
MNISTLoader retrieves corresponding MNIST images
Supports balanced batching and random sampling
CLIP Training
CLIPMNISTModel with ViT-B/32 + Text Encoder
Contrastive learning for image-text alignment
Output: 512-dim aligned embeddings
🚀 Key Features:
ViT-B/32: Adapted for 28x28 MNIST (7x7 patches)
Text Encoder: Transformer-based with simple tokenizer
Contrastive Loss: InfoNCE for cross-modal alignment
512-dim Embeddings: CLIP-style normalized features
Zero-shot Capability: Image↔text retrieval


✅ Caption Generator Test Results:
📝 Single Digit Tests:
Semua digit 0-9 berhasil dikonversi ke caption
Format: "A handwritten digit {number}"
Konsisten dan sesuai template
📦 Batch Test:
Batch processing berfungsi dengan baik
Input list digit → output list captions
🔢 Tensor Input Test:
Support PyTorch tensor input ✅
Tensor [3, 7, 1, 9] → captions yang sesuai
🔄 Alternative Templates Test:
5 template alternatif tersedia:
"The digit five"
"Number five"
"A written five"
"Handwritten five"
"The number five"

MNIST LOADER

EEG Signal → Digit Prediction → PROTOTYPE Image → CLIP Embedding
                                      ↓
                              ALWAYS SAME IMAGE
                                      ↓
                           Consistent Reconstruction!


EEG Signal → EEG Inferencer → Digit Prediction → Caption Generator → MNIST Loader → CLIP Training
