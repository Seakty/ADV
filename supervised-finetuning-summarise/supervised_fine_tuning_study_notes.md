# Supervised Fine-Tuning (SFT) for LLMs & Vision-Language Models

A practical summary of fine-tuning large language models and multimodal vision models using Hugging Face, PEFT (LoRA), and Unsloth.

---

## 1. What is Fine-Tuning?

Pretrained models (like LLaMA, Mistral, or Qwen) know general language rules because they were trained on massive web datasets. However, they are generalists, not specialists.

**Fine-tuning** takes that pretrained model and continues training it on a smaller, curated, task-specific dataset. Instead of spending millions training from scratch, we adapt existing model weights so it learns:

- A specific tone or output format (e.g., JSON extraction).
- Domain knowledge (e.g., medical, legal, robotics).
- Specialized capabilities (e.g., visual bounding boxes, dialogue).

```
[Pretrained LLM] + [Task-Specific Dataset] ---> Fine-Tuning ---> [Specialized Final Model]
```

---

## 2. Model & Preprocessor Setup

### Choosing the Right Model

Before writing code, consider:

1. **Task Alignment:** Does the model match the task (text-only, text-to-text, or image-text)?
2. **Model Size vs. Hardware:** Can your GPU handle the parameter count?
3. **Architecture & Ecosystem:** Is the model well-supported by Hugging Face / community tools?

### Loading Models: Specific vs. AutoModel

Hugging Face offers two main ways to load models:

- **Model-Specific Loaders:** Hardcoded to one architecture (e.g., `BertForSequenceClassification`). If you swap models later, you have to rewrite your imports.
- **AutoModel Loaders (Recommended):** Automatically picks the right architecture based on the model ID.

| AutoModel Variant                    | Purpose                                                            |
| :----------------------------------- | :----------------------------------------------------------------- |
| `AutoModelForCausalLM`               | Standard text generation (GPT, LLaMA)                              |
| `AutoModelForSeq2SeqLM`              | Encoder-decoder tasks like translation or summarization (T5, BART) |
| `AutoModelForSequenceClassification` | Text classification / sentiment analysis                           |
| `AutoModelForTokenClassification`    | Named entity recognition (NER)                                     |
| `AutoModelForImageTextToText`        | Multimodal / Vision-Language models (Qwen2-VL)                     |

### Processors & Tokenizers

For text models, we use a `Tokenizer`. For multimodal models (handling images and text), we use an `AutoProcessor`. It bundles the tokenizer and image feature extractor together:

```python
from transformers import AutoProcessor

model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
```

### Quantization (Saving GPU Memory)

Model weights are usually stored in 32-bit or 16-bit floating points (`FP32`/`FP16`). A 7B parameter model in FP16 needs around 14–16 GB of VRAM just to sit in memory.

**Quantization** reduces weight precision down to 8-bit or 4-bit numbers without significant quality loss:

- Cuts VRAM usage by 50% to 75%.
- Makes fine-tuning possible on consumer GPUs (e.g., RTX 3090, T4, Colab).

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForImageTextToText

# Configure 4-bit / 8-bit loading
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=quant_config,
    device_map="auto"
)
```

---

## 3. Data Preparation Pipeline

Preparing training data for chat and vision models follows a 5-step pipeline:

```
[1. Raw Data] -> [2. Conversation Format] -> [3. Chat Template] -> [4. Tokenization] -> [5. Data Collation]
```

### Step 1: Raw Data

Usually stored as JSON or CSV files containing the prompt, response, and any media paths:

```json
{
  "image": "images/sample_01.jpg",
  "prompt": "Locate the cup on the table.",
  "answer": "{\"target\": \"cup\", \"box\": [120, 80, 240, 310]}"
}
```

### Step 2: Conversation Formatting

Convert raw rows into standard multi-turn conversation objects with roles (`user`, `assistant`):

```python
def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["prompt"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]}
                ]
            }
        ]
    }
```

### Step 3: Chat Templates

Different models expect different role tokens. For example, Qwen uses `<|im_start|>user ... <|im_end|>`.

`processor.apply_chat_template()` turns your message lists into the exact prompt string the base model was trained to read:

```python
formatted_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

### Step 4 & 5: Tokenization and Data Collation

A **Data Collator** takes individual training examples, tokens, and pads them so all samples in a batch share the exact same sequence length.

```python
from unsloth.trainer import UnslothVisionDataCollator

# Handles dynamic padding and multimodal tensors for vision models
data_collator = UnslothVisionDataCollator(model, processor.tokenizer)
```

---

## 4. Supervised Trainer Setup

### Why PEFT?

Updating all billions of parameters (Full Fine-Tuning) requires massive compute and easily causes "catastrophic forgetting."

**PEFT (Parameter-Efficient Fine-Tuning)** freezes most base model weights and only trains a tiny subset of extra parameters (~0.1% to 1%).

| Method                     | How It Works                                                      | Trainable Params |
| :------------------------- | :---------------------------------------------------------------- | :--------------- |
| **LoRA** (Most common)     | Injects low-rank decomposition matrices into attention/MLP layers | ~0.1% – 1%       |
| **Adapters**               | Adds small bottleneck layers between transformer blocks           | ~1% – 5%         |
| **Prefix / Prompt Tuning** | Prepends trainable virtual tokens to the inputs                   | < 0.1%           |

### How LoRA Works

Instead of updating a huge weight matrix $W$ ($512 \times 512 = 262,144$ values), LoRA splits the update into two tiny matrices:

- Matrix $A$ ($512 \times 8$)
- Matrix $B$ ($8 \times 512$)

$$W' = W + (A \times B)$$

This reduces trainable parameters from 262,144 down to just 8,192 (over 97% reduction), while achieving nearly identical quality.

```python
from unsloth import FastVisionModel

# Setup LoRA on target layers
model = FastVisionModel.get_peft_model(
    model,
    r=16,                # Rank dimension
    lora_alpha=16,       # Scaling factor (usually >= r)
    lora_dropout=0.05,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    random_state=3407
)
model.print_trainable_parameters()
```

### SFT Trainer Configuration

Using Hugging Face `TRL`'s `SFTTrainer` and `SFTConfig`:

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./output-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    optim="adamw_8bit",             # Saves memory
    weight_decay=0.01,
    report_to="wandb"               # Log metrics to Weights & Biases
)

trainer = SFTTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset
)

# Start training
trainer.train()
```

---

## 5. Training Diagnosis

### Reading the Loss Curves

During training, we monitor two primary metrics:

1. **Training Loss:** Measures how well the model predicts answers on the training data.
   - Should steadily go down as the model learns.
2. **Validation Loss:** Measures how well the model performs on unseen data.

| Scenario                                              | Diagnosis                         | Solution                                                               |
| :---------------------------------------------------- | :-------------------------------- | :--------------------------------------------------------------------- |
| **Train loss goes down, Val loss goes down**          | Healthy learning                  | Keep training or stop at lowest val loss                               |
| **Train loss drops very low, Val loss starts rising** | **Overfitting** (memorizing data) | Add dropout, increase weight decay, or stop earlier                    |
| **Both Train and Val loss remain high**               | **Underfitting** (not learning)   | Increase learning rate, train for more epochs, or unfreeze more layers |

### Tracking with Weights & Biases (WandB)

Instead of guessing training progress from terminal output, log it directly to WandB:

```python
import wandb

wandb.init(
    project="qwen2-vl-finetune",
    name="run-experiment-01"
)
```

This tracks training loss, learning rate decay, GPU memory, and gradient norms live in your browser.

---

## 6. Model Checkpoints

A checkpoint saves the model state (weights, optimizer status, step counter) at a specific point in time.

### Why Checkpoints Matter

- **Fault Tolerance:** If training crashes or GPU disconnects, resume without losing hours of progress.
- **Best Model Selection:** Pick the checkpoint with the lowest validation loss instead of just the final epoch.
- **Lightweight Deployments:** You only need to share or store the LoRA adapter folder (a few megabytes) instead of the entire multi-gigabyte base model.

### Loading Checkpoints for Inference

You can load a trained checkpoint either directly or by attaching the adapter onto the clean base model:

```python
from unsloth import FastVisionModel

# Option A: Load checkpoint directly
model, tokenizer = FastVisionModel.from_pretrained(
    "./output-model/checkpoint-1000",
    load_in_4bit=True
)

# Option B: Load base model + attach adapter
base_model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    load_in_4bit=True
)
base_model.load_adapter("./output-model/checkpoint-1000")
```
