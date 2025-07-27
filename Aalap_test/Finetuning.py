#!/usr/bin/env python3
"""
TinyLlama Fine-tuning Example
A complete example for fine-tuning TinyLlama-1.1B locally on custom data.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import numpy as np
from typing import Dict, List

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-finetuned"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3


class CustomDataset:
    """Custom dataset class for preparing training data."""

    def __init__(self, data_path: str = None):
        self.conversations = []
        if data_path:
            self.load_data(data_path)
        else:
            # Create sample data if no path provided
            self.create_sample_data()

    def create_sample_data(self):
        """Create sample conversational data for demonstration."""
        sample_data = [
            {
                "instruction": "Explain what machine learning is",
                "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
            },
            {
                "instruction": "What are the benefits of renewable energy?",
                "response": "Renewable energy offers several benefits: it's environmentally friendly, reduces carbon emissions, provides energy independence, and becomes more cost-effective over time."
            },
            {
                "instruction": "How do you make a simple pasta dish?",
                "response": "To make simple pasta: 1) Boil salted water, 2) Cook pasta according to package directions, 3) Heat olive oil in a pan, 4) Add garlic and cook briefly, 5) Toss pasta with oil and garlic, 6) Season with salt, pepper, and cheese."
            },
            {
                "instruction": "Explain the water cycle",
                "response": "The water cycle consists of evaporation (water turns to vapor), condensation (vapor forms clouds), precipitation (rain/snow falls), and collection (water gathers in bodies of water), then the cycle repeats."
            },
            {
                "instruction": "What is photosynthesis?",
                "response": "Photosynthesis is the process by which plants use sunlight, carbon dioxide, and water to produce glucose and oxygen. It occurs in chloroplasts and is essential for life on Earth."
            }
        ]

        # Format data for chat template
        for item in sample_data:
            conversation = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["response"]}
            ]
            self.conversations.append(conversation)

    def load_data(self, data_path: str):
        """Load data from JSON file."""
        with open(data_path, 'r') as f:
            data = json.load(f)

        for item in data:
            if "conversation" in item:
                self.conversations.append(item["conversation"])
            else:
                # Convert instruction-response format to conversation
                conversation = [
                    {"role": "user", "content": item.get("instruction", item.get("input", ""))},
                    {"role": "assistant", "content": item.get("response", item.get("output", ""))}
                ]
                # Only add conversation if both instruction and response are non-empty
                if conversation[0]['content'] and conversation[1]['content']:
                    self.conversations.append(conversation)


def prepare_dataset(conversations: List[List[Dict]], tokenizer, max_length: int = 512):
    """Prepare dataset for training."""

    def format_conversation(conversation):
        """Format conversation using chat template."""
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

    # Format all conversations
    formatted_texts = [format_conversation(conv) for conv in conversations]

    # Tokenize
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        labels = []

        for ids in input_ids:
            labels.append([id if id != tokenizer.pad_token_id else -100 for id in ids])

        tokenized["labels"] = labels
        return tokenized

    # Create HuggingFace dataset
    dataset = HFDataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


def setup_model_and_tokenizer(model_name: str):
    """Setup model and tokenizer."""
    print(f"Loading model and tokenizer: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def train_model(model, tokenizer, train_dataset, output_dir: str):
    """Train the model."""

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
        pad_to_multiple_of=8,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,  # Effective batch size = BATCH_SIZE * 4
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to=[],  # Disable wandb/tensorboard logging
        load_best_model_at_end=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return trainer


def test_model(model_path: str, test_prompts: List[str]):
    """Test the fine-tuned model."""
    print(f"\nTesting fine-tuned model from: {model_path}")

    # Load fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    model.eval()

    print(f"Vocab size: {len(tokenizer)}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer length: {len(tokenizer)}")

    for prompt in test_prompts:
        print(f"\n{'=' * 50}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 50}")

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"Formatted prompt: {repr(formatted_prompt)}")

        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=inputs["input_ids"].shape[-1] + 150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Debug the full output first
        print(f"Full generated text: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print(f"Input text: {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}")

        # Decode
        input_length = inputs["input_ids"].shape[-1]
        if outputs.shape[-1] > input_length:
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = "No new tokens generated"

        print(f"Response: {response.strip()}")


# Also test the base model for comparison
def test_base_model():
    """Test the original base model to see if it works."""
    print("Testing base model for comparison...")

    model_path = "./tinyllama-finetuned"

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "What is artificial intelligence?"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Base model response: '{response.strip()}'")


def main():
    """Main training pipeline."""

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare data
    print("Preparing dataset...")
    dataset = CustomDataset()  # Uses sample data
    # To use your own data: dataset = CustomDataset("path/to/your/data.json")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

    # Prepare training dataset
    train_dataset = prepare_dataset(dataset.conversations, tokenizer, MAX_LENGTH)
    print(f"Training dataset size: {len(train_dataset)}")

    # Train model
    trainer = train_model(model, tokenizer, train_dataset, OUTPUT_DIR)

    # Test the model
    test_prompts = [
        "What is artificial intelligence?",
        "Explain how solar panels work",
        "Give me a recipe for chocolate cookies"
    ]

    test_model(OUTPUT_DIR, test_prompts)

    test_base_model()

    print(f"\nTraining completed! Model saved to: {OUTPUT_DIR}")
    """
    To use your fine-tuned model later:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained('OUTPUT_DIR')
    model = AutoModelForCausalLM.from_pretrained('OUTPUT_DIR')
    """


if __name__ == "__main__":
    main()