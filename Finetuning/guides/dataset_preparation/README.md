# Dataset Preparation and Formatting Guide

This comprehensive guide covers dataset preparation and formatting for LLaMA-Factory, including:

1. **Dataset Formats**: Instruction-response, conversation, preference, multimodal
2. **Data Quality**: Cleaning, filtering, and validation
3. **Data Augmentation**: Techniques for improving dataset diversity
4. **Format Conversion**: Converting between different dataset formats
5. **Dataset Tools**: Scripts and utilities for dataset processing
6. **Best Practices**: Data collection and preparation strategies

## Table of Contents

- [Dataset Formats](#dataset-formats)
- [Data Quality](#data-quality)
- [Data Augmentation](#data-augmentation)
- [Format Conversion](#format-conversion)
- [Dataset Tools](#dataset-tools)
- [Best Practices](#best-practices)

## Dataset Formats

### 1. Instruction-Response Format (Alpaca-style)

```json
[
  {
    "instruction": "Explain the concept of machine learning",
    "input": "in simple terms",
    "output": "Machine learning is a type of artificial intelligence..."
  }
]
```

**Use Cases**: Supervised fine-tuning, instruction following tasks

### 2. Conversation Format (ShareGPT-style)

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hello!"},
      {"from": "gpt", "value": "Hi there! How can I help you?"},
      {"from": "human", "value": "Explain quantum computing"},
      {"from": "gpt", "value": "Quantum computing is..."}
    ]
  }
]
```

**Use Cases**: Multi-turn conversations, chatbots, conversational AI

### 3. Preference Format (DPO/KTO-style)

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Write a story"},
      {"from": "gpt", "value": "I'll write a story"}
    ],
    "chosen": {"from": "gpt", "value": "Once upon a time..."},
    "rejected": {"from": "gpt", "value": "There once was..."}
  }
]
```

**Use Cases**: Preference optimization, alignment training

### 4. Multimodal Format

```json
[
  {
    "messages": [
      {"role": "user", "content": "Describe this image"},
      {"role": "assistant", "content": "This image shows..."}
    ],
    "images": ["path/to/image.jpg"],
    "audios": ["path/to/audio.wav"]
  }
]
```

**Use Cases**: Vision-language models, multimodal understanding

## Data Quality

### 1. Data Cleaning

```python
def clean_dataset(data):
    """Clean and filter dataset"""

    cleaned_data = []

    for item in data:
        # Remove empty or very short responses
        if len(item.get("output", "")) < 10:
            continue

        # Remove duplicates
        if item in cleaned_data:
            continue

        # Validate format
        if not validate_format(item):
            continue

        cleaned_data.append(item)

    return cleaned_data

def validate_format(item):
    """Validate data format"""
    required_fields = ["instruction", "output"]  # Adjust based on format

    for field in required_fields:
        if field not in item or not item[field]:
            return False

    return True
```

### 2. Data Filtering

```python
def filter_dataset(data, criteria):
    """Filter dataset based on criteria"""

    filtered_data = []

    for item in data:
        # Length filtering
        if len(item["output"]) < criteria["min_length"]:
            continue

        if len(item["output"]) > criteria["max_length"]:
            continue

        # Quality filtering (example)
        if criteria["remove_low_quality"]:
            quality_score = assess_quality(item)
            if quality_score < criteria["quality_threshold"]:
                continue

        filtered_data.append(item)

    return filtered_data
```

### 3. Data Validation

```python
def validate_dataset_format(data, format_type="instruction"):
    """Validate dataset format"""

    errors = []

    for i, item in enumerate(data):
        # Check required fields
        if format_type == "instruction":
            required = ["instruction", "output"]
        elif format_type == "conversation":
            required = ["conversations"]
        elif format_type == "preference":
            required = ["conversations", "chosen", "rejected"]

        for field in required:
            if field not in item:
                errors.append(f"Item {i}: Missing field '{field}'")

        # Validate conversation format
        if "conversations" in item:
            if not validate_conversation(item["conversations"]):
                errors.append(f"Item {i}: Invalid conversation format")

    return errors

def validate_conversation(conversations):
    """Validate conversation format"""

    if not conversations:
        return False

    # Check alternating human/gpt
    for i, msg in enumerate(conversations):
        if msg["from"] not in ["human", "gpt"]:
            return False

        # First message should be from human
        if i == 0 and msg["from"] != "human":
            return False

        # Check for empty messages
        if not msg.get("value", "").strip():
            return False

    return True
```

## Data Augmentation

### 1. Response Augmentation

```python
def augment_responses(data, augmentation_factor=2):
    """Augment dataset with variations of responses"""

    augmented_data = []

    for item in data:
        augmented_data.append(item)  # Original

        # Create variations
        for _ in range(augmentation_factor - 1):
            variation = create_response_variation(item)
            if variation:
                augmented_data.append(variation)

    return augmented_data

def create_response_variation(item):
    """Create a variation of a response"""

    # Simple example: paraphrase or restructure
    original = item["output"]

    # Add slight variations
    variations = [
        original,  # Original
        f"Let me explain: {original}",
        f"To put it simply: {original}",
        f"Here's how it works: {original}"
    ]

    # Randomly select a variation
    variation = random.choice(variations)

    return {
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": variation
    }
```

### 2. Instruction Augmentation

```python
def augment_instructions(data):
    """Augment instructions with different phrasings"""

    instruction_templates = [
        "Explain {}",
        "Describe {}",
        "What is {}?",
        "How does {} work?",
        "Tell me about {}",
        "Give me information about {}",
        "I want to learn about {}",
        "Can you explain {} to me?"
    ]

    augmented_data = []

    for item in data:
        # Original instruction
        augmented_data.append(item)

        # Create variations
        base_instruction = item["instruction"]

        for template in instruction_templates:
            if not base_instruction.startswith(template.split("{")[0]):
                variation = template.format(base_instruction)
                augmented_data.append({
                    "instruction": variation,
                    "input": item.get("input", ""),
                    "output": item["output"]
                })

    return augmented_data
```

### 3. Conversation Augmentation

```python
def augment_conversations(data):
    """Augment conversation data"""

    augmented_data = []

    for item in data:
        augmented_data.append(item)  # Original

        # Add follow-up questions
        follow_up = generate_follow_up(item["conversations"])
        if follow_up:
            augmented_data.append(follow_up)

    return augmented_data

def generate_follow_up(conversation):
    """Generate follow-up conversations"""

    if len(conversation) < 2:
        return None

    last_response = conversation[-1]["value"]

    # Generate follow-up based on content
    follow_up_questions = [
        "Can you explain that in more detail?",
        "What are the main benefits of that?",
        "Are there any drawbacks to consider?",
        "How does that compare to alternatives?",
        "Can you give me an example?"
    ]

    follow_up = random.choice(follow_up_questions)

    return {
        "conversations": conversation + [
            {"from": "human", "value": follow_up},
            {"from": "gpt", "value": "Certainly! Let me elaborate..."}
        ]
    }
```

## Format Conversion

### 1. Convert Instruction to Conversation Format

```python
def instruction_to_conversation(data):
    """Convert instruction-response to conversation format"""

    converted_data = []

    for item in data:
        # Create conversation from instruction and response
        conversation = [
            {"from": "human", "value": item["instruction"]},
            {"from": "gpt", "value": item["output"]}
        ]

        # Add input context if present
        if item.get("input"):
            conversation[0]["value"] += f" {item['input']}"

        converted_data.append({
            "conversations": conversation
        })

    return converted_data
```

### 2. Convert Conversation to Preference Format

```python
def conversation_to_preference(data):
    """Convert conversation to preference format"""

    preference_data = []

    for item in data:
        conversations = item["conversations"]

        # Need at least 2 messages (human + assistant)
        if len(conversations) < 2:
            continue

        # Find assistant response
        assistant_responses = [
            msg for msg in conversations
            if msg["from"] == "gpt"
        ]

        if len(assistant_responses) < 1:
            continue

        # Split into chosen and rejected
        # (This is a simplified example)
        chosen_response = assistant_responses[0]["value"]

        # Create a slightly modified version as "rejected"
        rejected_response = create_rejected_version(chosen_response)

        preference_data.append({
            "conversations": conversations[:-1],  # Remove last response
            "chosen": {"from": "gpt", "value": chosen_response},
            "rejected": {"from": "gpt", "value": rejected_response}
        })

    return preference_data
```

### 3. Convert Preference to KTO Format

```python
def preference_to_kto(data):
    """Convert preference format to KTO format"""

    kto_data = []

    for item in data:
        # Use chosen as desirable
        kto_data.append({
            "conversations": item["conversations"] + [item["chosen"]],
            "kto_tag": True  # Desirable
        })

        # Use rejected as undesirable
        kto_data.append({
            "conversations": item["conversations"] + [item["rejected"]],
            "kto_tag": False  # Undesirable
        })

    return kto_data
```

## Dataset Tools

### 1. Dataset Statistics

```python
def dataset_statistics(data, format_type="instruction"):
    """Calculate dataset statistics"""

    stats = {
        "total_samples": len(data),
        "avg_instruction_length": 0,
        "avg_response_length": 0,
        "format_errors": 0,
        "quality_score": 0
    }

    instruction_lengths = []
    response_lengths = []

    for item in data:
        # Instruction length
        if format_type == "instruction":
            instruction = item.get("instruction", "")
        elif format_type == "conversation":
            # Get human messages
            instruction = " ".join([
                msg["value"] for msg in item["conversations"]
                if msg["from"] == "human"
            ])
        else:
            instruction = ""

        instruction_lengths.append(len(instruction.split()))

        # Response length
        if format_type == "instruction":
            response = item.get("output", "")
        elif format_type == "conversation":
            response = " ".join([
                msg["value"] for msg in item["conversations"]
                if msg["from"] == "gpt"
            ])
        elif format_type == "preference":
            response = item.get("chosen", {}).get("value", "")

        response_lengths.append(len(response.split()))

    stats["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
    stats["avg_response_length"] = sum(response_lengths) / len(response_lengths)

    return stats
```

### 2. Dataset Quality Assessment

```python
def assess_dataset_quality(data):
    """Assess overall dataset quality"""

    quality_scores = []

    for item in data:
        score = 0

        # Length criteria (0-20 points)
        response_length = len(item.get("output", "").split())
        if 50 <= response_length <= 500:
            score += 10
        elif 20 <= response_length <= 1000:
            score += 5

        # Structure criteria (0-20 points)
        if has_good_structure(item):
            score += 10

        # Content criteria (0-20 points)
        if has_informative_content(item):
            score += 10

        # Clarity criteria (0-20 points)
        if has_clear_language(item):
            score += 10

        # Uniqueness criteria (0-20 points)
        if not is_duplicate(item, data):
            score += 10

        quality_scores.append(score)

    avg_quality = sum(quality_scores) / len(quality_scores)

    return {
        "average_quality": avg_quality,
        "quality_distribution": {
            "excellent": len([s for s in quality_scores if s >= 80]),
            "good": len([s for s in quality_scores if 60 <= s < 80]),
            "fair": len([s for s in quality_scores if 40 <= s < 60]),
            "poor": len([s for s in quality_scores if s < 40])
        }
    }
```

### 3. Dataset Deduplication

```python
def deduplicate_dataset(data, similarity_threshold=0.9):
    """Remove duplicate or very similar entries"""

    def calculate_similarity(text1, text2):
        """Calculate text similarity"""
        # Simple similarity measure
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    unique_data = []

    for item in data:
        is_duplicate = False

        for unique_item in unique_data:
            # Check instruction similarity
            sim = calculate_similarity(
                item.get("instruction", ""),
                unique_item.get("instruction", "")
            )

            if sim >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_data.append(item)

    return unique_data
```

## Best Practices

### 1. Data Collection Guidelines

1. **Diversity**: Ensure diverse topics, writing styles, and complexity levels
2. **Quality**: Prefer quality over quantity - 100 high-quality samples > 1000 low-quality
3. **Consistency**: Maintain consistent format throughout the dataset
4. **Balance**: Balance different types of instructions and responses
5. **Validation**: Always validate data format and quality before training

### 2. Dataset Preparation Workflow

```python
def prepare_dataset_for_training(raw_data, target_format="conversation"):
    """Complete dataset preparation workflow"""

    # 1. Clean data
    cleaned_data = clean_dataset(raw_data)

    # 2. Filter data
    criteria = {
        "min_length": 50,
        "max_length": 1000,
        "remove_low_quality": True,
        "quality_threshold": 60
    }
    filtered_data = filter_dataset(cleaned_data, criteria)

    # 3. Validate format
    errors = validate_dataset_format(filtered_data, target_format)
    if errors:
        print(f"Format errors found: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"- {error}")

    # 4. Convert format if needed
    if target_format == "conversation":
        prepared_data = instruction_to_conversation(filtered_data)
    else:
        prepared_data = filtered_data

    # 5. Augment data (optional)
    augmented_data = augment_instructions(prepared_data)

    # 6. Deduplicate
    final_data = deduplicate_dataset(augmented_data)

    # 7. Generate statistics
    stats = dataset_statistics(final_data, target_format)

    return final_data, stats
```

### 3. Quality Control

1. **Manual Review**: Sample 10-20% of data manually
2. **Automated Checks**: Use scripts for format validation
3. **Consistency Checks**: Ensure consistent formatting
4. **Length Analysis**: Check for appropriate response lengths
5. **Content Filtering**: Remove inappropriate or harmful content

### 4. Dataset Optimization

1. **Format Selection**: Choose format based on training objective
2. **Size Optimization**: Balance dataset size with quality
3. **Preprocessing**: Clean and normalize text data
4. **Tokenization**: Ensure compatibility with model tokenizer
5. **Caching**: Cache processed datasets for faster loading

## Summary

This guide provides comprehensive strategies for dataset preparation:

- **Format Support**: Multiple dataset formats for different use cases
- **Quality Assurance**: Data cleaning, filtering, and validation
- **Augmentation**: Techniques for increasing dataset diversity
- **Conversion**: Tools for converting between formats
- **Analysis**: Statistics and quality assessment tools
- **Best Practices**: Proven strategies for dataset preparation

For hands-on examples, see the [notebooks](../notebooks/) directory.
