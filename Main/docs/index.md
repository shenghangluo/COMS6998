# Confidence Prediction Training Pipeline - Documentation

## Documentation Guidelines

**This document establishes the permanent documentation standards for this repository.**

These guidelines must be followed and remembered permanently for all documentation-related operations:

### Core Principles

1. **Single Source of Truth**: `/docs` at the project root is the **sole directory** for repository documentation. No documentation files should exist outside this directory except for the root `README.md` (which serves as the entry point and overview). This is the only location where repository documentation is stored.

2. **Separation of Concerns**: Each document must serve **one clear purpose** (e.g., setup guide, API reference, architecture overview). Avoid mixing unrelated topics in the same file. Documents are organized by specific function and topic.

3. **No Duplication**: Do not repeat information across multiple documents. Information appears in **exactly one place**. Merge or reference existing docs instead of duplicating content. Related documents should reference each other rather than repeating information.

4. **Minimalism**: Do not create new documentation unless **absolutely necessary**. Always check for an existing file to update or extend first. New documentation should only be created when no existing document can adequately serve the purpose.

5. **Consistency**: Maintain consistent structure, tone, and formatting across all documents, in line with the repository's documentation style guide. All documents follow unified organizational patterns and writing standards.

6. **Version Control Awareness**: When restructuring, preserve git history where possible to maintain context. Documentation changes should be trackable and auditable.

### Enforcement

These guidelines are **persistent rules** that apply to all current and future documentation work in this repository. They must be consulted before creating, modifying, or removing any documentation.

### Document Index

This documentation is organized into focused, purpose-specific guides:

#### Getting Started
- **[Setup Guide](setup.md)** - Installation, dependencies, and initial configuration
- **[Quick Start](quickstart.md)** - Running your first training in 5 minutes

#### Core Documentation
- **[Training Guide](training.md)** - Complete guide to training models (full fine-tuning and LoRA)
- **[Dataset Guide](datasets.md)** - Dataset information, label mappings, and data preparation
- **[Evaluation Guide](evaluation.md)** - Metrics, evaluation procedures, and result interpretation
- **[Configuration Reference](configuration.md)** - All configuration options and examples

#### Technical Reference
- **[Architecture](architecture.md)** - Model architecture, MLP head design, and implementation details
- **[API Reference](api-reference.md)** - Code module documentation and usage examples

#### Advanced Topics
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Best Practices](best-practices.md)** - Recommended training strategies and optimization tips

#### Experimental Documentation
- **[Experimental Setup](experimental-setup.md)** - Hardware, software, and configuration details
- **[Reproducibility Guide](reproducibility.md)** - Step-by-step instructions to reproduce results
- **[Results](results.md)** - Experimental results and performance analysis

---

## Project Overview

This pipeline enables LLMs to predict step-level confidence scores using a special `<CONFIDENCE>` token and an MLP head trained via regression ([0, 1] scale).

### Key Features
- Multi-dataset training (PRM800K, REVEAL, eQASC)
- Full fine-tuning and LoRA support
- Comprehensive evaluation metrics (MSE, MAE, R², ECE, stratified MAE)
- Modular, production-ready architecture

### Quick Links
- [Get Started in 5 Minutes](quickstart.md)
- [Full Training Guide](training.md)
- [Architecture Overview](architecture.md)
- [Troubleshooting](troubleshooting.md)
