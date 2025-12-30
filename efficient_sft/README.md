# Efficient SFT - [Full Finetune, no LoRA]

## Plan of Action

### Models:
- [x] Test Inference with HF Implementation and Loss Calculation
- [x] Test Optimal Training
- [x] Add Gradient Checkpointing
- [x] Add CPUOffload Optimizer Support
- [ ] Packing Support, Batch Inference
- [ ] Add fused linear cross entropy loss
- [ ] MultiGPU Support (FSDP, DDP, etc)