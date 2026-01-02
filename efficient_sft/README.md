# Efficient SFT - [Full Finetune, no LoRA]

## Plan of Action

### Models:
- [x] Test Inference with HF Implementation and Loss Calculation
- [x] Test Optimal Training
- [x] Add Gradient Checkpointing
- [x] Add CPUOffload Optimizer Support
- [x] Add fused linear cross entropy loss
- [x] Add Quantization Support

-- Detour for a little while -- 

- [ ] Packing Support, Batch Inference
- [ ] MultiGPU Support (FSDP, DDP, etc)