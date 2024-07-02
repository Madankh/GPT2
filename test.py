import torch

# Check if the GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Print GPU name
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    # Check for TF32 support
    tf32_supported = torch.cuda.get_device_capability(device)[0] >= 8
    print(f"TF32 Supported: {tf32_supported}")
else:
    print("No GPU available.")
