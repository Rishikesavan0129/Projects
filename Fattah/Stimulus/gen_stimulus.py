import numpy as np
import os

# ==============================================================================
# FATTAH AI Configuration - 25k STRESS TEST
# ==============================================================================
NUM_BLOCKS = 25000       # 25,000 Frames (400,000 INT8 Elements)
ELEMENTS_PER_BLOCK = 16

# The exact path where Vivado will look for the files
OUTPUT_DIR = r"C:\Users\RISHIKESAVAN\Fattah\sim_data"

def relu_quantize(data):
    data = np.maximum(0, data)
    data = np.round(data * 127).astype(np.uint8)
    return data

def main():
    print("🧠 Initiating FATTAH 25,000-Block Stress Test...")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Generate Raw NPU Data
    raw_ai_data = np.random.normal(-0.5, 1.0, NUM_BLOCKS * ELEMENTS_PER_BLOCK)

    # 2. Apply NPU Activation
    quantized_data = relu_quantize(raw_ai_data)
    blocks = quantized_data.reshape((NUM_BLOCKS, ELEMENTS_PER_BLOCK))

    stimulus_lines = []
    expected_lines = []

    total_zeros = 0
    total_elements = NUM_BLOCKS * ELEMENTS_PER_BLOCK

    for block_idx, block in enumerate(blocks):
        dense_hex = "".join([f"{byte:02X}" for byte in reversed(block)])
        expected_lines.append(dense_hex)

        non_zero_indices = [i for i, val in enumerate(block) if val != 0]

        if len(non_zero_indices) == 0:
            packet_val = (1 << 13) | (16 << 8) | 0x00
            stimulus_lines.append(f"{packet_val:04X}")
            total_zeros += 16
        else:
            last_nz_idx = non_zero_indices[-1]
            skip_count = 0

            for i, val in enumerate(block):
                if val == 0:
                    skip_count += 1
                    total_zeros += 1
                else:
                    is_last = 1 if (i == last_nz_idx) else 0
                    packet_val = (is_last << 13) | (skip_count << 8) | int(val)
                    stimulus_lines.append(f"{packet_val:04X}")
                    skip_count = 0

    # Write to files
    stimulus_path = os.path.join(OUTPUT_DIR, "stimulus.hex")
    expected_path = os.path.join(OUTPUT_DIR, "expected_dense.hex")

    with open(stimulus_path, "w") as f:
        f.write("\n".join(stimulus_lines))

    with open(expected_path, "w") as f:
        f.write("\n".join(expected_lines))

    sparsity = (total_zeros / total_elements) * 100
    print(f"✅ Generated {NUM_BLOCKS} AI Blocks ({total_elements} elements).")
    print(f"✅ Natural AI Sparsity Achieved: {sparsity:.2f}%")
    print("✅ Ready for Vivado Simulation!")

if __name__ == "__main__":
    main()
