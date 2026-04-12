# SPDX-FileCopyrightText: © 2026 Git Happens for Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


# NOTE: The tests should use -32 to +31 weights
#  to fit the 6-bit weights array (ui_in[5:0])

@cocotb.test()
async def test_nn_project(dut):
    dut._log.info("Start NN Programmable Test")

    # Set the clock period to 20 ns (50 MHz)
    clock = Clock(dut.clk, 20, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Function to write unsigned/signed 8-bit to UI
    def to_8bit(val):
        return val & 0xFF

    dut._log.info("Loading Weights...")
    # W[0] = [1, -2, 3] => Addr 3, 4, 5
    weights = [1, -2, 3, 4, 5, -1, -3, 2, 1]
    for i, w in enumerate(weights):
        dut.ui_in.value = to_8bit(w)
        dut.uio_in.value = 16 + 3 + i # uio_in[4]=1 (16) + address
        await ClockCycles(dut.clk, 1)
        
    dut._log.info("Loading Biases...")
    # B = [5, -3, 1] => Addr 12, 13, 14
    biases = [5, -3, 1]
    for i, b in enumerate(biases):
        dut.ui_in.value = to_8bit(b)
        dut.uio_in.value = 16 + 12 + i 
        await ClockCycles(dut.clk, 1)

    dut._log.info("Setting PReLU Config to standard ReLU (00)...")
    dut.ui_in.value = 0
    dut.uio_in.value = 16 + 15  # io_write=1 (16) + addr 15
    await ClockCycles(dut.clk, 1)

    dut._log.info("Loading Inputs...")
    # X = [2, 3, 1] => Addr 0, 1, 2
    inputs = [2, 3, 1]
    for i, x in enumerate(inputs):
        dut.ui_in.value = to_8bit(x)
        dut.uio_in.value = 16 + i
        await ClockCycles(dut.clk, 1)

    dut._log.info("Starting calculation...")
    # Start Calc: io_addr = 15, write = 0, read = 0 => uio_in = 15
    dut.uio_in.value = 15
    await ClockCycles(dut.clk, 1)
    
    # Set to idle state inputs while calculating
    dut.uio_in.value = 0 
    
    # Wait for FSM to complete (MAC runs 3 clocks per neuron = 3 clocks * 3 neurons = 9 clocks minimum.)
    await ClockCycles(dut.clk, 20)

    dut._log.info("Reading outputs...")
    # Read Y[0] Expected: (2*1) + (3*-2) + (1*3) + 5 = 4
    dut.uio_in.value = 32 + 0 # uio_in[5]=1 (32) + address
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 4, f"Y[0] was {dut.uo_out.value}, expected 4"

    # Read Y[1] Expected: (2*4) + (3*5) + (1*-1) - 3 = 19
    dut.uio_in.value = 32 + 1
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 19, f"Y[1] was {dut.uo_out.value}, expected 19"

    # Read Y[2] Expected: (2*-3) + (3*2) + (1*1) + 1 = 2
    dut.uio_in.value = 32 + 2
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 2, f"Y[2] was {dut.uo_out.value}, expected 2"

    dut._log.info("--- Edge Case 1a: ReLU Negative Clamping ---")
    # Load inputs that will produce a strongly negative MAC result
    neg_inputs = [5, 5, 5]
    for i, x in enumerate(neg_inputs):
        dut.ui_in.value = to_8bit(x)
        dut.uio_in.value = 16 + i
        await ClockCycles(dut.clk, 1)
    
    # Load extremely negative weights for Neuron 0
    dut.ui_in.value = to_8bit(-10)
    for i in range(3):
        dut.uio_in.value = 16 + 3 + i
        await ClockCycles(dut.clk, 1)
        
    dut.uio_in.value = 15 # Start Calc
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 20)
    
    dut.uio_in.value = 32 + 0 # Read Y[0]
    await ClockCycles(dut.clk, 1)
    # MAC would be (5*-10)*3 + 5 = -145. ReLU should clamp to 0.
    assert dut.uo_out.value == 0, f"ReLU failed to clamp negative value! Y[0] was {dut.uo_out.value}"

    dut._log.info("--- Edge Case 1b: Programmable PReLU (Divide by 4) ---")
    # Keep the same inputs and weights (MAC = -145) but change PReLU config to 2'b10 (x/4)
    dut.ui_in.value = 2  # 2'b10
    dut.uio_in.value = 16 + 15
    await ClockCycles(dut.clk, 1)

    dut.uio_in.value = 15 # Start Calc
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 20)
    
    dut.uio_in.value = 32 + 0 # Read Y[0]
    await ClockCycles(dut.clk, 1)
    # MAC = -145. Shift right by 2 => -145 // 4 = -37
    # In 8-bit unsigned, -37 is 256 - 37 = 219
    expected_prelu = 219
    actual = dut.uo_out.value
    assert actual == expected_prelu, f"PReLU div-4 failed! Y[0] was {actual}, expected {expected_prelu}"

    dut._log.info("--- Edge Case 2: Positive Saturation ---")
    # Load inputs that will produce a massively positive MAC result (>127 max)
    pos_inputs = [15, 15, 15]
    for i, x in enumerate(pos_inputs):
        dut.ui_in.value = to_8bit(x)
        dut.uio_in.value = 16 + i
        await ClockCycles(dut.clk, 1)
    
    dut.ui_in.value = to_8bit(15) # Max positive weight
    for i in range(3):
        dut.uio_in.value = 16 + 3 + i
        await ClockCycles(dut.clk, 1)

    dut.uio_in.value = 15 # Start Calc
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 20)

    dut.uio_in.value = 32 + 0 # Read Y[0]
    await ClockCycles(dut.clk, 1)
    # MAC would be (15*15)*3 + 5 = 680. Must saturate to 127 (0x7F).
    assert dut.uo_out.value == 127, f"Saturation failed! Y[0] was {dut.uo_out.value}"

    dut._log.info("--- Edge Case 3: Zero Propagation ---")
    # Set inputs to all zeros
    for i in range(3):
        dut.ui_in.value = 0
        dut.uio_in.value = 16 + i
        await ClockCycles(dut.clk, 1)
    # Set PReLU configs to standard ReLU (00)
    dut.ui_in.value = 0
    dut.uio_in.value = 16 + 15
    await ClockCycles(dut.clk, 1)

    dut.uio_in.value = 15 # Start Calc
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 20)

    # Bias[0] = 5, Bias[1] = -3, Bias[2] = 1 (from previous loading)
    # Output should equal Bias if positive, or 0 if negative
    expected_zero_prop = [5, 0, 1]
    for i in range(3):
        dut.uio_in.value = 32 + i
        await ClockCycles(dut.clk, 1)
        assert dut.uo_out.value == expected_zero_prop[i], f"Zero prop failed! Y[{i}] was {dut.uo_out.value}, expected {expected_zero_prop[i]}"

    dut._log.info("--- Edge Case 4: Negative Inputs with Negative Weights ---")
    # Inputs: -4, -5, -2 => Addr 0, 1, 2
    neg_x = [-4, -5, -2]
    for i, x in enumerate(neg_x):
        dut.ui_in.value = to_8bit(x)
        dut.uio_in.value = 16 + i
        await ClockCycles(dut.clk, 1)

    # Weights for Neuron 0: -3, -2, -5Rightarrow Addr 3, 4, 5
    neg_w = [-3, -2, -5]
    for i, w in enumerate(neg_w):
        dut.ui_in.value = to_8bit(w)
        dut.uio_in.value = 16 + 3 + i
        await ClockCycles(dut.clk, 1)

    dut.uio_in.value = 15 # Start Calc
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 20)

    dut.uio_in.value = 32 + 0 # Read Y[0]
    await ClockCycles(dut.clk, 1)
    # (-4 * -3) + (-5 * -2) + (-2 * -5) + Bias[0](5) = 12 + 10 + 10 + 5 = 37
    assert dut.uo_out.value == 37, f"Neg*Neg failed! Y[0] was {dut.uo_out.value}, expected 37"

    dut._log.info("--- Edge Case 5: PReLU Alternate Configs (div 2, div 8) ---")
    # Create large negative MAC = -64 for Neuron 0 and Neuron 1 (using same inputs/weights for simplicity)
    # X = [8, 8, 8], W = [-3, -3, -3], B = [8, 8, 8] => (8 * -3) * 3 + 8 = -72 * 3 + 8 = -64
    x_val = 8
    for i in range(3):
        dut.ui_in.value = to_8bit(x_val)
        dut.uio_in.value = 16 + i
        await ClockCycles(dut.clk, 1)

    w_val = -3
    for i in range(9):
        dut.ui_in.value = to_8bit(w_val)
        dut.uio_in.value = 16 + 3 + i
        await ClockCycles(dut.clk, 1)

    b_val = 8
    for i in range(3):
        dut.ui_in.value = to_8bit(b_val)
        dut.uio_in.value = 16 + 12 + i
        await ClockCycles(dut.clk, 1)

    # Set PReLU: Neuron 0 to div 2 (01), Neuron 1 to div 8 (11), Neuron 2 to div 4 (10)
    # Config = 10_11_01 in binary => 45 in decimal
    dut.ui_in.value = 45
    dut.uio_in.value = 16 + 15
    await ClockCycles(dut.clk, 1)

    dut.uio_in.value = 15 # Start Calc
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 20)

    # Y[0] (div 2): -64 // 2 = -32 => 256 - 32 = 224
    dut.uio_in.value = 32 + 0 
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 224, f"PReLU div-2 failed! Y[0] was {dut.uo_out.value}, expected 224"

    # Y[1] (div 8): -64 // 8 = -8 => 256 - 8 = 248
    dut.uio_in.value = 32 + 1 
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 248, f"PReLU div-8 failed! Y[1] was {dut.uo_out.value}, expected 248"

    dut._log.info("All Programmable Matrix and Edge Case tests passed successfully!")

