# SPDX-FileCopyrightText: © 2026 Zoidis Vasileios
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


# NOTE: The tests should use -16 to +15 weights
#  to fit the 5-bit weights array (ui_in[4:0])

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
    
    # Wait for FSM to complete (MAC runs 3 clocks per neuron, PLUS 1 RELU clock = 4 clocks * 3 neurons = 12 clocks minimum.)
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

    dut._log.info("--- Edge Case 1: ReLU Negative Clamping ---")
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

    dut._log.info("All Programmable Matrix and Edge Case tests passed successfully!")

