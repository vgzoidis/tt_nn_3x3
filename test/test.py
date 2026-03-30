# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

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

    dut._log.info("All Programmable Matrix tests passed successfully!")

