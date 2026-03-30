![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# 3x3 Programmable Neural Network (Tiny TPU)

A fully programmable, hardware-based 3-neuron neural network designed for the [Tiny Tapeout](https://tinytapeout.com) educational ASIC project. Developed as part of the Digital Systems HW course (AUTH).

## Design Overview & Architecture

This project implements a sequential, shared-MAC Neural Network inside a **strict 1x1 ASIC Tile** limit. To satisfy the aggressive area footprint, the architecture multiplexes a single MAC (Multiply-Accumulate) unit across all neurons and inputs, managed by a custom Verilog Finite State Machine (FSM).

**Key Features:**
* **Programmable Weights & Biases:** Unlike static networks, this design allows dynamic loading of W (3x3 weight matrix), B (3-element bias vector), and X (3-element input vector) values.
* **Quantized Data Path:** To fit the 1x1 tile, the MAC operates on **6-bit signed integers** (Inputs & Weights range from -32 to +31), storing the results in a 14-bit accumulator with safe saturation logic to handle overflow constraints seamlessly.
* **Programmable PReLU Activation:** Programmable Parametric Rectified Linear Unit activation on output data. Configurable *per-neuron* via Address 15 using `ui_in[5:0]` as standard ReLU, x/2, x/4, or x/8.
* **Memory-mapped IO:** The `uio_in` pins act as a control bus (Address + Write/Read flags) to route inputs into the correct internal registers securely.

## How it Works

The design waits in the `IDLE` state while an external MCU (or testbench) streams the configuration. 

1. **Configuration Phase (`uio_in[4] = 1`, Write Mode):** 
   - `uio_in[3:0]` defines the target register address (0-2 for Inputs, 3-11 for Weights, 12-14 for Biases).
   - Data is applied concurrently to `ui_in[7:0]` (using the lower 6 bits, or all 8 for Biases).     
2. **Execution Phase (`uio_in = 15`, Start Calc):**
   - Apply a 6-bit value on `ui_in[5:0]` to configure the Programmable PReLU for **each neuron separately** (`ui_in[1:0]` for Neuron 0, `ui_in[3:2]` for Neuron 1, `ui_in[5:4]` for Neuron 2): `00` for standard ReLU, `01` for $x/2$, `10` for $x/4$, and `11` for $x/8$.       
   - The FSM switches to `MAC` state. It sequentially multiplies inputs with weights and accumulates safely.
   - It transitions to `RELU` state to apply the configured activation function and store the result.
   - Repeats until all 3 neurons have been evaluated.
3. **Read Phase (`uio_in[5] = 1`, Read Mode):**
   - Drives `uo_out` with the calculated Neural Network Output based on the requested address in `uio_in[1:0]`.

## How to Test It

The project is thoroughly verified mathematically and cyclically using a Python Coroutine Testbench (`cocotb`). 

### Prerequisites:
- Python 3+
- `iverilog` (Icarus Verilog)
- `make`

### Running the Tests:
1. Navigate to the testing directory:
   ```bash
   cd test
   ```
2. Install testbench dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the simulation:
   ```bash
   make
   ```

A successful output will log the sequential loading of Weights, Biases, and Inputs, followed by triggering the hardware calculation and asserting the exact dot-product values against the `uo_out` pin. If the simulation passes, the Verilog FSM is mathematically sound and clock-perfect!
