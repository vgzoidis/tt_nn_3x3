/*
 * Copyright (c) 2026 Git Happens for Tiny Tapeout
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_nn_3x3 (
    input  wire [7:0] ui_in,    // Dedicated inputs - 8-bit data in
    output wire [7:0] uo_out,   // Dedicated outputs - 8-bit data out
    input  wire [7:0] uio_in,   // IOs: Input path (Used for Control signals)
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path
    input  wire       ena,      // always 1 when powered
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);
    // TinyTapeout Pin Configuration
    assign uio_oe  = 8'b0000_0000;
    assign uio_out = 8'b0;

    wire [3:0] io_addr  = uio_in[3:0];                  // 0..14=Write Addr, 15=Start Calc
    wire       io_write = uio_in[4];                    // 1=Write to selected address
    wire       io_read  = uio_in[5];                    // 1=Read outputs
    wire       _unused  = &{1'b0, uio_in[7:6], 1'b0};   // Suppress unused signals warning

    // Registers & Storage
    reg signed [5:0] x [0:2]; // 3 x 6-bit Inputs
    reg signed [7:0] y [0:2]; // 3 x 8-bit Outputs

    // Programmable Weights and Biases
    reg signed [5:0] W [0:8];               // Weights
    reg signed [7:0] B [0:2];               // Biases
    reg        [1:0] prelu_config [0:2];    // Per-neuron config

    // Shared MAC Unit Engine
    reg signed [13:0] accumulator; 
    reg        [1:0] calc_step;
    reg        [1:0] current_neuron;

    // Multiply dynamically shifted Weight (W[0] always holds current weight)
    wire signed [11:0] product = x[calc_step] * W[0];
    
    // Saturation Logic for MAC Addition
    wire signed [13:0] product_ext = $signed({ {2{product[11]}}, product });
    wire signed [13:0] next_acc_calc = accumulator + product_ext;
    wire underflow = (accumulator[13] & product[11] & ~next_acc_calc[13]);     
    wire overflow  = (~accumulator[13] & ~product[11] & next_acc_calc[13]);    
    
    wire signed [13:0] next_acc_safe = overflow  ? 14'h1FFF : 
                                       underflow ? 14'h2000 : 
                                       next_acc_calc[13:0];
                                       
    // Shifted values for PReLU to avoid linting width warnings (Combinational)
    wire [7:0] s_next_1 = next_acc_safe[8:1];
    wire [7:0] s_next_2 = next_acc_safe[9:2];
    wire [7:0] s_next_3 = next_acc_safe[10:3];

    // Combinational PReLU Activation
    reg [7:0] y_next;
    always @(*) begin
        if (next_acc_safe[13]) begin
            case (prelu_config[current_neuron])
                2'b00: y_next = 8'd0;
                2'b01: y_next = ($signed(next_acc_safe) < -256) ? 8'h80 : s_next_1;
                2'b10: y_next = ($signed(next_acc_safe) < -512) ? 8'h80 : s_next_2;
                2'b11: y_next = ($signed(next_acc_safe) < -1024) ? 8'h80 : s_next_3;
            endcase
        end else if ($signed(next_acc_safe) > 127) begin
            y_next = 8'd127;
        end else begin
            y_next = next_acc_safe[7:0];
        end
    end

    localparam STATE_IDLE = 1'b0;
    localparam STATE_MAC  = 1'b1;
    reg state;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            calc_step <= 0;
            current_neuron <= 0;
            accumulator <= 0;
            
            y[0] <= 0; y[1] <= 0; y[2] <= 0;
            x[0] <= 0; x[1] <= 0; x[2] <= 0;
            B[0] <= 0; B[1] <= 0; B[2] <= 0;
            
            W[0] <= 0; W[1] <= 0; W[2] <= 0;
            W[3] <= 0; W[4] <= 0; W[5] <= 0;
            W[6] <= 0; W[7] <= 0; W[8] <= 0;
            
            prelu_config[0] <= 2'b00;
            prelu_config[1] <= 2'b00;
            prelu_config[2] <= 2'b00;

        end else if (ena) begin
            // 1) Input Data Loading (When IDLE)
            if (state == STATE_IDLE && io_write) begin
                case (io_addr)
                    4'd0: x[0] <= ui_in[5:0];
                    4'd1: x[1] <= ui_in[5:0];
                    4'd2: x[2] <= ui_in[5:0];

                    4'd3: W[0] <= ui_in[5:0];
                    4'd4: W[1] <= ui_in[5:0];
                    4'd5: W[2] <= ui_in[5:0];
                    4'd6: W[3] <= ui_in[5:0];
                    4'd7: W[4] <= ui_in[5:0];
                    4'd8: W[5] <= ui_in[5:0];
                    4'd9: W[6] <= ui_in[5:0];
                    4'd10: W[7] <= ui_in[5:0];
                    4'd11: W[8] <= ui_in[5:0];

                    4'd12: B[0] <= ui_in;
                    4'd13: B[1] <= ui_in;
                    4'd14: B[2] <= ui_in;
                    4'd15: begin
                        prelu_config[0] <= ui_in[1:0];
                        prelu_config[1] <= ui_in[3:2];
                        prelu_config[2] <= ui_in[5:4];
                    end
                    default: ;
                endcase
            end

            // 2) FSM Core for NN calculations
            case (state)
                STATE_IDLE: begin
                    if (io_addr == 4'd15 && !io_write && !io_read) begin
                        state <= STATE_MAC;
                        calc_step <= 0;
                        current_neuron <= 0;
                        accumulator <= $signed({ {6{B[0][7]}}, B[0] }); // Load initial Bias
                    end
                end

                STATE_MAC: begin
                    // Shift Weights for next calculation
                    W[0] <= W[1]; W[1] <= W[2]; W[2] <= W[3];
                    W[3] <= W[4]; W[4] <= W[5]; W[5] <= W[6];
                    W[6] <= W[7]; W[7] <= W[8]; W[8] <= W[0];

                    if (calc_step == 2) begin
                        // Save the combinationally calculated ReLU output
                        y[current_neuron] <= y_next;
                        calc_step <= 0;

                        if (current_neuron == 2) begin
                            state <= STATE_IDLE; // End of network
                        end else begin
                            current_neuron <= current_neuron + 1;
                            // Pre-load next Bias
                            accumulator <= $signed({ {6{B[current_neuron + 1][7]}}, B[current_neuron + 1] });
                        end
                    end else begin
                        // Accumulator saturating addition
                        accumulator <= next_acc_safe;
                        calc_step <= calc_step + 1;
                    end
                end
            endcase
        end
    end

    // Output Routing (Multiplexed output based on address)
    reg [7:0] out_mux;
    always @(*) begin
        if (io_read) begin
            case(io_addr[1:0])
                2'd0: out_mux = y[0];
                2'd1: out_mux = y[1];
                2'd2: out_mux = y[2];
                default: out_mux = 8'h00;
            endcase
        end else begin
            out_mux = 8'h00;
        end
    end
    
    assign uo_out = out_mux;

endmodule
