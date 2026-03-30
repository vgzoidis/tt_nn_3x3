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

    // ==========================================
    // TinyTapeout Pin Configuration
    // ==========================================
    assign uio_oe  = 8'b0000_0000;
    assign uio_out = 8'b0;

    wire [3:0] io_addr  = uio_in[3:0]; // 0..14=Write Addr, 15=Start Calc
    wire       io_write = uio_in[4];   // 1=Write to selected address
    wire       io_read  = uio_in[5];   // 1=Read outputs

    wire _unused = &{1'b0, uio_in[7:6], 1'b0}; // Suppress unused signals warning

    // ==========================================
    // Registers & Storage (Optimized for Tile area limits)
    // ==========================================
    reg signed [4:0] x [0:2]; // 3 x 5-bit Inputs
    reg signed [7:0] y [0:2]; // 3 x 8-bit Outputs

    // Programmable Weights and Biases (Reduced to 5-bit to fit density)
    reg signed [4:0] W [0:8]; // Weights (5-bit: -16 to +15)
    reg signed [7:0] B [0:2]; // Biases (8-bit)
    reg [1:0] prelu_config;   // 00: standard ReLU, 01: x/2, 10: x/4, 11: x/8

    // ==========================================
    // Three Parallel MAC Units
    // ==========================================
    reg signed [11:0] acc [0:2];     // 3 x 12-bit accumulators (one per neuron)
    reg [1:0] calc_step;              // Steps 0-2 for each input element

    // Exactly 3 multipliers (one per neuron) time-multiplexed across calc_step
    wire signed [9:0] prod_0 = x[calc_step] * W[0 + calc_step];
    wire signed [9:0] prod_1 = x[calc_step] * W[3 + calc_step];
    wire signed [9:0] prod_2 = x[calc_step] * W[6 + calc_step];

    // Sign-extended products for accumulation
    wire signed [11:0] prod_ext [0:2];
    assign prod_ext[0] = $signed({{2{prod_0[9]}}, prod_0});
    assign prod_ext[1] = $signed({{2{prod_1[9]}}, prod_1});
    assign prod_ext[2] = $signed({{2{prod_2[9]}}, prod_2});

    localparam STATE_IDLE = 2'b00;
    localparam STATE_MAC  = 2'b01;
    localparam STATE_RELU = 2'b10;
    reg [1:0] state;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            calc_step <= 0;
            
            y[0] <= 0; y[1] <= 0; y[2] <= 0;
            x[0] <= 0; x[1] <= 0; x[2] <= 0;
            B[0] <= 0; B[1] <= 0; B[2] <= 0;
            
            W[0] <= 0; W[1] <= 0; W[2] <= 0;
            W[3] <= 0; W[4] <= 0; W[5] <= 0;
            W[6] <= 0; W[7] <= 0; W[8] <= 0;
            prelu_config <= 2'b00;
            
            acc[0] <= 0; acc[1] <= 0; acc[2] <= 0;

        end else if (ena) begin
            // 1) Input Data Loading (When IDLE)
            if (state == STATE_IDLE && io_write) begin
                case (io_addr)
                    4'd0: x[0] <= ui_in[4:0];
                    4'd1: x[1] <= ui_in[4:0];
                    4'd2: x[2] <= ui_in[4:0];

                    4'd3: W[0] <= ui_in[4:0];
                    4'd4: W[1] <= ui_in[4:0];
                    4'd5: W[2] <= ui_in[4:0];
                    4'd6: W[3] <= ui_in[4:0];
                    4'd7: W[4] <= ui_in[4:0];
                    4'd8: W[5] <= ui_in[4:0];
                    4'd9: W[6] <= ui_in[4:0];
                    4'd10: W[7] <= ui_in[4:0];
                    4'd11: W[8] <= ui_in[4:0];

                    4'd12: B[0] <= ui_in;
                    4'd13: B[1] <= ui_in;
                    4'd14: B[2] <= ui_in;
                    4'd15: prelu_config <= ui_in[1:0];
                    default: ;
                endcase
            end

            // 2) FSM Core for NN calculations with 3 Parallel MACs
            case (state)
                STATE_IDLE: begin
                    if (io_addr == 4'd15 && !io_write && !io_read) begin
                        state <= STATE_MAC;
                        calc_step <= 0;
                        // Pre-load biases into accumulators
                        acc[0] <= $signed({ {4{B[0][7]}}, B[0] });
                        acc[1] <= $signed({ {4{B[1][7]}}, B[1] });
                        acc[2] <= $signed({ {4{B[2][7]}}, B[2] });
                    end
                end

                STATE_MAC: begin
                    // All 3 MACs compute in parallel: acc[i] += x[calc_step] * W[i*3 + calc_step]
                    acc[0] <= acc[0] + prod_ext[0];
                    acc[1] <= acc[1] + prod_ext[1];
                    acc[2] <= acc[2] + prod_ext[2];

                    if (calc_step == 2) begin
                        state <= STATE_RELU;
                        calc_step <= 0;
                    end else begin
                        calc_step <= calc_step + 1;
                    end
                end

                STATE_RELU: begin
                    // Apply PReLU activation function to all 3 neurons in parallel
                    // Neuron 0
                    if (acc[0][11]) begin
                        case (prelu_config)
                            2'b00: y[0] <= 8'd0;
                            2'b01: y[0] <= ($signed(acc[0]) < -256) ? 8'h80 : ($signed(acc[0]) >>> 1);
                            2'b10: y[0] <= ($signed(acc[0]) < -512) ? 8'h80 : ($signed(acc[0]) >>> 2);
                            2'b11: y[0] <= ($signed(acc[0]) >>> 3);
                        endcase
                    end else if (acc[0] > 127) begin
                        y[0] <= 8'd127;
                    end else begin
                        y[0] <= acc[0][7:0];
                    end

                    // Neuron 1
                    if (acc[1][11]) begin
                        case (prelu_config)
                            2'b00: y[1] <= 8'd0;
                            2'b01: y[1] <= ($signed(acc[1]) < -256) ? 8'h80 : ($signed(acc[1]) >>> 1);
                            2'b10: y[1] <= ($signed(acc[1]) < -512) ? 8'h80 : ($signed(acc[1]) >>> 2);
                            2'b11: y[1] <= ($signed(acc[1]) >>> 3);
                        endcase
                    end else if (acc[1] > 127) begin
                        y[1] <= 8'd127;
                    end else begin
                        y[1] <= acc[1][7:0];
                    end

                    // Neuron 2
                    if (acc[2][11]) begin
                        case (prelu_config)
                            2'b00: y[2] <= 8'd0;
                            2'b01: y[2] <= ($signed(acc[2]) < -256) ? 8'h80 : ($signed(acc[2]) >>> 1);
                            2'b10: y[2] <= ($signed(acc[2]) < -512) ? 8'h80 : ($signed(acc[2]) >>> 2);
                            2'b11: y[2] <= ($signed(acc[2]) >>> 3);
                        endcase
                    end else if (acc[2] > 127) begin
                        y[2] <= 8'd127;
                    end else begin
                        y[2] <= acc[2][7:0];
                    end

                    state <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

    // ==========================================
    // Output Routing (Multiplexed output based on address)
    // ==========================================
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
