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
    reg signed [5:0] x [0:2]; // 3 x 6-bit Inputs
    reg signed [7:0] y [0:2]; // 3 x 8-bit Outputs

    // Programmable Weights and Biases (6-bit to fit density)
    reg signed [5:0] W [0:8]; // Weights (6-bit: -32 to +31)
    reg signed [7:0] B [0:2]; // Biases (8-bit)
    reg [1:0] prelu_config [0:2]; // Per-neuron config (00: standard ReLU, 01: x/2, 10: x/4, 11: x/8)

    // ==========================================
    // Single Shared MAC Unit Engine
    // ==========================================
    // 6-bit x 6-bit = 12-bit product. 
    // 3 MACs + 8-bit bias + safety logic = 14-bit accumulator needed
    reg signed [13:0] accumulator; 
    reg [1:0] calc_step;
    reg [1:0] current_neuron;

    // Resolve the Weight index dynamically and multiply
    wire [3:0] weight_idx = ({2'b00, current_neuron} * 4'd3) + {2'b00, calc_step};
    wire signed [11:0] product = x[calc_step] * W[weight_idx];
    
    // Saturation Logic for MAC Addition
    wire signed [13:0] product_ext = $signed({ {2{product[11]}}, product });
    wire signed [13:0] next_acc_calc = accumulator + product_ext;
    wire underflow = (accumulator[13] & product[11] & ~next_acc_calc[13]);     
    wire overflow  = (~accumulator[13] & ~product[11] & next_acc_calc[13]);    
    
    wire signed [13:0] next_acc_safe = overflow  ? 14'h1FFF : 
                                       underflow ? 14'h2000 : 
                                       next_acc_calc[13:0];
                                       
    // Shifted values for PReLU to avoid linting width warnings
    wire [7:0] s_acc_1 = accumulator[8:1];
    wire [7:0] s_acc_2 = accumulator[9:2];
    wire [7:0] s_acc_3 = accumulator[10:3];

    localparam STATE_IDLE = 2'b00;
    localparam STATE_MAC  = 2'b01;
    localparam STATE_RELU = 2'b10;
    reg [1:0] state;

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
                    // Accumulator saturating addition
                    accumulator <= next_acc_safe;

                    if (calc_step == 2) begin
                        state <= STATE_RELU;
                        calc_step <= 0;
                    end else begin
                        calc_step <= calc_step + 1;
                    end
                end

                STATE_RELU: begin
                    // Programmable PReLU Activation
                    if (accumulator[13]) begin
                        case (prelu_config[current_neuron])
                            2'b00: y[current_neuron] <= 8'd0; // Standard ReLU
                            2'b01: y[current_neuron] <= ($signed(accumulator) < -256) ? 8'h80 : s_acc_1; // x/2
                            2'b10: y[current_neuron] <= ($signed(accumulator) < -512) ? 8'h80 : s_acc_2; // x/4
                            2'b11: y[current_neuron] <= ($signed(accumulator) < -1024) ? 8'h80 : s_acc_3; // x/8
                        endcase
                    end else if (accumulator > 127) begin
                        y[current_neuron] <= 8'd127; // Saturation positive     
                    end else begin
                        y[current_neuron] <= accumulator[7:0];
                    end

                    // Check if more neurons to calculate
                    if (current_neuron == 2) begin
                        state <= STATE_IDLE; // End of network
                    end else begin
                        current_neuron <= current_neuron + 1;
                        accumulator <= $signed({ {6{B[current_neuron + 1][7]}}, B[current_neuron + 1] }); // Pre-load next Bias
                        state <= STATE_MAC;
                    end
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
