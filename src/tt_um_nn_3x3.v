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
    assign uio_oe  = 8'b0000_0000; // Όλα τα uio ως inputs
    assign uio_out = 8'b0;

    // Control signals from uio_in
    wire [3:0] io_addr  = uio_in[3:0]; // 0..14=Write Addr, 15=Start Calc
    wire       io_write = uio_in[4];   // 1=Write to selected address
    wire       io_read  = uio_in[5];   // 1=Read outputs
    
    // Suppress unused signals warning
    wire _unused = &{1'b0, uio_in[7:6], 1'b0};

    // ==========================================
    // Registers & Storage (Optimized for 1x1 Tile area limits)
    // ==========================================
    reg signed [4:0] x [0:2]; // 3 x 5-bit Inputs
    reg signed [7:0] y [0:2]; // 3 x 8-bit Outputs

    // Programmable Weights and Biases (Reduced to 5-bit to fit density)
    reg signed [4:0] W [0:8]; // Weights (5-bit: -16 to +15)
    reg signed [7:0] B [0:2]; // Biases (8-bit)

    // ==========================================
    // The Single Shared MAC Unit Engine
    // ==========================================
    reg signed [11:0] accumulator; // 12-bit is enough for 5x5 + 5x5 + 5x5 + bias
    reg [1:0] calc_step;
    reg [1:0] current_neuron;

    // Resolve the Weight index dynamically and multiply
    wire [3:0] weight_idx = ({2'b00, current_neuron} * 4'd3) + {2'b00, calc_step};
    wire signed [9:0] product = x[calc_step] * W[weight_idx];
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
        end else if (ena) begin
            // 1) Input Data Loading (Όταν είμαστε σε IDLE)
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
                    default: ; // 15 is Start, no write
                endcase
            end

            // 2) FSM Core για υπολογισμούς NN
            case (state)
                STATE_IDLE: begin
                    if (io_addr == 4'd15 && !io_write && !io_read) begin // Trigger Start Calculation
                        state <= STATE_MAC;
                        calc_step <= 0;
                        current_neuron <= 0;
                        accumulator <= $signed({ {4{B[0][7]}}, B[0] }); // Φόρτωση αρχικού Bias
                    end
                end

                STATE_MAC: begin
                    // Accumulator += X[step] * W[idx]
                    accumulator <= accumulator + $signed({ {2{product[9]}}, product });

                    if (calc_step == 2) begin
                        state <= STATE_RELU;
                        calc_step <= 0;
                    end else begin
                        calc_step <= calc_step + 1;
                    end
                end

                STATE_RELU: begin
                    // ReLU Activation (Αν το MSB είναι 1 (αρνητικό), τότε 0. Αλλιώς saturate αν ξεπερνά τα όρια)
                    if (accumulator[11]) begin
                        y[current_neuron] <= 8'd0;
                    end else if (accumulator > 127) begin
                        y[current_neuron] <= 8'd127; // Saturation positive     
                    end else begin
                        y[current_neuron] <= accumulator[7:0];
                    end

                    // Check if more neurons to calculate
                    if (current_neuron == 2) begin
                        state <= STATE_IDLE; // Τέλος δικτύου
                    end else begin
                        current_neuron <= current_neuron + 1;
                        accumulator <= $signed({ {4{B[current_neuron + 1][7]}}, B[current_neuron + 1] }); // Pre-load next Bias
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
