# Timing constraints for the PIM logic synthesis (1 GHz PE clock domain).
create_clock -name clk -period 1.0 [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
set_clock_transition  0.04 [get_clocks clk]

set_input_delay  0.20 -clock clk [remove_from_collection [all_inputs] \
                                  [get_ports clk]]
set_output_delay 0.20 -clock clk [all_outputs]

set_max_area 0
set_max_fanout 16 [current_design]
set_max_transition 0.15 [current_design]
