############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project Harris
set_top harris_top
add_files Harris/src/harris.hpp
add_files Harris/src/top.cpp
add_files Harris/src/top.hpp
add_files -tb Harris/testbench/tb.cpp
add_files -tb Harris/Testpictures
open_solution "solution1"
set_part {xc7z020-clg400-1}
create_clock -period 10 -name default
#source "./Harris/solution1/directives.tcl"
csim_design -argv {150}
csynth_design
cosim_design
export_design -format ip_catalog
