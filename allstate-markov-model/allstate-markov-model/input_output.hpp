//
//  input_output.hpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/18/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#ifndef allstate_markov_model_input_output_hpp
#define allstate_markov_model_input_output_hpp

#include <fstream>
#include <iostream>
#include <vector>

int get_file_lines(std::string& filename);

void read_data(std::string& filename, std::vector<std::vector<int> >& data, int nrows, int ncols);

void read_markov_data(std::string& mfilename, std::string& tfilename, std::vector<std::vector<int> >& data, int nrows);

#endif
