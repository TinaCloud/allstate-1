//
//  input_output.cpp
//  allstate-markov-model
//
//  Created by Brandon Kelly on 5/18/14.
//  Copyright (c) 2014 Brandon Kelly. All rights reserved.
//

#include "input_output.hpp"

// return the number of lines in a text file
int get_file_lines(std::string& filename) {
    int number_of_lines = 0;
    std::string line;
    std::ifstream inputfile(filename.c_str());
    
    if (inputfile.good()) {
        while (std::getline(inputfile, line))
            ++number_of_lines;
        inputfile.close();
        return number_of_lines;
	} else {
		std::string errmsg("File ");
		errmsg.append(filename);
		errmsg.append(" does not exist.\n");
		throw std::runtime_error(errmsg);
	}
}

// read in the data
void read_data(std::string& filename, std::vector<std::vector<unsigned int> >& data, int nrows, int ncols) {
	std::ifstream input_file(filename.c_str());
    data.resize(nrows);
	for (int i = 0; i < nrows; ++i) {
		std::vector<unsigned int> this_data(ncols);
		for (int j = 0; j < ncols; ++j) {
			input_file >> this_data[j];
		}
		data[i] = this_data;
	}
	input_file.close();
}

void read_markov_data(std::string& mfilename, std::string& tfilename, std::vector<std::vector<int> >& data, int nrows) {
    std::ifstream input_file(mfilename.c_str());
    data.resize(nrows);
    // first get chain lengths
    std::vector<int> lengths(nrows);
    std::ifstream length_file(tfilename.c_str());
    for (int i=0; i<nrows; i++) {
        length_file >> lengths[i];
    }
    length_file.close();
    
	for (int i = 0; i < nrows; ++i) {
		std::vector<int> this_data(lengths[i]);
		for (int j = 0; j < lengths[i]; ++j) {
			input_file >> this_data[j];
		}
		data[i] = this_data;
	}
	input_file.close();

}