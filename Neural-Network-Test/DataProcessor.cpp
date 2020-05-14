/* 
 * File:   Preprocessor.cpp
 * Author: heshan
 * 
 * Created on April 20, 2018, 8:00 PM
 */

#include "DataProcessor.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

DataProcessor::DataProcessor() {
	MAX = 0;
	MIN = 0;
}

DataProcessor::DataProcessor(const DataProcessor& orig) { }

DataProcessor::~DataProcessor() { }

std::vector<double> DataProcessor::process(std::vector<double> vec, int vecType) {
	
	
	std::vector<double>::iterator max = std::max_element(vec.begin(), vec.end());
	std::vector<double>::iterator min = std::min_element(vec.begin(), vec.end());
	double maxnum = *max;
	double minnum = *min;
	for (std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it) 
	{
		*it = (*it-minnum)/(maxnum-minnum);
	}
	if (vecType == 1)
	{
		MAX = maxnum;
		MIN = minnum;
	}
	return vec;
}

std::vector<double> DataProcessor::postprocess(std::vector<double> vec) {

//    std::cout<<"post processing...\n";
    
    for(std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it) {
        *it =*it*(MAX -MIN)+MIN;
    }
    return vec;
}

double DataProcessor::postProcess(double val) {

//    std::cout<<"post processing...\n";
//    std::cout<<"\n"<<out_magnitude<<"******"<<"\n";
    return val* (MAX - MIN) + MIN;
}

int DataProcessor::printVector(std::vector<double> vec){
    
    for(std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it<<", ";
    }
    std::cout<<std::endl;
    return 0;
}
