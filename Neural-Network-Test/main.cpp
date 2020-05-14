#define _CRT_SECURE_NO_WARNINGS
#include <time.h>
#include <vector>
#include <iostream>
#include <math.h>

#include "LSTMNet.h"
#include "BPMN.h"
#include "DataProcessor.h"
#include "FileProcessor.h"

using namespace std;

////////////////////////////////////////LSTM Predict/////////////////////////////////////

int LSTM() {

	int memCells = 8; // number of memory cells
	int trainDataSize = 1256; // train data size,trainDataSize = 47(day),1256(hour)
	int inputVecSize = 24; // input vector size,inputVecSize = 3(day),24(hour)
	int timeSteps = 24; // unfolded time steps,timeSteps = 3(day),24(hour)
	float learningRate = 0.0001;
	int predictions = 296; // prediction points,predictions = 16(day),296(hour)
	int iterations = 10; // training iterations with training data
	

	// Adding the time series in to a vector and preprocessing
	DataProcessor* dataproc;
	dataproc = new DataProcessor();
	FileProcessor* fileProc;
	fileProc = new FileProcessor();
	std::vector<double> timeSeries;


	///////////// Data Sets //////////////////////////////

	timeSeries = fileProc->read("dataset/input/InternetTraff.txt" , 1);
	//timeSeries = fileProc->read("dataset/input/daily_traffic.txt" , 1);
	timeSeries = dataproc->process(timeSeries, 1);

	// Creating the input vector Array
	std::vector<double>* input;
	input = new std::vector<double>[trainDataSize];
	std::vector<double> inputVec;

	for (int i = 0; i < trainDataSize; i++) {
		inputVec.clear();
		for (int j = 0; j < inputVecSize; j++) {
			inputVec.push_back(timeSeries.at(i + j));
		}
		//inputVec = dataproc->process(inputVec, 0);
		input[i] = inputVec;
	}


	// Creating the target vector using the time series 
	std::vector<double>::const_iterator first = timeSeries.begin() + inputVecSize;
	std::vector<double>::const_iterator last = timeSeries.begin() + inputVecSize + trainDataSize;
	std::vector<double> targetVector(first, last);

	// Training the LSTM net
	LSTMNet lstm(memCells, inputVecSize);
	lstm.train(input, targetVector, trainDataSize, timeSteps, learningRate, iterations);

	// Open the file to write the time series predictions
	std::ofstream out_file;
	out_file.open("dataset/prediction/LSTMNet.txt", std::ofstream::out | std::ofstream::trunc);

	std::vector<double> inVec;
	input = new std::vector<double>[1];
	double result;
	double expected;
	double MSE = 0;
	double MAE = 0;
	double MAPE = 0;

	std::cout << std::fixed;

	for (int i = 0; i < predictions; i++) {

		inVec.clear();
		for (int j = 0; j < inputVecSize; j++) 
		{
			inVec.push_back(timeSeries.at(i + j + inputVecSize + trainDataSize));
		}

		//inVec = dataproc->process(inVec, 0);
		input[0] = inVec;

		result = lstm.predict(input);
		//std::cout << std::endl << "result: " << result << std::endl;
		expected = timeSeries.at(i + inputVecSize * 2 + trainDataSize);
		
		result = dataproc->postProcess(result);
		expected = dataproc->postProcess(expected);

		MSE += std::pow((expected - result), 2);
		MAE += abs(expected - result);
		MAPE += abs((expected - result) / result);

		out_file << result << "\n";
		//std::cout << "result processed: " << result << std::endl << std::endl;
	}

	double RMSE = std::pow(MSE/predictions,0.5);
	MAE /= predictions;
	MAPE = MAPE / predictions  ;
	std::cout << "Root Mean Squared Error: " << RMSE << "\n";
	std::cout << "Mean Absolute Error: " << MAE << "\n";
	std::cout << "Mean Absolute Percentage Error: " << MAPE << "\n";
	std::cout << std::scientific;
	return 0;
}

////////////////////////////////////Multi-layer BPNN and RBFNN//////////////////////////

int trainDataSize = 1256; // train data size,trainDataSize = 47(day),1256(hour)
int predictDataSize = 296; // prediction points,predictions = 16(day),296(hour)
const int total = 1657; // total data size,total = 69(day),1657(hour)

double MAX, MIN;

double** X; // input data
double** Y;  //desire output data
double* predict_data;
double* std_predict_data;


int level = 3;
int num[] = { 24,16,1 };


void data_process()
{
	//initialize the X and Y
	X = new double* [trainDataSize];
	Y = new double* [trainDataSize];
	for (int i = 0; i < trainDataSize; i++) {
		X[i] = new double[num[0]]();
		Y[i] = new double[num[-1]]();
	}
	
	predict_data = new double[total]();
	std_predict_data = new double[total]();

	//read the traffic intensity
	FILE* stream;
	if ((stream = fopen("dataset/input/InternetTraff.txt", "r")) == NULL)
	//if ((stream = fopen("dataset/input/daily_traffic.txt", "r")) == NULL)
	{
		cout << "打开文件失败!";
		exit(1);
	}
	float* predict_datax=new float[total];
	
	for (int i = 0; i < total; i++)
	{
		fscanf(stream, "%f", &predict_datax[i]);
		predict_data[i] = predict_datax[i];

	}
	fclose(stream);

	delete[] predict_datax;

	//find the max and min
	double max_data = predict_data[0];
	double min_data = predict_data[0];
	for (int i = 0; i < total; i++)
	{
		if (predict_data[i] > max_data)
		{
			max_data = predict_data[i];
		}
		if (predict_data[i] < min_data)
		{
			min_data = predict_data[i];
		}
	}

	MAX = max_data;
	MIN = min_data;

	//Normalize
	for (int i = 0; i < total; i++)
	{
		std_predict_data[i] = (predict_data[i] -MIN) / (MAX - MIN);
	}
	
	double* temp=new double[num[0]];
	for (int i = 0; i < trainDataSize; i++)
	{
		int count = 0;
		for (int j = i; j < i + num[0]; j++)
		{
			temp[count++] =std_predict_data[j];
		}
		for (int k = 0; k < num[0]; k++)
		{
			X[i][k] = temp[k];
		}
	}

	for (int m = num[0]; m < trainDataSize + num[0]; m++)
	{
		for (int k = 0; k < num[-1]; k++) 
		{
			Y[m - num[0]][k] = std_predict_data[m+k];
		}
	}
}


int Multi_layer_BPNN() {

	double MSE = 0;
	double MAE = 0;
	double MAPE = 0;

	data_process();

	BPMN bpmn(level, num);
	
	bpmn.init();

	for (int time = 0; time < 500; time++)
	{
		bpmn.train(X, Y);
	}

	double* prediction=new double[num[0]];
	std::ofstream out_file;
	out_file.open("dataset/prediction/BpNet.txt", std::ofstream::out | std::ofstream::trunc);
	for (int i = 0; i < predictDataSize; i++)
	{
		int count = 0;
		for (int j = i + num[0] + trainDataSize; j < i + num[0] * 2 + trainDataSize; j++)
		{
			prediction[count++] = std_predict_data[j];
		}
		double* result;

		result = bpmn.predict(prediction);

		for (int j = 0; j < num[level - 1]; j++)
		{
			result[j] = result[j] * (MAX - MIN) + MIN;
		}

		for (int j = 0; j < num[level - 1]; j++)
		{
			MSE += pow((result[j] - predict_data[j + i + num[0] * 2 + trainDataSize]),2);
			MAE += abs(result[j] - predict_data[j + i + num[0] * 2 + trainDataSize]);
			MAPE += abs(result[j] - predict_data[j + i + num[0] * 2 + trainDataSize]) / predict_data[j + i + num[0] * 2 + trainDataSize];
		}
		out_file << *result << endl;

	

	}
	double RMSE = pow(MSE / predictDataSize, 0.5);
	MAE /= predictDataSize;
	MAPE /= predictDataSize;
	cout << "RMSE:" << RMSE << '\n';
	cout << "MAE:" << MAE << '\n';
	cout << "MAPE:" << MAPE << '\n';
	return 0;

}


int main() {

	clock_t startTime_LSTM, endTime_LSTM;
	startTime_LSTM = clock();//计时开始

	// use LSTM to predict univariate time series
	LSTM();

	endTime_LSTM = clock();//计时结束
	std::cout << "LSTM:The run time is: " << float(endTime_LSTM - startTime_LSTM) / CLOCKS_PER_SEC << "s" << '\n';

	clock_t startTime_BPMN, endTime_BPMN;
	startTime_BPMN = clock();//计时开始

	// use BPMN to predict univariate time series
	Multi_layer_BPNN();

	endTime_BPMN = clock();//计时结束
	std::cout << "BPMN:The run time is: " << float(endTime_BPMN - startTime_BPMN) / CLOCKS_PER_SEC << "s";
}
