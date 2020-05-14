#include <functional>
#include<cmath>
#include "BPMN.h"

double* x;     //Input of the neural network
double* d;     //Desired output of the neural network

extern int trainDataSize;
extern int predictDataSize;

BPMN::BPMN(int level, int* num, double inital_eta , double inital_alpha ){
	this->level = level;
	this->num = num;
	this->eta = inital_eta;
	this->alpha = inital_alpha;
	
}
BPMN::BPMN(const BPMN& orig) {}

BPMN::~BPMN(){}

int BPMN::init()
{

	net = new double* [level];
	o = new double* [level];
	theta = new double* [level];
	delta = new double* [level];
	w = new double** [level - 1];
	lw = new double** [level - 1];
	for (int i = 0; i < level; i++)
	{
		net[i] = new double[num[i]];
		o[i] = new double[num[i]];
		theta[i] = new double[num[i]];
		delta[i] = new double[num[i]];
		for (int j = 0; j < num[i]; j++)
		{
			o[i][j] = 0;
			net[i][j] = 0;
			theta[i][j] = (2.0 * (double)rand() / RAND_MAX) - 1;
			delta[i][j] = 0;
		}
	}
	y = o[level - 1];
	for (int i = 0; i < (level - 1); i++)
	{
		w[i] = new double* [num[i]];
		lw[i] = new double* [num[i]];
		for (int j = 0; j < num[i]; j++)
		{
			w[i][j] = new double[num[i + 1]];
			lw[i][j] = new double[num[i + 1]];
			for (int k = 0; k < num[i + 1]; k++)
			{
				w[i][j][k] = (2.0 * (double)rand() / RAND_MAX) - 1;
				lw[i][j][k] = 0;
			}
			
		}
	}
	return 0;
}

double BPMN::sigmoid(double x)
{
	return (1.0 / (1.0 + exp(x)));
}

void BPMN:: forward(double* input)
{
	for (int i = 0; i < num[0]; i++)
	{
		net[0][i] = input[i];
		o[0][i] = input[i];
	}
	for (int i = 1; i < level; i++)
	{
		for (int j = 0; j < num[i]; j++)
		{
			net[i][j] = 0;
			for (int k = 0; k < num[i - 1]; k++)
			{
				net[i][j] += o[i - 1][k] * w[i - 1][k][j];
			}
			o[i][j] = sigmoid(-net[i][j] + theta[i][j]);
		}
	}
}

void BPMN::backward(double* d)
{
	for (int i = (level - 2); i >= 0; i--)
	{
		for (int k = 0; k < num[i + 1]; k++)
		{
			if (i == (level - 2))
			{
				delta[i][k] = (d[k] - y[k]) * y[k] * (1 - y[k]);
			}
			else {
				double E = 0;
				for (int h = 0; h < num[i + 2]; h++)
				{
					E += delta[i + 1][h] * w[i + 1][k][h];
				}
				delta[i][k] = o[i + 1][k] * (1 - o[i + 1][k]) * E;
			}
			theta[i][k] += eta * delta[i][k];
		}
	}
}

double* BPMN::predict(double* input)
{
	forward(input);
	return y;

	return 0;
}

void BPMN::train(double** input, double** real_num)
{

	for (int isamp = 0; isamp < trainDataSize; isamp++)//循环训练一次样品  
	{
		x = new double[num[0]];
		d = new double[num[-1]];

		for (int i = 0; i < num[0]; i++)
			x[i] = input[isamp][i]; //输入的样本  
		for (int m = 0; m < num[-1]; m++)
			d[m] = real_num[isamp][m]; //期望输出的样本  

		forward(x);
		backward(d);
		for (int i = (level - 2); i >= 0; i--)
		{
			for (int j = 0; j < num[i]; j++)
			{
				for (int k = 0; k < num[i + 1]; k++)
				{
					w[i][j][k] += eta * delta[i][k] * o[i][j] + alpha * lw[i][j][k];
					lw[i][j][k] = eta * delta[i][k] * o[i][j];
				}
			}
		}
		delete[] x;
		delete[] d;
	}

}

//double BPMN:: E(double* input, double* d)
//{
//	double MSE = 0;
//	double MAE = 0;
//	double MAPE = 0;
//	forward(input);
//	for (int j = 0; j < num[level - 1]; j++)
//	{
//		MSE += (y[j] - d[j]) * (y[j] - d[j]);
//		MAE += abs(y[j] - d[j]);
//		MAPE += abs(y[j] - d[j]) / d[j];
//	}
//	return sqrt(MSE);
//}

