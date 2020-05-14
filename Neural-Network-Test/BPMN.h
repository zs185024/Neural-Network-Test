#ifndef BPMN_H
#define BPMN_H


class BPMN{
public:
	BPMN(int level, int* num, double inital_eta = 0.001, double inital_alpha = 0.0001);
	BPMN(const BPMN& orig);
	virtual ~BPMN();
	
	/**
	* Initialize the NN
	* Random initialization
	 *
	* @return 0
	*/
	int init();

	/**
    * 
    * @param input: training data set(two dimensions)
    * @param real_num: target values(two dimensions)
	* @return 0
	*
	*/
	void train(double** input, double** real_num);

	/**
	* predict a future point using a input vector of size n
	*
	* input: {t-n,...,t-2,t-1,t}
	* result: {t+1}
	*
	* @param x: input vector for the prediction
	* @return result: predicted value
	*/
	double* predict(double* x);

	/**
	* calculate the error between output and target
	*@param x:the input(one dimension)
	*@param d:the output(one dimension)
	*@return result:E
	*/
	double E(double* x, double* d);

private:
	/**
	* calculate the output
	*@param x:the input (one dimension)
	*/
	void forward(double* x);
	
	/**
	* calculate the delta and theta£¨BP)
	*@param d:the output(one dimension)
	*/
	void backward(double* d);
	
	/**
	* Sigmoid function
	*
	* @param x
	* @return value
	*/
	double sigmoid(double x);
	
	
private:
	int level;   //Number of neuron layer
	int* num;    //Number of neurons in each neuron layer
	float eta;   //Learning efficiency
	float alpha; //Weight of connection weights on last time

	double* y;     //the final output of neural network

	double** net;   //General output of neurons
	double** o;     //Output of neurons(sigmoid(net))
	double** theta; //Threshold of neurons
	double** delta; //Process variable which is in order to calculate delta theta and delta w

	double*** w; //the connection weights
	double*** lw; //Connection weights on last time
	
	
};


#endif