
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

float counter;
float successcounter;

class TrainingData
{
public:
    TrainingData(const string imagefilename, const string labelfilename);
    bool isEof(void) { return m_image.eof(); }

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    void getTargetOutputs(vector<double> &targetOutputVals);
    void getFirstInputs();

private:
    ifstream m_image;
    ifstream m_label;
    char m_number;

    int pixelimage[28 + 1][28 + 1] = {0};
};


TrainingData::TrainingData(const string imagefilename, const string labelfilename)
{
    m_image.open(imagefilename.c_str(), ios::in | ios::binary);
    m_label.open(labelfilename.c_str(), ios::in | ios::binary);
}

void TrainingData::getFirstInputs() {
        char number;
    for (int i = 1; i <= 16; ++i) {
        m_image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        m_label.read(&number, sizeof(char));
	}
}
unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    for (int j = 1; j <= 28; ++j) {
        for (int i = 1; i <= 28; ++i) {
            m_image.read(&m_number, sizeof(char));
            if (m_number == 0) {
				pixelimage[i][j] = 0; 
			} else {
				pixelimage[i][j] = 1;
			}
        }
	}
	
    	// cout << "Image:" << endl;
	// for (int j = 1; j <= 28; ++j) {
	// 	for (int i = 1; i <= 28; ++i) {
	// 		cout << pixelimage[i][j];
	// 	}
	// 	cout << endl;
	// }

    for (int j = 1; j <= 28; ++j) {
        for (int i = 1; i <= 28; ++i) {
            int pos = i + (j - 1) * 28;
            inputVals.push_back(pixelimage[i][j]);
        }
	}

    return inputVals.size();
}

void TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    char number;

    targetOutputVals.clear();


    m_label.read(&number, sizeof(char));
    for (int i = 1; i <= 10; ++i) {
        if(i == (int)(number) + 1)
        {
            targetOutputVals.push_back(1.0);
        }
        else
		    targetOutputVals.push_back(0.0);
	}
    
    cout << "Label: " << (int)(number) << endl;
}


struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void loadInputWeights1(Layer &inputLayer, Layer &hiddenLayer1, Layer & outputLayer, string file_name);
    void loadInputWeights2(Layer &inputLayer, Layer &hiddenLayer1,Layer &hiddenLayer2, Layer & outputLayer, string file_name);

private:
    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return (double)(rand() % 6) / 10.0; }
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.5;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.9;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

void Neuron::loadInputWeights1(Layer &inputLayer, Layer &hiddenLayer1, Layer & outputLayer, string file_name)
{   
    ifstream file(file_name, ios::in);

    double value;
    for (unsigned n = 0; n < inputLayer.size()- 1; ++n) {
        for (unsigned m = 0; m <= hiddenLayer1.size() - 2; ++m) {
            Neuron &neuron = inputLayer[n];
            file >> value;
            neuron.m_outputWeights[m].weight = value;
            neuron.m_outputWeights[m].deltaWeight = value;
        }

    }
    for (unsigned n = 0; n < hiddenLayer1.size() -1; ++n) {
        for (unsigned m = 0; m <= outputLayer.size() - 2; ++m) {
            Neuron &neuron = hiddenLayer1[n];
            file >> value;
            neuron.m_outputWeights[m].weight = value;
            neuron.m_outputWeights[m].deltaWeight = value;
        }

    }

}
void Neuron::loadInputWeights2(Layer &inputLayer, Layer &hiddenLayer1, Layer &hiddenLayer2, Layer & outputLayer, string file_name)
{   
    ifstream file(file_name, ios::in);

    double value;
    for (unsigned n = 0; n < inputLayer.size()- 1; ++n) {
        for (unsigned m = 0; m <= hiddenLayer1.size() - 2; ++m) {
            Neuron &neuron = inputLayer[n];
            file >> value;
            neuron.m_outputWeights[m].weight = value;
            neuron.m_outputWeights[m].deltaWeight = value;
        }

    }
    for (unsigned n = 0; n < hiddenLayer1.size() -1; ++n) {
        for (unsigned m = 0; m <= hiddenLayer2.size() - 2; ++m) {
            Neuron &neuron = hiddenLayer1[n];
            file >> value;
            neuron.m_outputWeights[m].weight = value;
            neuron.m_outputWeights[m].deltaWeight = value;
        }

    }
    for (unsigned n = 0; n < hiddenLayer2.size() -1; ++n) {
        for (unsigned m = 0; m <= outputLayer.size() - 2; ++m) {
            Neuron &neuron = hiddenLayer2[n];
            file >> value;
            neuron.m_outputWeights[m].weight = value;
            neuron.m_outputWeights[m].deltaWeight = value;
        }

    }

}



double Neuron::transferFunction(double x)
{
    // tanh - output range [-1.0..1.0]

    return 1.0 / (1.0 + exp(-x));
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return x * (1.0 - x);
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}


// ****************** class Net ******************
class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }
    void load_model(string file_name);
    vector<Layer> getLayers(void) const { return m_layers; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::load_model(string file_name){
	
	// Input layer - Hidden layer

    if(m_layers.size() == 3)
    {
        unsigned layerNum = m_layers.size() - 3;
        Layer &inputlayer = m_layers[layerNum];
        Layer &hiddenLayer1 = m_layers[layerNum + 1];
        Layer &outputLayer = m_layers[layerNum + 2];


        inputlayer[layerNum].loadInputWeights1(inputlayer, hiddenLayer1, outputLayer, file_name);
    }
    else if(m_layers.size() == 4)
    {
        unsigned layerNum = m_layers.size() - 4;
        Layer &inputlayer = m_layers[layerNum ];
        Layer &hiddenLayer1 = m_layers[layerNum +1];
        Layer &hiddenLayer2 = m_layers[layerNum +2];
        Layer &outputLayer = m_layers[layerNum + 3];


        inputlayer[layerNum].loadInputWeights2(inputlayer, hiddenLayer1, hiddenLayer2, outputLayer, file_name);
    }
    
}

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));

        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0);
    }
}

int testvalue;
void showVectorVals(string label, vector<double> &v)
{
    testvalue = 0;
    int value = 0;
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
        ++value;
        if(v[i] == 1)
        {
            testvalue  = value - 1;
        }
    }

    cout << endl;
}

void isTargetEqualtoResult(vector<double> &t, vector<double> &r)
{

    int count;

    int predict = 0;
        for (int i = 1; i <= 9; ++i) {
			if (r[i] > r[predict]) {
				predict = i;
			}
		}
    
    counter++;

    if(predict == testvalue)
    {
            successcounter++;
            predict = 1;
    }
    else
{

}

}

float divide(int x, int y){return float(x) / float(y);}


void printSuccessrate()
{
    float result = divide(successcounter, counter) * 100;
    cout << endl << "Successrate: " << result << "%" << endl;
}

int main(int argc, char* argv[])
{   
    TrainingData testData("mnist/t10k-images.idx3-ubyte","mnist/t10k-labels.idx1-ubyte");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    topology.push_back(784);
    topology.push_back(atoi(argv[1]));
    topology.push_back(atoi(argv[2]));
    topology.push_back(10);

    // Reading file headers
    testData.getFirstInputs();
    Net myNet(topology);


    vector<double> inputVals, targetVals, resultVals;
    myNet.load_model("weight.txt");

    for(int testingEpoch = 1; testingEpoch <= 10000; ++testingEpoch)
    {
        cout << endl << "Epoch " << testingEpoch << endl;
        
        testData.getNextInputs(inputVals);

        //showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        testData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        isTargetEqualtoResult(targetVals, resultVals);
    }
    printSuccessrate();

    cout << endl << "Done" << endl;
    return 0;
}

