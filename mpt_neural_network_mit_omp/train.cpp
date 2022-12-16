

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

using namespace std;


const int width = 28;
const int height = 28;   
const int trainingsize = 60000;
const int NUM_THREADS = 12;
double globalweights1[784][128];
double globalweights2[128][128];
double globalweights3[128][10];

class Neuron;

typedef vector<Neuron> Layer;

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

    int pixelimage[width + 1][height + 1] = {0};
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

    char number;
    for (int j = 1; j <= width; ++j) {
        for (int i = 1; i <= height; ++i) {
            m_image.read(&number, sizeof(char));
            if (number == 0) {
				pixelimage[i][j] = 0; 
			} else {
				pixelimage[i][j] = 1;
			}
        }
	}

    for (int j = 1; j <= width; ++j) {
        for (int i = 1; i <= height; ++i) {
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

}


struct Connection
{
    double weight;
    double deltaWeight;
};


// ****************** class Neuron ******************
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    vector<Connection> getm_outputWeights() { return m_outputWeights; };
    unsigned getm_myIndex() { return m_myIndex; };
    double getWeight(int b, Neuron neuron);
    void updateGlobalWeights(Layer &prevLayer);


private:
    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) 
{
    int sign = rand() % 2;
    if(sign == 1)
    {
        double value = (double)(rand() % 6) / 10.0;
        return -value;
    } else
    {
        double value = (double)(rand() % 6) / 10.0;
        return value;
    }
}
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.035;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.9;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

void Neuron::updateGlobalWeights(Layer &prevLayer)
{
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            // Individual input, magnified by the gradient and train rate:
            eta
            * neuron.getOutputVal()
            * m_gradient
            // Also add momentum = a fraction of the previous delta weight:
            + alpha
            * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::getWeight(int b, Neuron neuron)
{
     double weightInput;

    weightInput = neuron.m_outputWeights[b].weight;

    return weightInput;
}


void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sumdow = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sumdow += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sumdow;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);

   // m_gradient = Neuron::transferFunctionDerivative(m_outputVal);

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
    double sumff = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n) {

        sumff += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sumff);
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
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }
    vector<Layer> getLayers(void) const { return m_layers; }
    void getweight(vector<Layer> m_layers);
    void setweight(vector<Layer> m_layers);

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::setweight(vector<Layer> m_layers)
{
    for(unsigned i = 0; i < m_layers.size() -2; i++)
    {
        for(unsigned j = 0; j < m_layers[i].size() -1; j++)
        {
            for(unsigned o = 0; o < m_layers[i][j].getm_outputWeights().size() -1; o++)
            m_layers[i][j].getm_outputWeights()[o].weight = 0.0;
        }
    }
}
void Net::getweight(vector<Layer> m_layers1)
{
    Layer inputLayer;
    Layer hiddenLayer1;
    Layer hiddenLayer2;
    Layer outputLayer;
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) 
    {
        Layer &layer = m_layers[layerNum];
        Layer &layer1 = m_layers1[layerNum];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {

            layer[n].getm_outputWeights()[layer[n].getm_myIndex()].weight += layer1[n].getm_outputWeights()[layer1[n].getm_myIndex()].weight;
            
        }
    }

}

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {

        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate hidden layer gradients

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {

        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {

        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }

}

void Net::feedForward(const vector<double> &inputVals)
{

    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward feed

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

class WeightsData
{
public:
    WeightsData(const string Path);
    void saveinFile(vector<Layer> m_layers);
    void write_matrix(vector<Layer> m_layers);

private:
    ofstream m_weightFile;
};

WeightsData::WeightsData(const string Path)
{
    m_weightFile.open(Path.c_str(), ios::out);

}

void WeightsData::write_matrix(vector<Layer> m_layers) 
{
   
    Layer inputLayer;
    Layer hiddenLayer1;
    Layer hiddenLayer2;
    Layer outputLayer;

    if(m_layers.size() == 3)
    {
        inputLayer = m_layers[m_layers.size() - 3];
        hiddenLayer1 = m_layers[m_layers.size()- 2];
        outputLayer = m_layers[m_layers.size() - 1];

        for (long long unsigned int n = 0; n < inputLayer.size() - 1; ++n) 
        {

            for (long long unsigned int b = 0; b < hiddenLayer1.size() - 1; ++b) 
            {
                m_weightFile << globalweights1[n][b] / 53998 << " ";
            }
        m_weightFile << endl;
        }

        for (long long unsigned int n = 0; n < hiddenLayer1.size(); ++n) 
        {

            for (long long unsigned int b = 0; b < outputLayer.size(); ++b) 
            {
                m_weightFile << globalweights2[n][b] << " ";
            }
        m_weightFile << endl;
        }
    }

    else if( m_layers.size() == 4)
    {
        inputLayer = m_layers[m_layers.size() - 4];
        hiddenLayer1 = m_layers[m_layers.size() - 3];
        hiddenLayer2 = m_layers[m_layers.size() - 2];
        outputLayer = m_layers[m_layers.size() - 1];

        for (long long unsigned int n = 0; n < inputLayer.size() - 1; ++n) 
        {

            for (long long unsigned int b = 0; b < hiddenLayer1.size() - 1; ++b) 
            {
                m_weightFile << globalweights1[n][b] << " ";
            }
        m_weightFile << endl;
        }
        for (long long unsigned int n = 0; n < hiddenLayer1.size() -1; ++n) 
        {

            for (long long unsigned int b = 0; b < hiddenLayer2.size() -1; ++b) 
            {
                m_weightFile << globalweights2[n][b] << " ";
            }
        m_weightFile << endl;
        }

        for (long long unsigned int n = 0; n < hiddenLayer2.size()-1; ++n) 
        {

            for (long long unsigned int b = 0; b < outputLayer.size()-1; ++b) 
            {
                m_weightFile << globalweights3[n][b] << " ";
            }
        m_weightFile << endl;
        }
    }

    m_weightFile.close();
}

void WeightsData::saveinFile(vector<Layer> m_layers)
{
   
    Layer inputLayer;
    Layer hiddenLayer1;
    Layer hiddenLayer2;
    Layer outputLayer;

    if(m_layers.size() == 3)
    {
        inputLayer = m_layers[m_layers.size() - 3];
        hiddenLayer1 = m_layers[m_layers.size()- 2];
        outputLayer = m_layers[m_layers.size() - 1];

        for (long long unsigned int n = 0; n < inputLayer.size() - 1; ++n) 
        {
            Neuron &neuron = inputLayer[n];
            for (long long unsigned int b = 0; b < hiddenLayer1.size() - 1; ++b) 
            {
                m_weightFile << neuron.getWeight(b, neuron) << " ";
            }
        m_weightFile << endl;
        }

        for (long long unsigned int n = 0; n < hiddenLayer1.size(); ++n) 
        {
            Neuron &neuron1 = hiddenLayer1[n];
            for (long long unsigned int b = 0; b < outputLayer.size(); ++b) 
            {
                m_weightFile << neuron1.getWeight(b, neuron1) << " ";
            }
        m_weightFile << endl;
        }
    }

    else if( m_layers.size() == 4)
    {
        inputLayer = m_layers[m_layers.size() - 4];
        hiddenLayer1 = m_layers[m_layers.size() - 3];
        hiddenLayer2 = m_layers[m_layers.size() - 2];
        outputLayer = m_layers[m_layers.size() - 1];

        for (long long unsigned int n = 0; n < inputLayer.size() - 1; ++n) 
        {
            Neuron &neuron = inputLayer[n];
            for (long long unsigned int b = 0; b < hiddenLayer1.size() - 1; ++b) 
            {
                m_weightFile << neuron.getWeight(b, neuron) << " ";
            }
        m_weightFile << endl;
        }
        for (long long unsigned int n = 0; n < hiddenLayer1.size() -1; ++n) 
        {
            Neuron &neuron1 = hiddenLayer1[n];
            for (long long unsigned int b = 0; b < hiddenLayer2.size() -1; ++b) 
            {
                m_weightFile << neuron1.getWeight(b, neuron1) << " ";
            }
        m_weightFile << endl;
        }

        for (long long unsigned int n = 0; n < hiddenLayer2.size()-1; ++n) 
        {
            Neuron &neuron2 = hiddenLayer2[n];
            for (long long unsigned int b = 0; b < outputLayer.size()-1; ++b) 
            {
                m_weightFile << neuron2.getWeight(b, neuron2) << " ";
            }
        m_weightFile << endl;
        }
    }

    m_weightFile.close();
}

void getGlobalWeights(vector<Layer> m_layers)
{
   
    Layer inputLayer;
    Layer hiddenLayer1;
    Layer hiddenLayer2;
    Layer outputLayer;

    if(m_layers.size() == 3)
    {
        inputLayer = m_layers[m_layers.size() - 3];
        hiddenLayer1 = m_layers[m_layers.size()- 2];
        outputLayer = m_layers[m_layers.size() - 1];

        for (long long unsigned int n = 0; n < inputLayer.size() - 1; ++n) 
        {
            Neuron &neuron = inputLayer[n];
            for (long long unsigned int b = 0; b < hiddenLayer1.size() - 1; ++b) 
            {

                globalweights1[n][b] += neuron.getWeight(b, neuron);
            }

        }

        for (long long unsigned int n = 0; n < hiddenLayer1.size(); ++n) 
        {
            Neuron &neuron1 = hiddenLayer1[n];
            for (long long unsigned int b = 0; b < outputLayer.size(); ++b) 
            {

                globalweights2[n][b] += neuron1.getWeight(b, neuron1);
            }

        }
    }

    else if( m_layers.size() == 4)
    {
        inputLayer = m_layers[m_layers.size() - 4];
        hiddenLayer1 = m_layers[m_layers.size() - 3];
        hiddenLayer2 = m_layers[m_layers.size() - 2];
        outputLayer = m_layers[m_layers.size() - 1];

        for (long long unsigned int n = 0; n < inputLayer.size() - 1; ++n) 
        {
            Neuron &neuron = inputLayer[n];
            for (long long unsigned int b = 0; b < hiddenLayer1.size() - 1; ++b) 
            {
  
                globalweights1[n][b] += neuron.getWeight(b, neuron);
            }

        }
        for (long long unsigned int n = 0; n < hiddenLayer1.size() -1; ++n) 
        {
            Neuron &neuron1 = hiddenLayer1[n];
            for (long long unsigned int b = 0; b < hiddenLayer2.size() -1; ++b) 
            {
      
                globalweights2[n][b] += neuron1.getWeight(b, neuron1);
            }

        }

        for (long long unsigned int n = 0; n < hiddenLayer2.size()-1; ++n) 
        {
            Neuron &neuron2 = hiddenLayer2[n];
            for (long long unsigned int b = 0; b < outputLayer.size()-1; ++b) 
            {
        
                globalweights3[n][b] += neuron2.getWeight(b, neuron2);
            }

        }
    }

}
int main(int argc, char* argv[])
{   

    auto start = chrono::high_resolution_clock::now();
    // Net globalNet({784,128,128,10});
    // globalNet.setweight(globalNet.getLayers());
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel 
    {

    ofstream error;
    error.open("errorplottan.txt", ios::out);

    vector<unsigned> topology;

    topology.push_back(784);
    topology.push_back(atoi(argv[1]));
    topology.push_back(atoi(argv[2]));
    topology.push_back(10);
    TrainingData trainData("mnist/train-images.idx3-ubyte","mnist/train-labels.idx1-ubyte");
    
    // Reading file headers
    trainData.getFirstInputs();
    vector<double> inputVals, targetVals, resultVals;
    
    Net myNet(topology);
    
        #pragma omp for
        for( int trainingEpoch = 1; trainingEpoch <= trainingsize; ++trainingEpoch)
        {
            trainData.getNextInputs(inputVals);

            myNet.feedForward(inputVals);

            myNet.getResults(resultVals);
            
            trainData.getTargetOutputs(targetVals);
            assert(targetVals.size() == topology.back());

            myNet.backProp(targetVals);
            
            // getGlobalWeights(myNet.getLayers());

            // if(trainingEpoch % 100 == 0)
            // {   
            //     error << trainingEpoch << ",";
            //     error << myNet.getRecentAverageError() << endl;
            // }
        }
        // WeightsData weightFile("weight.txt");
        // weightFile.saveinFile(myNet.getLayers());
    }
    // WeightsData weightFile("weight.txt");
    // weightFile.write_matrix(globalNet.getLayers());

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    // printf("%lf   \"%s\"\n", elapsed.count(), CFLAGS);
    printf("%lf s  \n", elapsed.count());

}
