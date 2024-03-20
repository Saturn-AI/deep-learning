#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

// Activation function (sigmoid)
MatrixXd sigmoid(MatrixXd z) {
    return 1 / (1 + (-z.array()).exp());
}

// Derivative of the activation function (sigmoid derivative)
MatrixXd sigmoid_derivative(MatrixXd z) {
    MatrixXd sig = sigmoid(z);
    return sig.array() * (1 - sig.array());
}

// Neural Network class
class NeuralNetwork {
private:
    vector<MatrixXd> weights;
    vector<MatrixXd> biases;
    int num_layers;
    
public:
    NeuralNetwork(vector<int> layers) {
        num_layers = layers.size();
        
        // Initialize weights and biases
        for (int i = 1; i < num_layers; ++i) {
            weights.push_back(MatrixXd::Random(layers[i], layers[i-1]));
            biases.push_back(MatrixXd::Random(layers[i], 1));
        }
    }
    
    // Forward propagation
    MatrixXd feedforward(MatrixXd input) {
        for (int i = 0; i < num_layers - 1; ++i) {
            input = sigmoid((weights[i] * input).colwise() + biases[i]);
        }
        return input;
    }
    
    // Training the network using stochastic gradient descent
    void train(vector<MatrixXd> train_data, int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < train_data.size(); ++i) {
                MatrixXd input = train_data[i].block(0, 0, train_data[i].rows() - 1, 1); // Input data
                MatrixXd target = train_data[i].block(train_data[i].rows() - 1, 0, 1, 1); // Target (0 for cat, 1 for dog)
                
                vector<MatrixXd> activations;
                vector<MatrixXd> zs;
                activations.push_back(input);
                
                // Forward propagation
                for (int j = 0; j < num_layers - 1; ++j) {
                    MatrixXd z = (weights[j] * activations[j]).colwise() + biases[j];
                    zs.push_back(z);
                    activations.push_back(sigmoid(z));
                }
                
                // Backpropagation
                MatrixXd delta = (activations[num_layers - 1] - target).array() * sigmoid_derivative(zs[num_layers - 2]).array();
                MatrixXd gradient_w = delta * activations[num_layers - 2].transpose();
                MatrixXd gradient_b = delta;
                weights[num_layers - 2] -= learning_rate * gradient_w;
                biases[num_layers - 2] -= learning_rate * gradient_b;
                
                for (int j = num_layers - 3; j >= 0; --j) {
                    delta = (weights[j+1].transpose() * delta).array() * sigmoid_derivative(zs[j]).array();
                    gradient_w = delta * activations[j].transpose();
                    gradient_b = delta;
                    weights[j] -= learning_rate * gradient_w;
                    biases[j] -= learning_rate * gradient_b;
                }
            }
        }
    }
};

int main() {
    // Example training data (features and labels)
    vector<MatrixXd> train_data;
    train_data.push_back((MatrixXd(3, 1) << 0.1, 0.2, 0).finished()); // Cat (0)
    train_data.push_back((MatrixXd(3, 1) << 0.7, 0.9, 1).finished()); // Dog (1)
    
    // Define network architecture (3 input nodes, 4 hidden nodes, 1 output node)
    vector<int> layers = {3, 4, 1};
    
    // Create neural network
    NeuralNetwork nn(layers);
    
    // Train the network
    nn.train(train_data, 1000, 0.1);
    
    // Test the network
    MatrixXd test_input(3, 1);
    test_input << 0.3, 0.5, 0;
    MatrixXd prediction = nn.feedforward(test_input);
    
    cout << "Prediction for test input: " << prediction << endl;
    
    return 0;
}
