#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdio>
#include<cmath>
using namespace std;

#define e 2.718282

class Matrix //this class is written to operate matrix operations easily in this code.
{
	unsigned row; //keeps row number
	unsigned column; //keeps column number
	double** m; //keeps 2d dynamic array's address
  public:
	Matrix(); 
	Matrix(unsigned, unsigned); 
	~Matrix();
	unsigned get_row(); //getter
	unsigned get_column(); //getter
	void print_matrix();
	void set_matrix(double*[]); //sets matrix's values 
	double** get_head(); //getter
	Matrix operator*(Matrix&); //operator* overloading
	Matrix operator+(Matrix&); //operator+ overloading
	void operator=(Matrix);    //operator= overloading
};

Matrix::Matrix()
{
	row = 0;
	column = 0;
	m = NULL;
}

Matrix::Matrix(unsigned x, unsigned y) //creates matrix with given sizes x and y, sets all values to 0.1
{
	row = x;
	column = y;
	m = new double*[row];
	for(int i = 0; i < row; i++)
	{
		m[i] = new double[column];
		
		for(int j = 0; j < column ; j++)
		{
			m[i][j] = 0.1;
		}
	}
}

Matrix::~Matrix()
{
	for(int i = 0; i < row; i++)
	{
		delete[] m[i];
	}
	delete[] m;
}

void Matrix::print_matrix()
{
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < column; j++)
		{
			cout << m[i][j] << " ";
		}
		cout << endl;
	}
}

unsigned Matrix::get_row()
{
	return row;
}

unsigned Matrix::get_column()
{
	return column;
}

double** Matrix::get_head()
{
	return m;
}

void Matrix::set_matrix(double* array[])
{
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < column; j++)
		{
			m[i][j] = array[i][j];
		}
	}
}

Matrix Matrix::operator+(Matrix& matrix)
{
	double** new_array;
	new_array = new double*[row];
	for(int i = 0; i < row; i++) //creates 2d array with same sizes of matrices
	{
		new_array[i] = new double[column];
		
		for(int j = 0; j < column; j++)
		{
			new_array[i][j] = 0;
		}
	}

	double** head_matrix = matrix.get_head();
	for(int i = 0; i < row; i++) //does sum operation
	{
		for(int j = 0; j < column; j++)
		{
			new_array[i][j] = m[i][j] + head_matrix[i][j];
		}
	}
	Matrix new_matrix(row,column); //creates new matrix to send
	new_matrix.set_matrix(new_array); //sets its values to sum results
	for(int i = 0; i < row; i++) //deletes 2d array to avoid memory leaks
	{
		delete[] new_array[i];
	}
	delete[] new_array;
	return new_matrix;
	
}

Matrix Matrix::operator*(Matrix& matrix)
{
	double** new_array;
	new_array = new double*[row];
	double** matrix_head = matrix.get_head();
	for(int i = 0; i < row; i++) //creates a 2d array to keep production results
	{
		new_array[i] = new double[matrix.get_column()];
		
		for(int j = 0; j < matrix.get_column(); j++)
		{
			new_array[i][j] = 0;
		}
	}
	for(int i = 0; i < row; i++) //does production
	{	
		for(int j = 0; j < matrix.get_column(); j++)
		{
			for(int x = 0; x < column; x++)
			{
				new_array[i][j] += m[i][x] * matrix_head[x][j];
			}
		}
	}
	Matrix new_matrix(row,matrix.get_column()); //creates a matrix to send results
	new_matrix.set_matrix(new_array); //sets its values to results
	for(int i = 0; i < row; i++) //deletes 2d array to avoid memory leaks
	{
		delete[] new_array[i];
	}
	delete[] new_array;
	return new_matrix;
	
}

void Matrix::operator=(Matrix matrix)
{
	for(int i = 0; i < row; i++) //delete old array to avoid memory leak
	{
		delete[] m[i];
	}
	delete[] m;
	row = matrix.get_row();
	column = matrix.get_column();
	double** head_matrix = matrix.get_head();
	m = new double*[row];
	for(int i = 0; i < row; i++) //creates new 2d array and assign its values to values of new matix
	{
		m[i] = new double[column];
		
		for(int j = 0; j < column ; j++)
		{
			m[i][j] = head_matrix[i][j];
		}
	}
}

double sigmoid(double x)
{
	double result = 1.0/(1.0 + pow(e,-x));
	return result;
}

double RelU(double x)
{
	if(x > 0)
	{
		return x;
	}
	else
	{
		return 0;
	}
}

double LeakyRelU(double x)
{
	double a = 0.1 * x;
	if(x > a)
	{
		return x;
	}
	else
	{
		return a;
	}
}
//////////////////////////////////////////////////////////////////////////////// Matrix Class and functions

class Neuron //abstract base class
{
  protected:
	double z;
	double a;
  public: 
	virtual void active() = 0;
	Neuron()
	{}
	void print_neuron();
	double get_value();
	void set_value(double);
};

class SigmoidNeuron:public Neuron
{
  public:
	SigmoidNeuron();
	SigmoidNeuron(double);
	void active();
};

class RelUNeuron:public Neuron
{
  public:
	RelUNeuron();
	RelUNeuron(double);
	void active();
};

class LRelUNeuron:public Neuron
{
  public:
	LRelUNeuron();
	LRelUNeuron(double);
	void active();
};

class Layer
{
	Neuron** neurons; //pointer of pointer array, it keeps head of neuron array
	int count;        //number of neurons in the layer
  public:
	Layer();
	Layer(int, int, int, int[]);
	~Layer();
	void print_layer(); //prints layer
	double** get_value_vector(); //generates X value vector for matrix operation
	void activate_layer();
	void set_layer(double); //setter
	int get_count(); //getter
};

class Network
{
	Layer** layers; //pointer of pointer array, it keeps head of layer array
	int layer_number; //number of layers in the network
  public:
	Network();
	Network(int,int[],int[],int[]);
	~Network();
	void print_network(); //prints network
	void run_network(Matrix*[], Matrix*[]); //does matrix operations in network with given matrices
};

void Neuron::print_neuron()
{
	cout << a << endl;
}

double Neuron::get_value()
{
	return a;
}

void Neuron::set_value(double x)
{
	z = x;
}

SigmoidNeuron::SigmoidNeuron()
{
	a = 0;
	z = 0;
}

SigmoidNeuron::SigmoidNeuron(double value)
{
	a = value;
	z = 0;
}

void SigmoidNeuron::active()
{
	a = sigmoid(z);
}

RelUNeuron::RelUNeuron()
{
	a = 0;
	z = 0;
}

RelUNeuron::RelUNeuron(double value)
{
	a = value;
	z = 0;
}

void RelUNeuron::active()
{
	a = RelU(z);
}

LRelUNeuron::LRelUNeuron()
{
	a = 0;
	z = 0;
}

LRelUNeuron::LRelUNeuron(double value)
{
	a = value;
	z = 0;
}

void LRelUNeuron::active()
{
	a = LeakyRelU(z);
}

Layer::Layer()
{
	neurons = NULL;
	count = 0;
}

Layer::Layer(int neuron_count, int neuron_type, int layer_number, int row[])
{
	Neuron** new_neurons = new Neuron*[neuron_count]; //generates a dynamic array to keep neurons
	for(int i = 0; i < neuron_count; i++)
	{
		if(layer_number == 0) //if layer is first, then create neurons with given values
		{
			if(neuron_type == 0)
			{
				SigmoidNeuron* n_neuron = new SigmoidNeuron(row[i]);
				new_neurons[i] = n_neuron;
			}
			else if(neuron_type == 2)
			{
				RelUNeuron* n_neuron = new RelUNeuron(row[i]);
				new_neurons[i] = n_neuron;
			}
			else if(neuron_type == 1)
			{
				LRelUNeuron* n_neuron = new LRelUNeuron(row[i]);
				new_neurons[i] = n_neuron;
			}
		}
		else //if layer is not first, then creates neurons as default
		{
			if(neuron_type == 0)
			{
				SigmoidNeuron* n_neuron = new SigmoidNeuron();
				new_neurons[i] = n_neuron;
			}
			else if(neuron_type == 2)
			{
				RelUNeuron* n_neuron = new RelUNeuron();
				new_neurons[i] = n_neuron;
			}
			else if(neuron_type == 1)
			{
				LRelUNeuron* n_neuron = new LRelUNeuron();
				new_neurons[i] = n_neuron;
			}
		}
	}
	count = neuron_count;
	neurons = new_neurons;
}

Layer::~Layer()
{
	for(int i = 0; i < count; i++)
	{
		delete neurons[i];
	}
	delete[] neurons;
}

void Layer::print_layer()
{
	for(int i = 0; i < count; i++)
	{
		neurons[i]->print_neuron();
	}
}

double** Layer::get_value_vector() //creates a dynamic array and gets neurons' values
{
	double** array = new double*[count];
	for(int i = 0; i < count; i++)
	{
		array[i] = new double[1];
		for(int j = 0; j < 1 ; j++)
		{
			array[i][j] = neurons[i]->get_value();
		}
	}
	return array;
}

int Layer::get_count()
{
	return count;
}

void Layer::activate_layer() //activates all neurons
{
	for(int i = 0; i < count; i++)
	{
		neurons[i]->active();
	}
}

void Layer::set_layer(double x) //sets all neurons of layer with given value
{
	for(int i = 0; i < count; i++)
	{
		neurons[i]->set_value(x);
	}
}

Network::Network()
{
	layers = NULL;
}

Network::Network(int row1, int row2[], int row3[], int row4[]) //creates all network by using given indexes. These indexes are named with respect to input.txt.
{
	Layer** new_layers = new Layer*[row1]; 
	for(int i = 0; i < row1; i++)
	{
		Layer* n_layer = new Layer(row2[i], row3[i], i, row4);
		new_layers[i] = n_layer;
	}
	layer_number = row1;
	layers = new_layers;
}

Network::~Network()
{
	for(int i = 0; i < layer_number; i++)
	{
		delete layers[i];
	}
	delete[] layers;
}

void Network::print_network()
{
	for(int i = 0; i < layer_number; i++)
	{
		cout << "Layer " << i << ":" << endl;
		layers[i]->print_layer();
	}
}

void Network::run_network(Matrix* w[], Matrix* b[])
{	
	for(int i = 0; i < layer_number; i++)
	{
		if(i == 0) //if layer is 0
		{
			Matrix x_matrix(layers[i]->get_count(),1);
			double** value_vector = layers[i]->get_value_vector(); //(*)creates a dynamic 2D array
			x_matrix.set_matrix(value_vector);					   //creates a matrix by using given 2D array
			Matrix result_matrix(b[i]->get_row(),1);			   //creates a matrix to keep result
			result_matrix = (*w[i] * x_matrix) + *b[i];			   //does matrix operation
			double** head_result = result_matrix.get_head();  	   //gets the matrix's 2d array pointer
			layers[i+1]->set_layer(head_result[0][0]);			   //sets z values of layer
			for(int i = 0; i < x_matrix.get_row(); i++)				//deletes 2d array which is created in (*)
			{
				delete[] value_vector[i];
			}
			delete[] value_vector;
		}
		else if(i != 0 && i != layer_number-1) //unless layer is first or last layer
		{
			layers[i]->activate_layer(); 							//activates all neurons in layer
			Matrix x_matrix(layers[i]->get_count(),1);				
			double** value_vector = layers[i]->get_value_vector();	//(*)creates a dynamic 2D array
			x_matrix.set_matrix(value_vector);						//creates a matrix by using given 2D array
			Matrix result_matrix(b[i]->get_row(),1);				//creates a matrix to keep result
			result_matrix = (*w[i] * x_matrix) + *b[i];				//does matrix operation
			double** head_result = result_matrix.get_head();		//sets z values of layer
			layers[i+1]->set_layer(head_result[0][0]);				//gets the matrix's 2d array pointer
			for(int i = 0; i < x_matrix.get_row(); i++)				//deletes 2d array which is created in (*)
			{
				delete[] value_vector[i];
			}
			delete[] value_vector;
		}
		else //if layer is last layer
		{
			layers[i]->activate_layer(); //activates layer
		}
	}
	
	
}

int main(int argc, char *argv[])
{
	ifstream file(argv[1]);
	string line;
	int row1;
	int row2[64], row3[64], row4[64];
	int row2_i = 0, row3_i = 0, row4_i = 0;
	int row_number = 0;
	char error1[] = "Unidentified activation function!";
	char error2[] = "Input shape does not match!";
	try
	{
		while(getline(file,line)) //this part of code reads input file and seperates it to 4 rows like first row of file is row1, second row of file is row2[] etc..
		{
			int i = 0, j = 0;
			while(line[i] != '\0') //read whole row one by one
			{
				switch(row_number) //controls where it is reading
				{
					case 0:
						row1 = line[0] - '0';
					case 1:
						if(line[i] != '-' && line[i] != ' ')
						{
							row2[j] = line[i] - '0'; //converting string to int
							row2_i++;
							j++;
						}
						else if(line[i] == '-') //if number is negative
						{
							i++;
							row2[j] = (-1)*(line[i] - '0'); //converting string to int
							row2_i++;
							j++;
						}
						break;
					case 2:
						if(line[i] != '-' && line[i] != ' ')
						{
							row3[j] = line[i] - '0'; //converting string to int
							if(row3[j]!= 0 && row3[j]!= 1 && row3[j]!= 2) throw error1;
							row3_i++;
							j++;
						}
						else if(line[i] == '-') //if number is negative
						{
							i++;
							row3[j] = (-1)*(line[i] - '0'); //converting string to int
							if(row3[j]!= 0 && row3[j]!= 1 && row3[j]!= 2) throw error1;
							row3_i++;
							j++;
						}
						break;
					case 3:
						if(line[i] != '-' && line[i] != ' ')
						{
							row4[j] = line[i] - '0'; //converting string to int
							row4_i++;
							j++;
						}
						else if(line[i] == '-') //if number is negative
						{
							i++;
							row4[j] = (-1)*(line[i] - '0'); //converting string to int
							row4_i++;
							j++;
						}
						break;
				}
				i++;
			}
			row_number++;
		}
		if(row2_i-1 != row1 || row3_i != row1 || row4_i != row1) throw error2;
		//Creating proper w matrices and b vectors with default 0.1 values. The number of w matrices and b vectors always equals to row1-1(layer count - 1)
		Matrix *w_matrices[row1-1];
		Matrix *b_matrices[row1-1];
		for(int i = 0; i < row1-1; i++)
		{
			Matrix* w = new Matrix(row2[i+1],row2[i]);
			Matrix* b = new Matrix(row2[i+1],1);
			w_matrices[i] = w;
			b_matrices[i] = b;
		}
		
		Network networkk(row1,row2,row3,row4);
		networkk.run_network(w_matrices,b_matrices);
		networkk.print_network();
		for(int i = 0; i < row1-1; i++) //deletes dynamic operation matrices to avoid memory leaks
		{
			delete w_matrices[i];
			delete b_matrices[i];
		}
	}
	catch(const char* error)
	{
		cout << error << endl;
	}
	
	
	return 0;
}