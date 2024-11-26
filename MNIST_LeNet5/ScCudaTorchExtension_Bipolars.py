import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def calculate_mean_absolute_error(y_true, y_pred):
    mae = torch.mean(torch.abs(y_pred - y_true))
    return mae

def tensor_similarity(tensor1, tensor2, threshold=0.90):
  assert tensor1.shape == tensor2.shape
  similarity = torch.nn.functional.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0)
  return similarity

ScCudaTorch = load(
    name="sc_cuda_torch",
    sources=["ScCudaTorch.cpp", "ScCudaTorch.cu"],
    verbose=True
)

############################ NN DEFINITION ############################
def normalize_to_01(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        conv1_out = self.conv1(x)
        pool_out = self.pool(F.relu(conv1_out))
        #nomalised_pool_out = torch.clamp(pool_out, 0, 1)
        nomalised_pool_out = normalize_to_01(pool_out)
        conv2_inter = self.conv2(nomalised_pool_out)
        conv2_out = self.pool(F.relu(conv2_inter))
        #normalised_conv2_out = torch.clamp(conv2_out, 0, 1)
        normalised_conv2_out = normalize_to_01(conv2_out)
        flattened = normalised_conv2_out.view(-1, 16 * 4 * 4)
        fc1_out = F.relu(self.fc1(flattened))
        #normalised_fc1_out = torch.clamp(fc1_out, 0, 1)
        normalised_fc1_out = normalize_to_01(fc1_out)
        fc2_out = F.relu(self.fc2(normalised_fc1_out))
        #normalised_fc2_out = torch.clamp(fc2_out, 0, 1)
        normalised_fc2_out = normalize_to_01(fc2_out)
        fc3_out = self.fc3(normalised_fc2_out)
        return conv1_out, pool_out, nomalised_pool_out, conv2_inter, conv2_out, normalised_conv2_out, flattened, fc1_out, normalised_fc1_out, fc2_out, normalised_fc2_out, fc3_out

net = GarmentClassifier()
PATH = './mnist_net.pth'
net.load_state_dict(torch.load(PATH))

#print(" ")
#print(" ")

############################ LOAD NN-HORSE INFERED OUTPUTS ############################

original_NN_mnist_tensors = torch.load('original_mnist_tensors.pth')

original_image = original_NN_mnist_tensors['images']
conv1_out = original_NN_mnist_tensors['conv1_out']
pool_out = original_NN_mnist_tensors['pool_out']
normalised_pool_out = original_NN_mnist_tensors['normalised_pool_out']
conv2_inter = original_NN_mnist_tensors['conv2_inter']
conv2_out = original_NN_mnist_tensors['conv2_out']
normalised_conv2_out = original_NN_mnist_tensors['normalised_conv2_out']
flattened = original_NN_mnist_tensors['flattened']
fc1_out = original_NN_mnist_tensors['fc1_out']
normalised_fc1_out = original_NN_mnist_tensors['normalised_fc1_out']
fc2_out = original_NN_mnist_tensors['fc2_out']
normalised_fc2_out = original_NN_mnist_tensors['normalised_fc2_out']
fc3_out = original_NN_mnist_tensors['fc3_out']

randomNumberGenType = ScCudaTorch.RandomGeneratorType
rng_type = randomNumberGenType.MT19937

torch.set_printoptions(sci_mode=False)

############################ CUSTOM SC CONV2D WTH NO. INPUTS/OUTS ############################
#random_matrix_input and random_matrix_kernel should change for every conv2d
def Sc_Conv2_py(custom_weights, custom_biases, input_tensor, h_INPUT, h_KERNEL, input_channels, output_channels, bitstream_Length):
    batch_size, _, height, width = input_tensor.shape
    output_tensor = torch.zeros(output_channels, height - 4, width - 4)  # Output dimensions after 5x5 convolution
    number_Accumulations = 5*5*input_channels

    for out_channel in range(output_channels):
        output = torch.zeros(height - 4, width - 4)  # Temporary output for each output channel
        for in_channel in range(input_channels):
            Sc_custom_weights = custom_weights[out_channel, in_channel, :, :].tolist()
            Sc_custom_Input = input_tensor[0, in_channel, :, :].tolist()

            random_matrix_INPUT = ScCudaTorch.generate_random_matrix(h_INPUT, bitstream_Length, rng_type)
            random_matrix_KERNEL = ScCudaTorch.generate_random_matrix(h_KERNEL, bitstream_Length, rng_type) 

            output += torch.tensor(ScCudaTorch.ScCudaConv2d(Sc_custom_Input, Sc_custom_weights, random_matrix_INPUT, random_matrix_KERNEL))
        
        output = (2 * (output / (number_Accumulations * bitstream_Length)) - 1) * number_Accumulations
        #output = (output / (number_Accumulations * bitstream_Length)) * number_Accumulations                
        output += custom_biases[out_channel]  # Add the bias for the current output channel
        output_tensor[out_channel, :, :] = output  # Assign to the output tensor
    return output_tensor

class_name = 'Sneaker'
print("BIPOLARS COMPACT: ", class_name)

PATH = './correct_images.pth'
correct_images_loaded = torch.load(PATH)
images_list = correct_images_loaded[class_name]

start_time = time.time()

mperrConv1 = 0
sim1 = 0
timeConv1 = 0
mperrConv2 = 0 
timeConv2 = 0 
sim2 = 0
mperrFCL3 = 0
timeFCL3 = 0
sim3 = 0
mperrFCL4 = 0
timeFCL4 = 0
sim4 = 0
mperrFCL5 = 0
timeFCL5 = 0
sim5 = 0

for images in images_list:

    conv1_out, pool_out, nomalised_pool_out, conv2_inter, conv2_out, normalised_conv2_out, flattened, fc1_out, normalised_fc1_out, fc2_out, normalised_fc2_out, fc3_out = net(images) 
    ############################ FIRST SC CONV2D LAYER ####################################
    custom_weights = net.conv1.weight.data
    custom_biases = net.conv1.bias.data

    #print("BIPOLARS")
    bitstream_Length = 512

    input_channels = 1 
    output_channels = 6

    h_INPUT = 28*28 # height of the matrix //NUMBER OF ELEMENTS IN INPUT
    h_KERNEL = 5*5  # height of the matrix //NUMBER OF ELEMENTS IN KERNEL

    start_time = time.time()
    #print("############################ FIRST SC CONV2D LAYER ####################################")
    custom_output = Sc_Conv2_py(custom_weights, custom_biases, original_image, h_INPUT, h_KERNEL, input_channels, output_channels, bitstream_Length)
    #print(" ")
    #print(f"Execution time: {(time.time() - start_time):.2f} seconds")
    #print(" ")
    timeConv1 += (time.time() - start_time)
    #torch.set_printoptions(sci_mode=False)
    #print(" ")
    similarity_score = tensor_similarity(conv1_out.squeeze(0), custom_output)
    #print(f"Similarity score (1rst conv2d wthout pool/relu): {similarity_score}")
    #print(" ")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(conv1_out.squeeze(0), custom_output)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")
    sim1 += similarity_score
    mperrConv1 += error_percentage

    ############################ POOLING LAYER ####################################
    #print("############################ 1ST POOL+RELU OUTPUT SC CONV2D TENSOR ####################################")
    pool = nn.MaxPool2d(2, 2)
    pool2d_tensor = pool(F.relu(custom_output.unsqueeze(0)))
    #print(" ")
    #print(" ")

    ############################ NORMALIZING POOLING LAYER #################################### 
    pool2d_tensor_norm = normalize_to_01(pool2d_tensor)
    similarity_score = tensor_similarity(normalised_pool_out, pool2d_tensor_norm)
    #print(f"Similarity score (1rst conv2d wth Pool & Norm): {similarity_score}")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(normalised_pool_out, pool2d_tensor_norm)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")

    ############################ SECOND SC CONV2D LAYER ####################################
    custom_weights = net.conv2.weight.data
    custom_biases = net.conv2.bias.data

    input_channels = 6 
    output_channels = 16

    #bitstream_Length = 10000

    h_INPUT = 12*12 # height of the matrix //NUMBER OF ELEMENTS IN INPUT
    h_KERNEL = 5*5  # height of the matrix //NUMBER OF ELEMENTS IN KERNEL

    start_time = time.time()
    #print("############################ SECOND SC CONV2D LAYER ####################################")
    secnd_conv2d = Sc_Conv2_py(custom_weights, custom_biases, pool2d_tensor_norm, h_INPUT, h_KERNEL, input_channels, output_channels, bitstream_Length)
    #print(" ")
    #print(f"Execution time: {(time.time() - start_time):.2f} seconds")
    #print(" ")
    timeConv2 += (time.time() - start_time)
    torch.set_printoptions(sci_mode=False)

    ############################ COMPARE SECOND SC CONV2D LAYER OUTPUT ####################################
    #print(" ")
    similarity_score = tensor_similarity(conv2_inter.squeeze(0), secnd_conv2d)
    #print(f"Similarity score (2nd conv2d wthout pool/relu): {similarity_score}")
    #print(" ")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(conv2_inter.squeeze(0), secnd_conv2d)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")
    sim2 += similarity_score
    mperrConv2 += error_percentage
    ############ FLATTEN 2ND SC CONV2D TENSOR ################
    #print("############################ 2ND POOL+RELU OUTPUT SC CONV2D TENSOR ############################")
    pool = nn.MaxPool2d(2, 2)
    sc_conv2_out = pool(F.relu(secnd_conv2d))#.unsqueeze(0)

    norm_sc_conv2_out = normalize_to_01(sc_conv2_out)
    similarity_score = tensor_similarity(normalised_conv2_out.squeeze(0), norm_sc_conv2_out)
    #print(" ")
    #print(f"Similarity score (2ND conv2d wth Pool & Norm): {similarity_score}")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(normalised_conv2_out.squeeze(0), norm_sc_conv2_out)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")

    #sc_conv2_out = pool(F.relu(secnd_conv2d))
    sc_flattened = torch.flatten(norm_sc_conv2_out, 0)
    ############################ NORMALIZING FLATTENING LAYER ####################################
    sc_flattened_list = sc_flattened.squeeze(0).squeeze(0).tolist()

    ########### 1RST LINEAR LAYER ::SC:: ################
    #print(" ")
    #print(" ")
    #print("############### 1ST SC LINEAR LAYER ################")

    #bitstream_Length = 40000

    new_weightsfc = net.fc1.weight.data.t().tolist()
    #new_weightsfc = torch.flatten(new_weightsfc, 0).tolist() #np.array(new_weightsfc).flatten()
    #new_weightsfc = new_weightsfc.tolist()

    new_biasfc = net.fc1.bias.data
    new_biasfc = new_biasfc.tolist()

    noInputs = 16*4*4
    noOutputs = 120

    new_INPUTfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noInputs, bitstream_Length, rng_type)
    new_weightsfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noInputs*noOutputs, bitstream_Length, rng_type)
    new_biasfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noOutputs, bitstream_Length, rng_type)

    no_accumulations = noInputs + 1

    start_time = time.time()
    #print(" ")
    output_sc_FcLayer1 = torch.tensor(ScCudaTorch.ScCudaFcLayer(sc_flattened_list, new_weightsfc, new_biasfc, new_INPUTfc_rndomMatrix, new_weightsfc_rndomMatrix, new_biasfc_rndomMatrix, noOutputs))
    #print(f"Execution time: {(time.time() - start_time):.2f} seconds")
    output_sc_FcLayer1 = (2*(output_sc_FcLayer1/(no_accumulations*bitstream_Length))-1)*no_accumulations
    timeFCL3 += (time.time() - start_time)
    ############################ COMPARE 1ST SC FC LAYER ####################################
    first_scfc_layer = F.relu(output_sc_FcLayer1)
    #print(" ")
    similarity_score = tensor_similarity(fc1_out.squeeze(0), first_scfc_layer)
    #print(f"Similarity score (1rst sc fc layer): {similarity_score}")
    #print(" ")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(fc1_out.squeeze(0), first_scfc_layer)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")
    sim3 += similarity_score
    mperrFCL3 += error_percentage
    ############################ NORMALIZING output_sc_FcLayer1 LAYER ####################################
    norm_sc_FcLayer1_tensor = normalize_to_01(first_scfc_layer)

    similarity_score = tensor_similarity(normalised_fc1_out.squeeze(0), norm_sc_FcLayer1_tensor)
    #print(f"Similarity score (1rst sc fc layer) wth Norm: {similarity_score}")
    #print(" ")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(normalised_fc1_out.squeeze(0), norm_sc_FcLayer1_tensor)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")
    output_sc_FcLayer1_tensor_norm = norm_sc_FcLayer1_tensor.squeeze(0).squeeze(0).tolist()

    ########### 2ND LINEAR LAYER ::SC:: ################
    #print(" ")
    #print("############### 2ND SC LINEAR LAYER ################")

    new_weightsfc = net.fc2.weight.data.t().tolist()
    #new_weightsfc = torch.flatten(new_weightsfc, 0).tolist() #np.array(new_weightsfc).flatten()
    #new_weightsfc = new_weightsfc.tolist()

    new_biasfc = net.fc2.bias.data
    new_biasfc = new_biasfc.tolist()

    noInputs = 120
    noOutputs = 84

    new_INPUTfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noInputs, bitstream_Length, rng_type)
    new_weightsfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noInputs*noOutputs, bitstream_Length, rng_type)
    new_biasfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noOutputs, bitstream_Length, rng_type)

    no_accumulations = noInputs + 1

    start_time = time.time()
    #print(" ")
    output_sc_FcLayer2 = torch.tensor(ScCudaTorch.ScCudaFcLayer(output_sc_FcLayer1_tensor_norm, new_weightsfc, new_biasfc, new_INPUTfc_rndomMatrix, new_weightsfc_rndomMatrix, new_biasfc_rndomMatrix, noOutputs))
    #print(f"Execution time: {(time.time() - start_time):.2f} seconds")
    output_sc_FcLayer2 = (2*(output_sc_FcLayer2/(no_accumulations*bitstream_Length))-1)*no_accumulations
    timeFCL4 += (time.time() - start_time)
    ############################ COMPARE 2nd SC FC LAYER ####################################
    second_scfc_layer = F.relu(output_sc_FcLayer2)
    #print(" ")
    similarity_score = tensor_similarity(fc2_out.squeeze(0), second_scfc_layer)
    #print(f"Similarity score (2nd sc fc layer): {similarity_score}")
    #print(" ")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(fc2_out.squeeze(0), second_scfc_layer)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    sim4 += similarity_score
    mperrFCL4 += error_percentage
    ############################ NORMALIZING output_sc_FcLayer2 LAYER ####################################
    norm_sc_FcLayer2_tensor = normalize_to_01(second_scfc_layer)
    #print(" ")
    similarity_score = tensor_similarity(normalised_fc2_out.squeeze(0), norm_sc_FcLayer2_tensor)
    #print(f"Similarity score (2nd sc fc layer wth Norm) : {similarity_score}")
    #print(" ")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(normalised_fc2_out.squeeze(0), norm_sc_FcLayer2_tensor)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")
    output_sc_FcLayer2_tensor_norm = norm_sc_FcLayer2_tensor.squeeze(0).squeeze(0).tolist()
    #print(" ")

    ################ 3rd LINEAR LAYER ::SC:: ################
    #print(" ")
    #print(" ")
    #print("################ 3rd SC LINEAR LAYER ################")
    #new_INPUTfc = outputs_activation2.squeeze(0).squeeze(0).tolist()

    new_weightsfc = net.fc3.weight.data.t().tolist()

    new_biasfc = net.fc3.bias.data
    new_biasfc = new_biasfc.tolist()

    noInputs = 84
    noOutputs = 10

    new_INPUTfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noInputs, bitstream_Length, rng_type)
    new_weightsfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noInputs*noOutputs, bitstream_Length, rng_type)
    new_biasfc_rndomMatrix = ScCudaTorch.generate_random_matrix(noOutputs, bitstream_Length, rng_type)

    no_accumulations = noInputs + 1

    start_time = time.time()
    #print(" ")
    output_sc_FcLayer3 = torch.tensor(ScCudaTorch.ScCudaFcLayer(output_sc_FcLayer2_tensor_norm, new_weightsfc, new_biasfc, new_INPUTfc_rndomMatrix, new_weightsfc_rndomMatrix, new_biasfc_rndomMatrix, noOutputs))
    #print(f"Execution time: {(time.time() - start_time):.2f} seconds")
    output_sc_FcLayer3 = (2*(output_sc_FcLayer3/(no_accumulations*bitstream_Length))-1)*no_accumulations
    #output_sc_FcLayer = F.relu(output_sc_FcLayer)
    #print(" ")
    #print(" ")
    timeFCL5 += (time.time() - start_time)
    ############################ COMPARE 3rd SC FC LAYER ####################################
    similarity_score = tensor_similarity(fc3_out.squeeze(0), output_sc_FcLayer3)
    #print(f"Similarity score (3rd sc fc layer): {similarity_score}")
    #print(" ")
    #print(" ")
    error_percentage = calculate_mean_absolute_error(fc3_out.squeeze(0), output_sc_FcLayer3)
    #print(f"mean absolute error: {error_percentage:.2f}")
    #print(" ")
    #print(" ")
    sim5 += similarity_score
    mperrFCL5 += error_percentage
    #print(" ")
    #print("SC_FcLayer3:")
    #print(output_sc_FcLayer3)
    #print(" ")
    #print("ORIGINAL_FcLayer3:")
    #print(fc3_out)
    #print(" ")
    #print(" ")

    # Define labels for each of the 10 values
    labels = {
        0: 'T-shirt/top',
        1: 'Trouser ',
        2: 'Pullover ',
        3: 'Dress ',
        4: 'Coat ',
        5: 'Sandal ',
        6: 'Shirt ',
        7: 'Sneaker ',
        8: 'Bag ',
        9: 'Ankle Boot '
    }

    # Find the index of the maximum value in the tensor
    max_index = torch.argmax(output_sc_FcLayer3).item()
    # Retrieve the corresponding label and the maximum value
    max_label = labels[max_index]
    max_value = output_sc_FcLayer3[max_index].item()
    # Print the label with the highest value
    #print(f"Final Output: {max_label}: {max_value}")

    #print(" ")
    #print(" ")

print(" ")
print(" ")
    
mperrConv1 = mperrConv1 /100
print("mperrConv1: ", mperrConv1)
timeConv1 = timeConv1 /100
print("timeConv1: ", timeConv1)
mperrConv2 = mperrConv2/100 
print("mperrConv2: ", mperrConv2)
timeConv2 = timeConv2/100
print("timeConv2: ", timeConv2)
mperrFCL3 = mperrFCL3/100
print("mperrFCL3: ", mperrFCL3)
timeFCL3 = timeFCL3/100
print("timeFCL3: ", timeFCL3)
mperrFCL4 = mperrFCL4/100
print("mperrFCL4: ", mperrFCL4)
timeFCL4 = timeFCL4/100
print("timeFCL4: ", timeFCL4)
mperrFCL5 = mperrFCL5/100
print("mperrFCL5: ", mperrFCL5)
timeFCL5 = timeFCL5/100
print("timeFCL5: ", timeFCL5)

print("sim1: ", sim1/100)
print("sim2: ", sim2/100)
print("sim3: ", sim3/100)
print("sim4: ", sim4/100)
print("sim5: ", sim5/100)


'''
############################ SECOND CONV2D LAYER :::REAL::: ####################################
def process_input_output_channels_separately(custom_weights, custom_biases, input_tensor):
    batch_size, _, height, width = input_tensor.shape
    output_tensor = torch.zeros(batch_size, 16, height - 4, width - 4)  # Output dimensions after 5x5 convolution
    
    for out_channel in range(16):
        output = torch.zeros(batch_size, 1, height - 4, width - 4)  # Temporary output for each output channel
        for in_channel in range(4):
            conv_layer = nn.Conv2d(1, 1, 5)
            with torch.no_grad():
                # Each conv_layer gets a specific 5x5 kernel filter from custom_weights
                conv_layer.weight = nn.Parameter(custom_weights[out_channel, in_channel, :, :].unsqueeze(0).unsqueeze(0))
                conv_layer.bias = nn.Parameter(torch.tensor([0.0]))  # Bias is set to zero temporarily
            ##print(custom_weights[out_channel, in_channel, :, :].unsqueeze(0).unsqueeze(0))
            input_single_channel = input_tensor[:, in_channel, :, :].unsqueeze(1)  # Extract one input channel
            ##print(input_single_channel)
            output += conv_layer(input_single_channel)  # Sum the convolutions of each input channel
            ##print(conv_layer(input_single_channel))
        output += custom_biases[out_channel]  # Add the bias for the current output channel
        output_tensor[:, out_channel, :, :] = output.squeeze(1)  # Assign to the output tensor
    
    return output_tensor

weights_ = net.conv2.weight.data
bias_ = net.conv2.bias.data
custom_output__2 = process_input_output_channels_separately(weights_, bias_, pool2d_tensor)

#print("############################ SECOND CONV2D LAYER :::REAL::: ####################################")
error_percentage = calculate_mean_absolute_error(conv2_inter, custom_output__2)
#print(f"mean absolute error: {error_percentage:.2f}")
#print(" ")
'''

'''
########### 1st LINEAR LAYER ::ORIGINAL:: ################
# Define the fully connected layer with given weights and bias
class CustomLinearLayer(nn.Module):
    def __init__(self, weights, bias):
        super(CustomLinearLayer, self).__init__()
        # Transpose the weights to be (output_size, input_size)
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))#.t())
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

inputs_tensor = sc_flattened
weights = net.fc1.weight.data#.t()
bias = net.fc1.bias.data

# Create the custom linear layer
layer = CustomLinearLayer(weights, bias)
# Forward pass without activation function
outputs_no_activation = layer(inputs_tensor)
outputs_activation = F.relu(outputs_no_activation)
##print("Outputs without activation:")
##print(outputs_no_activation2)
#print(" ")
#print("########### 1st LINEAR LAYER ################")
error_percentage = calculate_mean_absolute_error(fc1_out.squeeze(0), outputs_activation.squeeze(0).squeeze(0))
#print(f"mean absolute error: {error_percentage:.2f}")
#print(" ")
#print(" ")

########### 2nd LINEAR LAYER ::ORIGINAL:: ################

#inputs_tensor = output_sc_FcLayer.unsqueeze(0)
inputs_tensor = outputs_activation.unsqueeze(0)
weights = net.fc2.weight.data#.t()
bias = net.fc2.bias.data

# Create the custom linear layer
layer = CustomLinearLayer(weights, bias)
# Forward pass without activation function
outputs_no_activation2 = layer(inputs_tensor)
outputs_activation2 = F.relu(outputs_no_activation2)
##print("Outputs without activation:")
##print(outputs_no_activation2)
#print("########### 2nd LINEAR LAYER ::ORIGINAL:: ################")
error_percentage = calculate_mean_absolute_error(fc2_out.squeeze(0), outputs_activation2.squeeze(0).squeeze(0))
#print(f"mean absolute error: {error_percentage:.2f}")
#print(" ")
#print(" ")
'''
