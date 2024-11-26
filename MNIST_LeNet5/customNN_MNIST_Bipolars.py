import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ScCudaTorch = load(
    name="sc_cuda_torch",
    sources=["ScCudaTorch.cpp", "ScCudaTorch.cu"],
    verbose=True
)

randomNumberGenType = ScCudaTorch.RandomGeneratorType
rng_type = randomNumberGenType.MT19937

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

# Define your Net class
class ScNet(nn.Module):
    def __init__(self, custom_weights1, custom_biases1, custom_weights2, custom_biases2, 
                 new_weightsfc3, new_biasfc3, new_INPUTfc_rndomMatrix3, new_weightsfc_rndomMatrix3, new_biasfc_rndomMatrix3, noOutputs3,
                 new_weightsfc4, new_biasfc4, new_INPUTfc_rndomMatrix4, new_weightsfc_rndomMatrix4, new_biasfc_rndomMatrix4, noOutputs4,
                 new_weightsfc5, new_biasfc5, new_INPUTfc_rndomMatrix5, new_weightsfc_rndomMatrix5, new_biasfc_rndomMatrix5, noOutputs5,
                 bitstream_Length):
        super().__init__()
        # Initialize parameters
        self.custom_weights1 = custom_weights1
        self.custom_biases1 = custom_biases1
        self.custom_weights2 = custom_weights2
        self.custom_biases2 = custom_biases2
        self.new_weightsfc3 = new_weightsfc3
        self.new_biasfc3 = new_biasfc3
        self.new_INPUTfc_rndomMatrix3 = new_INPUTfc_rndomMatrix3
        self.new_weightsfc_rndomMatrix3 = new_weightsfc_rndomMatrix3
        self.new_biasfc_rndomMatrix3 = new_biasfc_rndomMatrix3
        self.noOutputs3 = noOutputs3
        self.new_weightsfc4 = new_weightsfc4
        self.new_biasfc4 = new_biasfc4
        self.new_INPUTfc_rndomMatrix4 = new_INPUTfc_rndomMatrix4
        self.new_weightsfc_rndomMatrix4 = new_weightsfc_rndomMatrix4
        self.new_biasfc_rndomMatrix4 = new_biasfc_rndomMatrix4
        self.noOutputs4 = noOutputs4
        self.new_weightsfc5 = new_weightsfc5
        self.new_biasfc5 = new_biasfc5
        self.new_INPUTfc_rndomMatrix5 = new_INPUTfc_rndomMatrix5
        self.new_weightsfc_rndomMatrix5 = new_weightsfc_rndomMatrix5
        self.new_biasfc_rndomMatrix5 = new_biasfc_rndomMatrix5
        self.noOutputs5 = noOutputs5
        self.bitstream_Length = bitstream_Length
        self.pool = nn.MaxPool2d(2, 2)
    
    def normalize_to_01(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    def forward(self, original_image):
        ################::SC FORWARD PROPAGATION:: ################
        custom_output = Sc_Conv2_py(self.custom_weights1, self.custom_biases1, original_image, h_INPUT1, h_KERNEL1, input_channels1, output_channels1, self.bitstream_Length)
        pool2d_tensor = self.pool(F.relu(custom_output.unsqueeze(0)))
        pool2d_tensor_norm = self.normalize_to_01(pool2d_tensor)
        secnd_conv2d = Sc_Conv2_py(self.custom_weights2, self.custom_biases2, pool2d_tensor_norm, h_INPUT2, h_KERNEL2, input_channels2, output_channels2, self.bitstream_Length)
        sc_conv2_out = self.pool(F.relu(secnd_conv2d))
        norm_sc_conv2_out = self.normalize_to_01(sc_conv2_out)
        sc_flattened = torch.flatten(norm_sc_conv2_out, 0)
        sc_flattened_list = sc_flattened.squeeze(0).squeeze(0).tolist()

        output_sc_FcLayer1 = torch.tensor(ScCudaTorch.ScCudaFcLayer(sc_flattened_list, self.new_weightsfc3, self.new_biasfc3, self.new_INPUTfc_rndomMatrix3, self.new_weightsfc_rndomMatrix3, self.new_biasfc_rndomMatrix3, self.noOutputs3))
        #output_sc_FcLayer1 = (output_sc_FcLayer1/(no_accumulations3*self.bitstream_Length))*no_accumulations3
        output_sc_FcLayer1 = (2*(output_sc_FcLayer1/(no_accumulations3*bitstream_Length))-1)*no_accumulations3
        first_scfc_layer = F.relu(output_sc_FcLayer1)
        norm_sc_FcLayer1_tensor = self.normalize_to_01(first_scfc_layer)
        output_sc_FcLayer1_tensor_norm = norm_sc_FcLayer1_tensor.squeeze(0).squeeze(0).tolist()

        output_sc_FcLayer2 = torch.tensor(ScCudaTorch.ScCudaFcLayer(output_sc_FcLayer1_tensor_norm, self.new_weightsfc4, self.new_biasfc4, self.new_INPUTfc_rndomMatrix4, self.new_weightsfc_rndomMatrix4, self.new_biasfc_rndomMatrix4, self.noOutputs4))
        #output_sc_FcLayer2 = (output_sc_FcLayer2/(no_accumulations4*self.bitstream_Length))*no_accumulations4
        output_sc_FcLayer2 = (2*(output_sc_FcLayer2/(no_accumulations4*bitstream_Length))-1)*no_accumulations4
        second_scfc_layer = F.relu(output_sc_FcLayer2)
        norm_sc_FcLayer2_tensor = self.normalize_to_01(second_scfc_layer)
        output_sc_FcLayer2_tensor_norm = norm_sc_FcLayer2_tensor.squeeze(0).squeeze(0).tolist()

        output_sc_FcLayer3 = torch.tensor(ScCudaTorch.ScCudaFcLayer(output_sc_FcLayer2_tensor_norm, self.new_weightsfc5, self.new_biasfc5, self.new_INPUTfc_rndomMatrix5, self.new_weightsfc_rndomMatrix5, self.new_biasfc_rndomMatrix5, self.noOutputs5))
        #output_sc_FcLayer3 = (output_sc_FcLayer3/(no_accumulations5*self.bitstream_Length))*no_accumulations5
        output_sc_FcLayer3 = (2*(output_sc_FcLayer3/(no_accumulations5*bitstream_Length))-1)*no_accumulations5
        
        return output_sc_FcLayer3.unsqueeze(0)
        ################::END SC FORWARD PROPAGATION:: ################

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

############################ PARAMETERS ####################################
bitstream_Length = 1280
print("BIPOLARS COMPACT", bitstream_Length)
############################ FIRST SC CONV2D LAYER #########################
custom_weights1 = net.conv1.weight.data
custom_biases1 = net.conv1.bias.data
input_channels1 = 1 
output_channels1 = 6
h_INPUT1 = 28*28 
h_KERNEL1 = 5*5 
############################ SECOND SC CONV2D LAYER ########################
custom_weights2 = net.conv2.weight.data
custom_biases2 = net.conv2.bias.data
input_channels2 = 6 
output_channels2 = 16
h_INPUT2 = 12*12
h_KERNEL2 = 5*5  
############### 1ST SC LINEAR LAYER ################
new_weightsfc3 = net.fc1.weight.data.t().tolist()
new_biasfc3 = net.fc1.bias.data
new_biasfc3 = new_biasfc3.tolist()
noInputs3 = 16*4*4
noOutputs3 = 120
no_accumulations3 = noInputs3 + 1
new_INPUTfc_rndomMatrix3 = ScCudaTorch.generate_random_matrix(noInputs3, bitstream_Length, rng_type)
new_weightsfc_rndomMatrix3 = ScCudaTorch.generate_random_matrix(noInputs3*noOutputs3, bitstream_Length, rng_type)
new_biasfc_rndomMatrix3 = ScCudaTorch.generate_random_matrix(noOutputs3, bitstream_Length, rng_type)
############### 2ND SC LINEAR LAYER ################
new_weightsfc4 = net.fc2.weight.data.t().tolist()
new_biasfc4 = net.fc2.bias.data
new_biasfc4 = new_biasfc4.tolist()
noInputs4 = 120
noOutputs4 = 84
no_accumulations4 = noInputs4 + 1
new_INPUTfc_rndomMatrix4 = ScCudaTorch.generate_random_matrix(noInputs4, bitstream_Length, rng_type)
new_weightsfc_rndomMatrix4 = ScCudaTorch.generate_random_matrix(noInputs4*noOutputs4, bitstream_Length, rng_type)
new_biasfc_rndomMatrix4 = ScCudaTorch.generate_random_matrix(noOutputs4, bitstream_Length, rng_type)
################ 3rd LINEAR LAYER ::SC:: ################
new_weightsfc5 = net.fc3.weight.data.t().tolist()
new_biasfc5 = net.fc3.bias.data
new_biasfc5 = new_biasfc5.tolist()
noInputs5 = 84
noOutputs5 = 10
no_accumulations5 = noInputs5 + 1
new_INPUTfc_rndomMatrix5 = ScCudaTorch.generate_random_matrix(noInputs5, bitstream_Length, rng_type)
new_weightsfc_rndomMatrix5 = ScCudaTorch.generate_random_matrix(noInputs5*noOutputs5, bitstream_Length, rng_type)
new_biasfc_rndomMatrix5 = ScCudaTorch.generate_random_matrix(noOutputs5, bitstream_Length, rng_type)

scNet = ScNet(custom_weights1, custom_biases1, custom_weights2, custom_biases2, 
          new_weightsfc3, new_biasfc3, new_INPUTfc_rndomMatrix3, new_weightsfc_rndomMatrix3, new_biasfc_rndomMatrix3, noOutputs3,
          new_weightsfc4, new_biasfc4, new_INPUTfc_rndomMatrix4, new_weightsfc_rndomMatrix4, new_biasfc_rndomMatrix4, noOutputs4,
          new_weightsfc5, new_biasfc5, new_INPUTfc_rndomMatrix5, new_weightsfc_rndomMatrix5, new_biasfc_rndomMatrix5, noOutputs5,
          bitstream_Length)

import torchvision
import torchvision.transforms as transforms

from datetime import datetime

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

'''''
def get_images_with_label(dataloader, label, num_images):
    images_list = []
    labels_list = []
    for images, labels in dataloader:
        for i in range(len(labels)):
            if labels[i] == label:
                images_list.append(images[i].unsqueeze(0))
                labels_list.append(labels[i])
                if len(images_list) >= num_images:
                    return torch.cat(images_list), torch.tensor(labels_list)
    return torch.cat(images_list), torch.tensor(labels_list)


num_images = 1
label_index = classes.index('bird')
images, labels = get_images_with_label(testloader, label_index, num_images)

print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(1)))

outputs = scNet(images)
conv1_out, pool_out,nomalised_pool_out, conv2_out, conv2_inter,normalised_conv2_out, flattened, fc1_out,normalised_fc1_out, fc2_out,normalised_fc2_out, fc3_out = net(images)
outputs_original = fc3_out

print(outputs)
print(outputs.shape)
print(outputs_original)
print(outputs_original.shape)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(1)))

_, predicted_or = torch.max(outputs_original, 1)

print('Predicted Original: ', ' '.join(f'{classes[predicted_or[j]]:5s}'
                              for j in range(1)))
'''''

# prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# correct_images = {classname: [] for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         conv1_out, pool_out, nomalised_pool_out, conv2_inter, conv2_out, normalised_conv2_out, flattened, fc1_out, normalised_fc1_out, fc2_out, normalised_fc2_out, fc3_out = net(images)
#         _, predictions = torch.max(fc3_out, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
                
#                 # Only store the first 100 correctly predicted images per class
#                 if len(correct_images[classes[label]]) < 100:
#                     correct_images[classes[label]].append(images.cpu())
                    
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
# PATH = './correct_images.pth'
# torch.save(correct_images, PATH)


class_name = 'Sneaker'
print("BIPOLARS COMPACT: ", class_name, bitstream_Length)

correct_pred_class = 0
total_pred_class = 0

PATH = './correct_images.pth'
correct_images_loaded = torch.load(PATH)
images_list = correct_images_loaded[class_name]

start_time = time.time()
with torch.no_grad():
    for images in images_list:
        
        images = images.squeeze(0)
        
        outputs = scNet(images.unsqueeze(0))
        _, predictions = torch.max(outputs, 1)
        
        true_label = classes.index(class_name)
        
        if predictions.item() == true_label:
            correct_pred_class += 1
        
        total_pred_class += 1

if total_pred_class > 0:
    accuracy_class = 100 * float(correct_pred_class) / total_pred_class
    print(f'Accuracy for class: {class_name} with scnet is {accuracy_class:.1f} %')
else:
    print(f'No predictions for class: {class_name}')
print(f"Execution time: {(time.time() - start_time):.2f} seconds")
