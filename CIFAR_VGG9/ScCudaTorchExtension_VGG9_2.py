import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ScCudaTorch = load(
    name="sc_cuda_torch",
    sources=["ScCudaTorch_2.cpp", "ScCudaTorch_UniPinConst_2.cu"],
    verbose=True
)

randomNumberGenType = ScCudaTorch.RandomGeneratorType
rng_type = randomNumberGenType.MT19937

# def mean_absolute_percentage_error(y_pred, y_true):
#   ape = torch.abs((y_pred - y_true) / y_true) * 100
#   mape = torch.mean(ape)
#   return mape

def mean_absolute_percentage_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    epsilon = 1e-8
    y_true = torch.clamp(y_true, min=epsilon)
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
    return mape

############################ CUSTOM SC CONV2D WTH NO. INPUTS/OUTS ############################
#random_matrix_input and random_matrix_kernel should change for every conv2d
def Sc_Conv2_py(custom_weights, custom_biases, input_tensor, h_INPUT, h_KERNEL, input_channels, output_channels, bitstream_Length):
    batch_size, _, height, width = input_tensor.shape
    batch_size, _, heightK, widthK = custom_weights.shape
    output_tensor = torch.zeros(output_channels, height - 2, width - 2)  # Output dimensions after 3x3 convolution
    number_Accumulations = 3*3*input_channels

    for out_channel in range(output_channels):
        output = torch.zeros(height - 2, width - 2)  # Temporary output for each output channel
        for in_channel in range(input_channels):

            Sc_custom_weights = torch.flatten(custom_weights[out_channel, in_channel, :, :], 0)
            Sc_custom_weights = Sc_custom_weights.tolist()
            #print(Sc_custom_weights)
            #Sc_custom_weights = custom_weights[out_channel, in_channel, :, :].tolist()
            
            Sc_custom_Input = torch.flatten(input_tensor[0, in_channel, :, :], 0)
            Sc_custom_Input = Sc_custom_Input.tolist()
            #print(Sc_custom_Input)
            #Sc_custom_Input = input_tensor[0, in_channel, :, :].tolist()

            random_matrix_INPUT = ScCudaTorch.generate_random_matrix(h_INPUT, bitstream_Length, rng_type)
            random_matrix_KERNEL = ScCudaTorch.generate_random_matrix(h_KERNEL, bitstream_Length, rng_type) 

            output += torch.tensor(ScCudaTorch.ScCudaConv2d(Sc_custom_Input, Sc_custom_weights, random_matrix_INPUT, random_matrix_KERNEL, bitstream_Length, height, width, heightK)) 

        #output = (2 * (output / (number_Accumulations * bitstream_Length)) - 1) * number_Accumulations
        output = (output / (number_Accumulations * bitstream_Length)) * number_Accumulations                
        output += custom_biases[out_channel]  # Add the bias for the current output channel
        output_tensor[out_channel, :, :] = output  # Assign to the output tensor
    return output_tensor

class ScNet(nn.Module):
    def __init__(self, custom_weights1, custom_biases1, custom_weights2, custom_biases2,
                 custom_weights3, custom_biases3, custom_weights4, custom_biases4,
                 custom_weights5, custom_biases5, custom_weights6, custom_biases6, 
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
        self.custom_weights3 = custom_weights3
        self.custom_biases3 = custom_biases3
        self.custom_weights4 = custom_weights4
        self.custom_biases4 = custom_biases4
        self.custom_weights5 = custom_weights5
        self.custom_biases5 = custom_biases5
        self.custom_weights6 = custom_weights6
        self.custom_biases6 = custom_biases6
        # FCLayer parameters
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
        self.pad1 = nn.ZeroPad2d(1)
    
    def forward(self, original_image):
        # Block 1: External padding before each conv layer + ReLU + Clamp [0, 1]
        propagated1 = self.pad1(original_image)
        propagated1 = Sc_Conv2_py(self.custom_weights1, self.custom_biases1, propagated1, h_INPUT1, h_KERNEL1, input_channels1, output_channels1, self.bitstream_Length)
        propagated1 = F.relu(propagated1.unsqueeze(0))
        propagated1 = torch.clamp(propagated1, 0, 1)
        propagated2 = self.pad1(propagated1)
        propagated2 = Sc_Conv2_py(self.custom_weights2, self.custom_biases2, propagated2, h_INPUT2, h_KERNEL2, input_channels2, output_channels2, self.bitstream_Length)
        propagated2 = F.relu(propagated2.unsqueeze(0))
        propagated2 = torch.clamp(propagated2, 0, 1)
        propagated2 = self.pool(propagated2)
        # Block 2: External padding before each conv layer + ReLU + Clamp [0, 1]
        propagated3 = self.pad1(propagated2)
        propagated3 = Sc_Conv2_py(self.custom_weights3, self.custom_biases3, propagated3, h_INPUT3, h_KERNEL3, input_channels3, output_channels3, self.bitstream_Length)
        propagated3 = F.relu(propagated3.unsqueeze(0))
        propagated3 = torch.clamp(propagated3, 0, 1)
        propagated4 = self.pad1(propagated3)
        propagated4 = Sc_Conv2_py(self.custom_weights4, self.custom_biases4, propagated4, h_INPUT4, h_KERNEL4, input_channels4, output_channels4, self.bitstream_Length)
        propagated4 = F.relu(propagated4.unsqueeze(0))
        propagated4 = torch.clamp(propagated4, 0, 1)
        propagated4 = self.pool(propagated4)
        # Block 3: External padding before each conv layer + ReLU + Clamp [0, 1]
        propagated5 = self.pad1(propagated4)
        propagated5 = Sc_Conv2_py(self.custom_weights5, self.custom_biases5, propagated5, h_INPUT5, h_KERNEL5, input_channels5, output_channels5, self.bitstream_Length)
        propagated5 = F.relu(propagated5.unsqueeze(0))
        propagated5 = torch.clamp(propagated5, 0, 1)
        propagated6 = self.pad1(propagated5)
        propagated6 = Sc_Conv2_py(self.custom_weights6, self.custom_biases6, propagated6, h_INPUT6, h_KERNEL6, input_channels6, output_channels6, self.bitstream_Length)
        propagated6 = F.relu(propagated6.unsqueeze(0))
        propagated6 = torch.clamp(propagated6, 0, 1)
        propagated6 = self.pool(propagated6)
        # Flatten the feature maps for fully connected layers
        sc_flattened = torch.flatten(propagated6, 0)
        sc_flattened_list = sc_flattened.squeeze(0).squeeze(0).tolist()
        # Fully connected layers + ReLU + Clamp [0, 1]
        output_sc_FcLayer1 = torch.tensor(ScCudaTorch.ScCudaFcLayer(sc_flattened_list, self.new_weightsfc3, self.new_biasfc3, self.new_INPUTfc_rndomMatrix3, self.new_weightsfc_rndomMatrix3, self.new_biasfc_rndomMatrix3, self.noOutputs3, self.bitstream_Length))
        output_sc_FcLayer1 = (output_sc_FcLayer1/(no_accumulations3*self.bitstream_Length))*no_accumulations3
        first_scfc_layer = F.relu(output_sc_FcLayer1)
        norm_sc_FcLayer1_tensor = torch.clamp(first_scfc_layer, 0, 1)
        output_sc_FcLayer1_tensor_norm = norm_sc_FcLayer1_tensor.squeeze(0).squeeze(0).tolist()

        output_sc_FcLayer2 = torch.tensor(ScCudaTorch.ScCudaFcLayer(output_sc_FcLayer1_tensor_norm, self.new_weightsfc4, self.new_biasfc4, self.new_INPUTfc_rndomMatrix4, self.new_weightsfc_rndomMatrix4, self.new_biasfc_rndomMatrix4, self.noOutputs4, self.bitstream_Length))
        output_sc_FcLayer2 = (output_sc_FcLayer2/(no_accumulations4*self.bitstream_Length))*no_accumulations4
        second_scfc_layer = F.relu(output_sc_FcLayer2)
        norm_sc_FcLayer2_tensor = torch.clamp(second_scfc_layer, 0, 1)
        output_sc_FcLayer2_tensor_norm = norm_sc_FcLayer2_tensor.squeeze(0).squeeze(0).tolist()

        output_sc_FcLayer3 = torch.tensor(ScCudaTorch.ScCudaFcLayer(output_sc_FcLayer2_tensor_norm, self.new_weightsfc5, self.new_biasfc5, self.new_INPUTfc_rndomMatrix5, self.new_weightsfc_rndomMatrix5, self.new_biasfc_rndomMatrix5, self.noOutputs5, self.bitstream_Length))
        output_sc_FcLayer3 = (output_sc_FcLayer3/(no_accumulations5*self.bitstream_Length))*no_accumulations5
        
        return output_sc_FcLayer3.unsqueeze(0)
 
class SCCNN9(nn.Module):
    def __init__(self, num_classes=10):
        super(SCCNN9, self).__init__()
        
        # Padding externally (ZeroPad2d) for each convolution
        self.pad1 = nn.ZeroPad2d(1)  # 1 padding on all sides for 3x3 kernel
        
         # Block 1: 2 Conv layers (with external padding) + MaxPooling
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=0)  # padding set to 0
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 2 Conv layers (with external padding) + MaxPooling
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 2 Conv layers (with external padding) + MaxPooling (Extra conv layer added)
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)  # Extra convolutional layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 4 * 4, 256)  # Assuming input image size is 32x32 (CIFAR10-like)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        # Block 1: External padding before each conv layer + ReLU + Clamp [0, 1]
        x = self.pad1(x)
        x = F.relu(self.conv1_1(x))
        x = torch.clamp(x, 0, 1)
        x = self.pad1(x)
        x = F.relu(self.conv1_2(x))
        x = torch.clamp(x, 0, 1)
        x = self.pool1(x)
        
        # Block 2: External padding before each conv layer + ReLU + Clamp [0, 1]
        x = self.pad1(x)
        x = F.relu(self.conv2_1(x))
        x = torch.clamp(x, 0, 1)
        x = self.pad1(x)
        x = F.relu(self.conv2_2(x))
        x = torch.clamp(x, 0, 1)
        x = self.pool2(x)
        
        # Block 3: External padding before each conv layer + ReLU + Clamp [0, 1]
        x = self.pad1(x)
        x = F.relu(self.conv3_1(x))
        x = torch.clamp(x, 0, 1)
        x = self.pad1(x)
        x = F.relu(self.conv3_2(x))  # New convolutional layer
        x = torch.clamp(x, 0, 1)
        x = self.pool3(x)
        
        # Flatten the feature maps for fully connected layers
        x = x.view(-1, 32 * 4 * 4)  # Adjust based on input size
        
        # Fully connected layers + ReLU + Clamp [0, 1]
        x = F.relu(self.fc1(x))
        x = torch.clamp(x, 0, 1)
        x = F.relu(self.fc2(x))
        x = torch.clamp(x, 0, 1)
        x = self.fc3(x)
        
        return x

# Create the model instance
sccnn9 = SCCNN9(num_classes=10)
PATH = './cifar_sccnn9_67.pth'
sccnn9.load_state_dict(torch.load(PATH))

############################ PARAMETERS ####################################
bitstream_Length = 128
print("SC VGG9: ", bitstream_Length)
############################ FIRST SC CONV2D LAYER #########################
custom_weights1 = sccnn9.conv1_1.weight.data
custom_biases1 = sccnn9.conv1_1.bias.data
input_channels1 = 3 
output_channels1 = 8
h_INPUT1 = 34*34 
h_KERNEL1 = 3*3 
############################ SECOND SC CONV2D LAYER ########################
custom_weights2 = sccnn9.conv1_2.weight.data
custom_biases2 = sccnn9.conv1_2.bias.data
input_channels2 = 8
output_channels2 = 8
h_INPUT2 = 34*34
h_KERNEL2 = 3*3
############################ THIRD SC CONV2D LAYER #########################
custom_weights3 = sccnn9.conv2_1.weight.data
custom_biases3 = sccnn9.conv2_1.bias.data
input_channels3 = 8 
output_channels3 = 16
h_INPUT3 = 18*18 
h_KERNEL3 = 3*3 
############################ FOURTH SC CONV2D LAYER ########################
custom_weights4 = sccnn9.conv2_2.weight.data
custom_biases4 = sccnn9.conv2_2.bias.data
input_channels4 = 16
output_channels4 = 16
h_INPUT4 = 18*18
h_KERNEL4 = 3*3
############################ FIFTH SC CONV2D LAYER #########################
custom_weights5 = sccnn9.conv3_1.weight.data
custom_biases5 = sccnn9.conv3_1.bias.data
input_channels5 = 16 
output_channels5 = 32
h_INPUT5 = 10*10 
h_KERNEL5 = 3*3 
############################ SIXTH SC CONV2D LAYER ########################
custom_weights6 = sccnn9.conv3_2.weight.data
custom_biases6 = sccnn9.conv3_2.bias.data
input_channels6 = 32
output_channels6 = 32
h_INPUT6 = 10*10
h_KERNEL6 = 3*3
############### 1ST SC LINEAR LAYER ################
new_weightsfc3 = torch.flatten(sccnn9.fc1.weight.data.t(), 0)
new_weightsfc3 = new_weightsfc3.tolist()
#new_weightsfc3 = sccnn9.fc1.weight.data.t().tolist()
new_biasfc3 = sccnn9.fc1.bias.data
new_biasfc3 = new_biasfc3.tolist()
noInputs3 = 32*4*4
noOutputs3 = 256
no_accumulations3 = noInputs3 + 1
new_INPUTfc_rndomMatrix3 = ScCudaTorch.generate_random_matrix(noInputs3, bitstream_Length, rng_type)
new_weightsfc_rndomMatrix3 = ScCudaTorch.generate_random_matrix(noInputs3*noOutputs3, bitstream_Length, rng_type)
new_biasfc_rndomMatrix3 = ScCudaTorch.generate_random_matrix(noOutputs3, bitstream_Length, rng_type)
############### 2ND SC LINEAR LAYER ################
new_weightsfc4 = torch.flatten(sccnn9.fc2.weight.data.t(), 0)
new_weightsfc4 = new_weightsfc4.tolist()
#new_weightsfc4 = sccnn9.fc2.weight.data.t().tolist()
new_biasfc4 = sccnn9.fc2.bias.data
new_biasfc4 = new_biasfc4.tolist()
noInputs4 = 256
noOutputs4 = 120
no_accumulations4 = noInputs4 + 1
new_INPUTfc_rndomMatrix4 = ScCudaTorch.generate_random_matrix(noInputs4, bitstream_Length, rng_type)
new_weightsfc_rndomMatrix4 = ScCudaTorch.generate_random_matrix(noInputs4*noOutputs4, bitstream_Length, rng_type)
new_biasfc_rndomMatrix4 = ScCudaTorch.generate_random_matrix(noOutputs4, bitstream_Length, rng_type)
################ 3rd LINEAR LAYER ::SC:: ################
new_weightsfc5 = torch.flatten(sccnn9.fc3.weight.data.t(), 0)
new_weightsfc5 = new_weightsfc5.tolist()
#new_weightsfc5 = sccnn9.fc3.weight.data.t().tolist()
new_biasfc5 = sccnn9.fc3.bias.data
new_biasfc5 = new_biasfc5.tolist()
noInputs5 = 120
noOutputs5 = 10
no_accumulations5 = noInputs5 + 1
new_INPUTfc_rndomMatrix5 = ScCudaTorch.generate_random_matrix(noInputs5, bitstream_Length, rng_type)
new_weightsfc_rndomMatrix5 = ScCudaTorch.generate_random_matrix(noInputs5*noOutputs5, bitstream_Length, rng_type)
new_biasfc_rndomMatrix5 = ScCudaTorch.generate_random_matrix(noOutputs5, bitstream_Length, rng_type)

scNet = ScNet(custom_weights1, custom_biases1, custom_weights2, custom_biases2,
          custom_weights3, custom_biases3, custom_weights4, custom_biases4,
          custom_weights5, custom_biases5, custom_weights6, custom_biases6, 
          new_weightsfc3, new_biasfc3, new_INPUTfc_rndomMatrix3, new_weightsfc_rndomMatrix3, new_biasfc_rndomMatrix3, noOutputs3,
          new_weightsfc4, new_biasfc4, new_INPUTfc_rndomMatrix4, new_weightsfc_rndomMatrix4, new_biasfc_rndomMatrix4, noOutputs4,
          new_weightsfc5, new_biasfc5, new_INPUTfc_rndomMatrix5, new_weightsfc_rndomMatrix5, new_biasfc_rndomMatrix5, noOutputs5,
          bitstream_Length)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# import torchvision
# import torchvision.transforms as transforms

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
# #                                         download=True, transform=transform)
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
# #                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1,
#                                          shuffle=False, num_workers=1)

# #prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# correct_images = {classname: [] for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         out = sccnn9(images)
#         _, predictions = torch.max(out, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
                
#                 # Only store the first 100 correctly predicted images per class
#                 if len(correct_images[classes[label]]) < 30:
#                     correct_images[classes[label]].append(images.cpu())
                    
#             total_pred[classes[label]] += 1

# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
# PATH = './correct_images_CIFAR.pth'
# torch.save(correct_images, PATH)



# class_name = 'plane'
# print("SC VGG9 COMPACT: ", class_name)

# correct_pred_class = 0
# total_pred_class = 0

# PATH = './correct_images_CIFAR.pth'
# correct_images_loaded = torch.load(PATH)
# images_list = correct_images_loaded[class_name]

# start_time = time.time()
# with torch.no_grad():
#     for images in images_list:
        
#         images = images.squeeze(0)
        
#         outputs = scNet(images.unsqueeze(0))
#         _, predictions = torch.max(outputs, 1)
        
#         true_label = classes.index(class_name)
        
#         if predictions.item() == true_label:
#             correct_pred_class += 1
        
#         total_pred_class += 1
# print(f"Execution time: {(time.time() - start_time):.2f} seconds")

# if total_pred_class > 0:
#     accuracy_class = 100 * float(correct_pred_class) / total_pred_class
#     print(f'Accuracy for class: {class_name} with scnet is {accuracy_class:.1f} %')
# else:
#     print(f'No predictions for class: {class_name}')



class_name = 'car'
print("SC VGG9 COMPACT: ", class_name)

correct_pred_class = 0
total_pred_class = 0
mape = 0

PATH = './correct_images_CIFAR.pth'
correct_images_loaded = torch.load(PATH)
images_list = correct_images_loaded[class_name]

start_time = time.time()
with torch.no_grad():
    for images in images_list:
        
        images = images.squeeze(0)
        
        outputs = scNet(images.unsqueeze(0))
        _, predictions = torch.max(outputs, 1)
        
        true_label = classes.index(class_name)

        outs_original = sccnn9(images)
        mape += mean_absolute_percentage_error(outputs, outs_original)
        
        if predictions.item() == true_label:
            correct_pred_class += 1
        
        total_pred_class += 1
print(f"Execution time: {(time.time() - start_time):.2f} seconds")

if total_pred_class > 0:
    accuracy_class = 100 * float(correct_pred_class) / total_pred_class
    print(f'Accuracy for class: {class_name} with scnet is {accuracy_class:.1f} %')
else:
    print(f'No predictions for class: {class_name}')

print(f"mean_absolute_percentage_error: ", mape/30)