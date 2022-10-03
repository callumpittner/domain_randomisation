---READ ME---

The programs are described as follows:

best_model_with_dropout.py -> Build the optimal CNN model for greyscale format, with Dropout included.
**PROGRAM FEATURES**
Trains a CNN architecture over a set number of epochs (change as needed). Outputs visualisations
for the accuracy and losses.


keras_tuner_cnn_optimiser.py -> Keras_tuner hyperparameter optimisation script, search over different architectures. 
**PROGRAM FEATURES**
Performs extensive hyperparameter optimisation. This script will require expensive resources and time.
Please ensure program does not crash or power does not cut to prevent data loss.
Parameters to consider: image size, batch size, number of classes, validation split, No. CNN layers, Pooling.


blender_domain_randomise.py -> The main Blender dataset generator script, with random domains.
**PROGRAM FEATURES**
Performs domain randomisation over best practices found through a literature review.
Future work should delve into different methods *Refer to Report*


To run the first two CNN programs it is recommended to use GPUs and run via command line.
Install the following in the Python environment:
**INSTALL ALL TO LATEST STABLE VERSIONS**
--numpy
--pandas
--matplotlib
--os
--tensorflow
--keras
--keras_tuner

For the Blender script it is recommended to use the Blender's script IDE
Install the following:
**INSTALL ALL TO LATEST STABLE VERSIONS**
--bpy
--random
--math
--mathutils
--os