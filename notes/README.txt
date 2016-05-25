How is running represented in mouse motor cortex?


Overview:

Here we provide data recorded from the motor cortex of a mouse while he is running in various directions (or just holding still) on a floating styrofoam sphere.  The data is taken using two-photon calcium imaging in the laboratory of Prof. Sandra Kuhlman.  Raw data of this type consists of images of the cortical surface where you can see neurons light up with a burst of fluorescence whenever the neuron is active.  We have pre-processed the data for you so that instead of dealing with images, you have a time series of flourescence activity from each of the 124 simultaneously recorded neurons.  Your task is to see if you can find a relationship between the neural activity and the mouse's behavior.


Data:

The data are available as a Matlab (.mat) file.  The .mat file consists of a single variable called data, a structure with the following fields:

data.sig
data.label
data.first_file_inds

data.sig is a T-x-N matrix of flourescence activity from T timepoints and N neurons.  T=43988 and N=124.  

data.label is a T-x-1 vector that describes the mouse's behavior at every time point.  It can take on 5 values:
0: mouse not moving
1: mouse moving between 315 and 45 degrees
2: mouse moving between 45 and 135 degrees
3: mouse moving between 135 and 225 degrees
4: mouse moving between 225 and 315 degrees

The data come from 12 non-contiguous recording blocks.  All time points are contiguous *except* where the files were joined together.  data.first_file_inds is a 12-x-1 vector of the first time index of each of the twelve recording blocks.


Questions/Hints/Directions:

How well can you predict what the mouse is doing from the flourescence?

Is the prediction better for future activity, current activity, or past activity?

Two-photon calcium images are characterized by rapid onsets and slow offsets.  The rapid onsets tend to be reliable indicators of electrical activity in the neuron, but the offset can be slow as the calcium is flushed out of the cell.  Do you get a better relationship between fluorescence and behavior if you only consider the onsets?

Recordings were made at 15Hz, i.e., every timepoint represents ~67ms of neural activity.
