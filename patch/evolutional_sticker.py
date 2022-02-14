from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import cv2
import matplotlib

#Parameters
BatchLength=1  #32 images are in a minibatch
Size=[28, 28, 1] #Input img will be resized to this size
ImageNum=84
LearningRate = 1e-1 #learning rate of the algorithm
NumClasses = 2 #number of output classes
EvalFreq=100 #evaluate on every 100th iteration
#select sticker position randomly
StickerColors =[255.0, 255.0,0.0, 0.0]#255 white, 0 Black 
StickerSize=[2,2,1]
StickerPosition=[6,6]

#load data
TrainData= np.load('train_data.npy')
TrainLabels=np.load('train_labels.npy')
TestData= np.load('test_data.npy')
TestLabels=np.load('test_labels.npy')


# Create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2] ]) #network input
InputLabels = tf.placeholder(tf.int32, [None]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)

    
#loads the same model, but all variables are frozen in this model
NumKernels = [32,32,32]
def MakeConvNet(Input,Size):
    CurrentInput=Input
    CurrentFilters = Size[2] #the input dim at the first layer is 1, since the input image is grayscale
    for i in range(len(NumKernels)): #number of layers
        with tf.variable_scope('conv'+str(i)):
                NumKernel=NumKernels[i]
                W = tf.get_variable('W',[3,3,CurrentFilters,NumKernel], trainable=False)
                Bias = tf.get_variable('Bias',[NumKernel],initializer=tf.constant_initializer(0.0),trainable=False)
		
                CurrentFilters = NumKernel
                ConvResult = tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1],padding='VALID') #VALID, SAME
                ConvResult= tf.add(ConvResult, Bias)
                
                ReLU = tf.nn.relu(ConvResult)
                
                CurrentInput = tf.nn.max_pool(ReLU,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    #add fully connected network
    with tf.variable_scope('FC'):
	    CurrentShape=CurrentInput.get_shape()
	    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
	    FC = tf.reshape(CurrentInput, [-1, FeatureLength])
	    FCInput = FC
	    W = tf.get_variable('W',[FeatureLength,NumClasses], trainable=False)
	    FC = tf.matmul(FC, W)
	    Bias = tf.get_variable('Bias',[NumClasses], trainable=False)
	    FC = tf.add(FC, Bias)
	    #FC = tf.nn.softmax(FC)	    
    return FC

	
# Construct model
PredWeights = MakeConvNet(InputData, Size)



# Initializing the variables
Init = tf.global_variables_initializer()

# Launch the session
with tf.Session() as Sess:
	Sess.run(Init)
	saver = tf.train.Saver() # we restore all variables 
	saver.restore(Sess, "./model/mymodel-110")
	#saver.restore(Sess, "./model/mymodelbounded-110")
	OrigData=np.reshape(TestData[ImageNum,:,:,:],[BatchLength]+Size)
	print(np.min(OrigData))
	print(np.max(OrigData))
	Pred  = Sess.run([ PredWeights ], feed_dict={InputData: OrigData,InputLabels: TestLabels[ImageNum]})
	Pred=Pred[0][0]
	print(Pred)
	#this is in class 1...lets move it to classs zero
	MinWeight=Pred[1]-Pred[0]
	MinPos=[]
	MinSize=[]
	print(MinWeight)
	#we have two white and two black stickers
	GenomeSize=500
	NumSteps=10
	KeepRatio=0.2
	NewRatio=0.1
	GeneratedRatio=1-(KeepRatio+NewRatio)
	MutationFactor=0.2
	Positions=np.zeros((GenomeSize,8))
	Sizes=np.zeros((GenomeSize,8))
	
	#generate Initial Genome
	for i in range(GenomeSize):
		Positions[i,:]=np.random.uniform(0,Size[1],8)
		Sizes[i,:]=np.random.uniform(0,5,8)
	
	
	for St in range(NumSteps):
		#calc weights
		Weights=np.zeros(GenomeSize)
		for i in range(GenomeSize):
			#put the sticker on the image:
			StickerData=np.copy(OrigData)
			StickerData[0,int(Positions[i,0]):int(Positions[i,0]+Sizes[i,0]),int(Positions[i,1]):int(Positions[i,1]+Sizes[i,1]),:]=0.0
			StickerData[0,int(Positions[i,2]):int(Positions[i,2]+Sizes[i,2]),int(Positions[i,3]):int(Positions[i,3]+Sizes[i,3]),:]=0.0
			StickerData[0,int(Positions[i,4]):int(Positions[i,4]+Sizes[i,4]),int(Positions[i,5]):int(Positions[i,5]+Sizes[i,5]),:]=255.0
			StickerData[0,int(Positions[i,6]):int(Positions[i,6]+Sizes[i,6]),int(Positions[i,7]):int(Positions[i,7]+Sizes[i,7]),:]=255.0

			Pred  = Sess.run([ PredWeights ], feed_dict={InputData: StickerData,InputLabels: TestLabels[ImageNum]})
			Pred=Pred[0][0]
			Weights[i]=Pred[1]-Pred[0]
			if Weights[i]<MinWeight:
				MinWeight=Weights[i]
				MinPos=Positions[i,:]
				MinSize=Sizes[i,:]
		print(MinWeight)
		#order the Population
		Indices=range(GenomeSize)
		Weights, Indices = zip(*sorted(zip(Weights, Indices)))
		KeptIndices=Indices[0:int(KeepRatio*GenomeSize)]
		GeneratedIndices=int((1.0-NewRatio)*GenomeSize)
		NewPositions=np.zeros((GenomeSize,8))
		NewSizes=np.zeros((GenomeSize,8))
		#elitism - keep the best elements
		for a in range(len(KeptIndices)):
			NewPositions[a,:]=Positions[KeptIndices[a],:]
			NewSizes[a,:]=Sizes[KeptIndices[a],:]
		#crossover for the generated ones
		for a in range(len(KeptIndices),GeneratedIndices):
			#select two samples
			Indices=np.random.choice(range(len(KeptIndices)), 2, replace=False)
			#select point of the crossover
			CrossPoint=np.random.randint(0,9)
			NewPositions[a,0:CrossPoint]=Positions[KeptIndices[Indices[0]]][0:CrossPoint]
			NewPositions[a,CrossPoint:8]=Positions[KeptIndices[Indices[1]]][CrossPoint:8]
			NewSizes[a,0:CrossPoint]=Sizes[KeptIndices[Indices[0]]][0:CrossPoint]
			NewSizes[a,CrossPoint:8]=Sizes[KeptIndices[Indices[1]]][CrossPoint:8]
		#rest is new
		for a in range(len(KeptIndices),GenomeSize):
			NewPositions[a,:]=np.random.uniform(0,Size[1],8)
			NewSizes[a,:]=np.random.uniform(0,5,8)
		
		#random mutation 
		for a in range(GenomeSize):
			if np.random.uniform<MutationFactor:
				NewPositions[a,:]+=numpy.random.normal(0,3,8)
				NewSizes[a,:]+=numpy.random.normal(0,3,8)
				for i in range(8):
					if NewSizes[a,i]>5:
						NewSizes[a,i]=5
					if NewSizes[a,i]<0:
						NewSizes[a,i]=0
		
		Positions=NewPositions
		Sizes=NewSizes
print(MinWeight)
print(MinPos)
print(MinSize)
