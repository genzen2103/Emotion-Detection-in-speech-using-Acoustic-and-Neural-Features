from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from pyAudioAnalysis import audioBasicIO
import Audio_Feature_Extraction as AFE
from sklearn import mixture
from sklearn.externals import joblib
import glob
import os
import keras

#print(keras.__version__)

def extract_MFCCs(x,Fs,window_size,overlap,VTH_Multiplier=0.05,VTH_range=100,deltas=False):
	
	# energy = [ s**2 for s in x]
	# Voiced_Threshold = VTH_Multiplier*np.mean(energy)
	# clean_samples=[]
	
	# for i in xrange(0,len(x),VTH_range):
	# 	sample_set_th = np.mean(energy[i:i+VTH_range])
	# 	if sample_set_th>Voiced_Threshold:
	# 		sample=x[i:i+VTH_range]
	# 		clean_samples.extend(list(sample))

	MFCC13_F = AFE.stFeatureExtraction(x, Fs, window_size, overlap)

	energy=None
	clean_samples=None

	if not deltas:
		return MFCC13_F
	
	delta_MFCC = np.zeros(MFCC13_F.shape)
	for t in range(delta_MFCC.shape[1]):
		index_t_minus_one,index_t_plus_one=t-1,t+1
		
		if index_t_minus_one<0:    
			index_t_minus_one=0
		if index_t_plus_one>=delta_MFCC.shape[1]:
			index_t_plus_one=delta_MFCC.shape[1]-1
		
		delta_MFCC[:,t]=0.5*(MFCC13_F[:,index_t_plus_one]-MFCC13_F[:,index_t_minus_one] )
	
	
	double_delta_MFCC = np.zeros(MFCC13_F.shape)
	for t in range(double_delta_MFCC.shape[1]):
		
		index_t_minus_one,index_t_plus_one, index_t_plus_two,index_t_minus_two=t-1,t+1,t+2,t-2
		
		if index_t_minus_one<0:
			index_t_minus_one=0
		if index_t_plus_one>=delta_MFCC.shape[1]:
			index_t_plus_one=delta_MFCC.shape[1]-1
		if index_t_minus_two<0:
			index_t_minus_two=0
		if index_t_plus_two>=delta_MFCC.shape[1]:
			index_t_plus_two=delta_MFCC.shape[1]-1
		  
		double_delta_MFCC[:,t]=0.1*( 2*MFCC13_F[:,index_t_plus_two]+MFCC13_F[:,index_t_plus_one]
									-MFCC13_F[:,index_t_minus_one]-2*MFCC13_F[:,index_t_minus_two] )
	
	Combined_MFCC_F = np.concatenate((MFCC13_F,delta_MFCC,double_delta_MFCC),axis=1)
	
	return Combined_MFCC_F


if __name__=="__main__":
	emotions={'angry_female':0,'angry_male':1,'happy_female':2,'happy_male':3,'neutral_female':4,'neutral_male':5,'sad_female':6,'sad_male':7,'surprised_female':8,'surprised_male':9}
	window = 0.100
	window_overlap = 0.50
	voiced_threshold_mul = 0.50
	voiced_threshold_range=100
	n_mixtures = 128
	max_iterations = 400
	calc_deltas=True
	n_hidden=21
	n_input=39
	b_size=5
	n_epochs=10
	n_cross_val=5

	n_output=len(emotions)
	model = Sequential()
	model.add(Dense(n_hidden, input_shape=(n_input,),activation='relu'))
	model.add(Dense(n_output,activation='softmax'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

	spct=0
	total_sp=len(glob.glob('train_wavdata/*'))

	
	if len(glob.glob('train_models/*.pkl'))>0:
		for f in glob.glob('train_models/*.pkl'):
			os.remove(f)

	x_train,y_train=[],[]


	for emotion in sorted(glob.glob('train_wavdata/*')):
		
		#print (spct/float(total_sp))*100.0,'% completed'

		emotion_name=emotion.replace('train_wavdata/','')
		
		all_emotion_Fs,all_emotion_data=0,[]
		file_list=glob.glob(emotion+'/*.wav')

		for sample_file in file_list:

			[Fs, x] = audioBasicIO.readAudioFile(sample_file)
			
			if all_emotion_Fs==0:	
				all_emotion_Fs=Fs

			if Fs==all_emotion_Fs:
				features = extract_MFCCs(x,Fs,window*Fs,window_overlap*Fs,voiced_threshold_mul,voiced_threshold_range,calc_deltas)
				all_emotion_data.append(features)
			else:	
				print sample_file+" skipped due to mismatch in frame rate"

		all_emotion_data=np.concatenate(all_emotion_data,0)
		print all_emotion_data.shape
		all_emotion_labels = []
		for i in xrange(all_emotion_data.shape[0]):
			d = np.zeros(len(emotions))
			d[emotions[emotion_name]]=1
			all_emotion_labels.append(d)
		all_emotion_labels=np.array(all_emotion_labels)

		for i in xrange(all_emotion_data.shape[0]):
			x_train.append(all_emotion_data[i])
			y_train.append(all_emotion_labels[i])
		
		spct+=1

	x_train=np.array(x_train)
	y_train=np.array(y_train)
	
	print "Training data size:",x_train.shape

	cv_scores=[]
	for cv in xrange(n_cross_val):
		print "Cross Validation Iteration:",cv
		perm=np.random.permutation(x_train.shape[0])
		x_train=x_train[perm]
		y_train=y_train[perm]
		model.fit(x_train,y_train,nb_epoch=n_epochs,batch_size=b_size,validation_split=0.25)

		print "Training Completed"


		x_test,y_test=[],[]
		for emotion in emotions:
			full_data=[]
			tct=0
			for testcasefile in glob.glob('test_wavdata/'+emotion+'/*.wav'):
				[Fs, x] = audioBasicIO.readAudioFile(testcasefile)
				features = extract_MFCCs(x,Fs,window*Fs,window_overlap*Fs,voiced_threshold_mul,voiced_threshold_range,calc_deltas)
				full_data.append(features)
				tct+=1
			full_data=np.concatenate(full_data,0)
			full_labels=np.zeros(full_data.shape)
			for i in xrange(full_data.shape[0]):
				full_labels[i][emotions[emotion]]=1
				x_test.append(full_data[i])
				y_test.append(full_labels[i])

		x_test=np.array(x_test)
		y_test=np.array(y_test)
		print "Test data size:",x_test.shape

		classes = model.predict(x_test, batch_size=b_size)
		print classes.shape
		pred_labels=[]
		actual_labels=[]
		for i in xrange(classes.shape[0]):
			pred_labels.append(np.argmax(classes[i]))
			actual_labels.append(np.argmax(y_test[i]))

		confusion_matrix=np.zeros((len(emotions),len(emotions)))
		acc=0
		for i in xrange(len(y_test)):
			confusion_matrix[ actual_labels[i] ][ pred_labels[i] ]+=1
			if actual_labels[i]==pred_labels[i]:
				acc+=1
		print confusion_matrix
		print "Accuracy=",acc/float(tct)
		cv_scores.append((acc/float(tct))*100)
	print "Mean=",np.mean(cv_scores),"SD=",np.std(cv_scores)






