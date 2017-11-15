from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
from sklearn.externals import joblib
import glob
import os
import keras



if __name__=="__main__":
	emotions={'angry_female':0,'angry_male':1,'happy_female':2,'happy_male':3,'neutral_female':4,'neutral_male':5,'sad_female':6,'sad_male':7}
	n_hidden1=30
	n_hidden2=15
	n_input=54
	b_size=50
	n_epochs=50
	n_cross_val=1
	n_output=len(emotions)

	model = Sequential()
	model.add(Dense(n_hidden1, input_shape=(n_input,),activation='relu'))
	model.add(Dense(n_hidden2, activation='relu'))
	model.add(Dense(n_output,activation='softmax'))

	x_train,y_train=[],[]


	for emotion in emotions:
		all_emotion_data=np.genfromtxt('matrices/train/'+emotion+"_all_features", delimiter=',')
		# all_emotion_data=all_emotion_data[:,:-1]
		print all_emotion_data.shape
		all_emotion_labels = []
		for i in xrange(all_emotion_data.shape[0]):
			d = np.zeros(len(emotions))
			d[emotions[emotion]]=1
			all_emotion_labels.append(d)
		all_emotion_labels=np.array(all_emotion_labels)

		for i in xrange(all_emotion_data.shape[0]):
			x_train.append(all_emotion_data[i])
			y_train.append(all_emotion_labels[i])
		

	x_train=np.array(x_train)
	y_train=np.array(y_train)
	
	print "Training data size:",x_train.shape

	cv_scores=[]
	for cv in xrange(n_cross_val):
		print "Cross Validation Iteration:",cv

		model.compile(loss='categorical_crossentropy',optimizer='adamax',metrics=['accuracy'])

		perm=np.random.permutation(x_train.shape[0])
		x_train=x_train[perm]
		y_train=y_train[perm]
		model.fit(x_train,y_train,epochs=n_epochs,batch_size=b_size,validation_split=0.20)

		print "Training Completed"


		total_tests=0
		t_acc=0
		total_frames=0
		f_acc=0
		confusion_matrix=np.zeros((len(emotions),len(emotions)))
		for emotion in emotions:
			for testcasefile in glob.glob('matrices/test/'+emotion+'*'):
				total_tests+=1
				data=np.genfromtxt(testcasefile, delimiter=',')
				# data=data[:,:-1]
				labels=np.zeros((data.shape[0],n_output))
				labels[:,emotions[emotion]]+=1
				print "Test data size:",data.shape
				#print "Score:",model.evaluate(data, labels, batch_size=b_size)
				classes = model.predict(data, batch_size=b_size)
				pred_labels=[ np.argmax(classes[i]) for i in xrange(classes.shape[0]) ]
	
				for i in xrange(len(pred_labels)):
					total_frames+=1
					confusion_matrix[emotions[emotion]][pred_labels[i]]+=1
					if pred_labels[i]==emotions[emotion]:
						f_acc+=1
				
				if np.argmax(confusion_matrix[emotions[emotion]])==emotions[emotion]:
					t_acc+=1

		frame_acc=(f_acc/float(total_frames))*100
		test_acc=(t_acc/float(total_tests))*100
		print "Frame Level Accuracy=",frame_acc
		print "Test Accuracy=",test_acc
		cv_scores.append(test_acc)
		print confusion_matrix
	
	print "Mean=",np.mean(cv_scores),"SD=",np.std(cv_scores)






