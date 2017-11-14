from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Input, Concatenate,LSTM
import numpy as np
from sklearn.externals import joblib
import glob
import os
import keras


if __name__=="__main__":
	emotions={'angry_female':0,'angry_male':1,'happy_female':2,'happy_male':3,'neutral_female':4,'neutral_male':5,'sad_female':6,'sad_male':7}
	n_hidden1=20 #HP
	n_hidden2=10 #HP
	n_hc_input=52
	b_size=50 #HP
	n_epochs=10 #HP
	n_cross_val=1
	n_output=len(emotions)
	lstm_ip = 160 #const
	lstm_out = 16 #HP

	inputFr = Input(shape=(lstm_ip,1,))
	inputHc = Input(shape=(n_hc_input,))

	lstm_model = Sequential()
	lstm_model.add(LSTM(lstm_out,input_shape=(lstm_ip,1)))

	frame_enc = lstm_model(inputFr)

	dnn_input = Concatenate()([frame_enc,inputHc])

	desnse1 = Dense(n_hidden1 ,activation='relu')(dnn_input)

	desnse2 = Dense(n_hidden2, activation='relu')(desnse1)

	softmax = Dense(n_output,activation='softmax')(desnse2)

	model = Model(inputs = [inputFr,inputHc],outputs=softmax)
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	model.summary()


	xf_train,xh_train,y_train=[],[],[]

	
	for emotion in emotions:
		for file in glob.glob('ravdess_data/ALL/testing_data/'+emotion+'*_hc'):
			fr=np.genfromtxt(file.replace('_hc','')+'_fr', delimiter=',')
			hc=np.genfromtxt(file, delimiter=',')
			#print hc.shape,fr.shape
			labels = []
			for i in xrange(hc.shape[0]):
				d = np.zeros(len(emotions))
				d[emotions[emotion]]=1
				labels.append(d)
			labels=np.array(labels)

			for i in xrange(hc.shape[0]):
				xf_train.append( fr[i] )
				xh_train.append( hc[i] )
				y_train.append(labels[i])


	xf_train=np.array(xf_train)
	xh_train=np.array(xh_train)
	y_train=np.array(y_train)

	
	print "Training data size:",xf_train.shape,xh_train.shape

	cv_scores=[]
	for cv in xrange(n_cross_val):
		print "Cross Validation Iteration:",cv
		perm=np.random.permutation(xf_train.shape[0])
		xf_train=xf_train[perm]
		xh_train=xh_train[perm]
		y_train=y_train[perm]	

		model.fit([xf_train.reshape(xf_train.shape[0],xf_train.shape[1],1),xh_train],y_train,epochs=n_epochs,batch_size=b_size,validation_split=0.20)

		print "Training Completed"


		total_tests=0
		t_acc=0
		total_frames=0
		f_acc=0
		confusion_matrix=np.zeros((len(emotions),len(emotions)))
		for emotion in emotions:
			for testcasefile in glob.glob('ravdess_data/ALL/testing_data/'+emotion+'*_hc'):
				total_tests+=1
				fr_data=np.genfromtxt(file.replace('_hc','')+'_fr', delimiter=',')
				hc_data=np.genfromtxt(file, delimiter=',')
				#print hc_data.shape,fr_data.shape
				labels=np.zeros((fr_data.shape[0],n_output))
				labels[:,emotions[emotion]]+=1
				print "Test data size:",fr_data.shape
				print "Score:",model.evaluate([ fr_data.reshape(fr_data.shape[0],fr_data.shape[1],1),hc_data ], labels, batch_size=b_size)
				classes = model.predict([ fr_data.reshape(fr_data.shape[0],fr_data.shape[1],1),hc_data ], batch_size=b_size)
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
	
	print "Mean=",np.mean(cv_scores),"SD=",np.std(cv_scores)






