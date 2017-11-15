import numpy as np
from pyAudioAnalysis import audioBasicIO
import Audio_Feature_Extraction as AFE
from sklearn import mixture
from sklearn.externals import joblib
import glob
import os

def extract_MFCCs(x,Fs,window_size,overlap,deltas=False):

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

def extract_prosody(file_name,emo_folder):
	features=[]
	for line in open("Features/test_wavdata/prosody"+"/"+emo_folder+'/'+file_name+"_prosody"):
		row=map(float,line.strip().split(',')) 
		features.append(row)
	return np.array(features)

def extract_lpcc(file_name,emo_folder):
	features=[]
	for line in open("Features/test_wavdata/LPCC"+"/"+emo_folder+'/'+file_name+"_lpcc"):
		row=map(float,line.strip().split(' ')) 
		row=row[1:]
		features.append(row)
	return np.array(features)

if __name__=="__main__":

	window = 0.020
	window_overlap = 0.010
	calc_deltas=False

	all_emotions=sorted(glob.glob('../emodb_wav/test_wavdata/*'))

	all_emotions=[emo for emo in all_emotions if 'disgust' not in emo and 'fear' not in emo and 'boredom' not in emo]
	# print(all_emotions)
	emotions = { all_emotions[k]:k for k in range(len(all_emotions)) }
	
	num_test_cases={}
	tct={}
	for e in emotions:
		num_test_cases[e.replace('../emodb_wav/test_wavdata/','')]=len(os.listdir(e))-1
		tct[e.replace('../emodb_wav/test_wavdata/','')]=0

	print num_test_cases

	emotion_names = { all_emotions[k].replace('../emodb_wav/test_wavdata/',''):k for k in range(len(all_emotions)) }

	total_emotions=len(num_test_cases)


	confusion_matrix = np.zeros((total_emotions,total_emotions))


	for emotion in all_emotions:

		emotion_name=emotion.replace('../emodb_wav/test_wavdata/','')
		# speaker_name=speaker.replace(emotion+'/','')
		for testcasefile in glob.glob(emotion+'/*.wav'):

			[Fs, x] = audioBasicIO.readAudioFile(testcasefile)
			mfcc_features = extract_MFCCs(x,Fs,window*Fs,window_overlap*Fs,calc_deltas)
			actual_file_name = testcasefile.replace(emotion+"/",'')
			# print actual_file_name

			prosody_features = extract_prosody(actual_file_name,emotion_name)
			lpcc_features = extract_lpcc(actual_file_name,emotion_name)
			#print mfcc_features.shape,prosody_features.shape,lpcc_features.shape
			
			if mfcc_features.shape[0]==prosody_features.shape[0] and prosody_features.shape[0]==lpcc_features.shape[0]:
				pass
			else:
				min_shape=min([ mfcc_features.shape[0],prosody_features.shape[0],lpcc_features.shape[0] ])
				if mfcc_features.shape[0]!=min_shape:
					mfcc_features=mfcc_features[0:min_shape]
				if prosody_features.shape[0]!=min_shape:
					prosody_features=prosody_features[0:min_shape]
				if lpcc_features.shape[0]!=min_shape:
					lpcc_features=rpcc_features[0:min_shape]

			max_score=-np.inf
			max_emotion_name=""
			all_emotion_test=np.concatenate([mfcc_features,prosody_features,lpcc_features],1)

			# np.savetxt('matrices/test/'+emotion_name+"#"+str(tct[emotion_name])+"_all_features", all_emotion_test, delimiter=",")
			
			for modelfile in sorted(glob.glob('train_models/*.pkl')):
				gmm = joblib.load(modelfile) 
				score=gmm.score(all_emotion_test)
				#print score
				if score>max_score:
					max_score,max_emotion_name=score,modelfile.replace('train_models/','').replace('.pkl','')
			
			print emotion_name+" -> "+max_emotion_name+(" Y" if emotion_name==max_emotion_name  else " N")
			
			confusion_matrix[ emotion_names[emotion_name] ][emotion_names[max_emotion_name]]+=1
			tct[emotion_name]+=1
	print total_emotions
	print tct
	print "Confusion Matrix:\n",confusion_matrix
	print "Accuracy: ",(sum([ confusion_matrix[i][j] if i==j  else 0 for i in xrange(total_emotions) for j in xrange(total_emotions) ] )*100)/float(sum([i for i in tct.values()]))





