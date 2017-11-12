import numpy as np
from pyAudioAnalysis import audioBasicIO
import Audio_Feature_Extraction as AFE
from sklearn import mixture
from sklearn.externals import joblib
import glob
import os

def extract_MFCCs(file_name,x,Fs,window_size,overlap,deltas=False):

	MFCC13_F = AFE.stFeatureExtraction(file_name,x, Fs, window_size, overlap)

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

def extract_prosody(file_name,sp_folder,emo_folder,type_folder):
	features=[]
	for line in open("Features/prosody/"+type_folder+"/"+emo_folder+"/"+sp_folder+'/'+file_name+"_prosody"):
		row=map(float,line.strip().split(',')) 
		features.append(row)
	return np.array(features)

def extract_lpcc(file_name,sp_folder,emo_folder,type_folder):
	features=[]
	for line in open("Features/LPCC/"+type_folder+"/"+emo_folder+"/"+sp_folder+'/'+file_name+"_lpcc"):
		row=map(float,line.strip().split(' ')) 
		row=row[1:]
		features.append(row)
	return np.array(features)

if __name__=="__main__":

	window = 0.020
	window_overlap = 0.010
	n_mixtures = 128
	max_iterations = 400
	calc_deltas=True

	all_emotions=sorted(glob.glob('train_wavdata/*'))

	emotions = { all_emotions[k]:k for k in range(len(all_emotions)) }

	if len(glob.glob('train_models/*.pkl'))>0:
		for f in glob.glob('train_models/*.pkl'):
			os.remove(f)

	for emotion in all_emotions:

		emotion_name=emotion.replace('train_wavdata/','')
		
		all_emotion_Fs,all_emotion_data=0,[]

		speaker_list=glob.glob(emotion+'/*')

		for speaker in speaker_list:

			speaker_name=speaker.replace('train_wavdata/'+emotion_name+'/','')

			file_list=glob.glob(speaker+'/*.wav')

			for sample_file in file_list:
				try:
					[Fs, x] = audioBasicIO.readAudioFile(sample_file)
				except:
					continue
				if all_emotion_Fs==0:	all_emotion_Fs=Fs
				if Fs==all_emotion_Fs:
					mfcc_features = extract_MFCCs("Features/frame_data/"+sample_file,x,Fs,window*Fs,window_overlap*Fs,calc_deltas)

					actual_file_name = sample_file.replace('train_wavdata/'+emotion_name+'/'+speaker_name+'/','')
					#print actual_file_name
					
					prosody_features = extract_prosody(actual_file_name,speaker_name,emotion_name,"train_prosdata")

					lpcc_features = extract_lpcc(actual_file_name,speaker_name,emotion_name,"train_lpccdata")

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
							lpcc_features=lpcc_features[0:min_shape]

					
					all_emotion_data.append( np.concatenate([mfcc_features,prosody_features,lpcc_features],1) )
				else:	
					print sample_file+" skipped due to mismatch in frame rate"

		all_emotion_data = np.concatenate(all_emotion_data,0)

		np.savetxt('Features/ALL/training_data/'+emotion_name+"_all_features", all_emotion_data, delimiter=",")
		
		print emotion_name,all_emotion_data.shape

		try:
			gmm = mixture.GaussianMixture(n_components=n_mixtures, covariance_type='diag' , max_iter = max_iterations ).fit(all_emotion_data)
		except:
			print "ERROR : Error while training model for file "+emotion
		
		try:
			joblib.dump(gmm,'train_models/'+emotion_name+'.pkl')
		except:
			 print "ERROR : Error while saving model for "+emotion_name


	print "Training Completed"