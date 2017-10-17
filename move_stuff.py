import glob
import os
from shutil import copyfile
import subprocess

emotions = ['angry','calm','disgust','fearful','happy','neutral','sad','surprised']
sets = ['train_wavdata','test_wavdata']
sexs = ['male','female']

# for s in sets:
# 	for folder in emotions:
# 		if not os.path.exists(s+'/'+folder+'/male'):
# 			os.makedirs(s+'/'+folder+'/male')
# 		if not os.path.exists(s+'/'+folder+'/female'):
# 			os.makedirs(s+'/'+folder+'/female')

# for emo in emotions:
# 	for g in ['male','female']:
# 		for file in glob.glob(emo+'/'+g+'/*.wav'):
# 			flist = file.split('-')
# 			num_s=flist[len(flist)-1].replace(".wav","")
# 			num=int(num_s)
# 			print num
# 			if num%2==0 and num in [4,14,22]:
# 				copyfile(file,"test_wavdata/"+file)
# 				os.remove(file)
# 			if num%2!=0 and num in [3,13,21]:
# 				copyfile(file,"test_wavdata/"+file)
# 				os.remove(file)


for s in sets:
	for e in emotions:
		for g in sexs:
			for file in glob.glob("/".join([s,e,g])+"/*.wav"):
				cmd = ['sox',file,"/".join([s,e,g])+"/output.wav",'remix','1']
				subprocess.call(cmd)
				os.remove(file)
				os.rename("/".join([s,e,g])+"/output.wav",file)