from flask import Flask
from flask import jsonify
from flask import request
from flask_mysqldb import MySQL
from keras.applications import vgg16
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Sequential
from keras.models import Model
from keras import backend as K
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.serving import make_server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simplejson as json
import urllib.request
import requests
import shutil
import os
from googletrans import Translator
import collections
from gensim import corpora, models, similarities
import jieba
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from sqlalchemy_utils import create_database, database_exists

app = Flask(__name__)

class DataStore():
	a = None
	b = None
	c = None
	d = None

port = 9005
   
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'tritonre'

mysql = MySQL(app)

@app.route('/post-image', methods=['POST'])
def postImage():
	

	getToken=request.args['token']

	
	print(getToken)

	if getToken == "5fK2Zh50fDVXJxsCp7KZzYejopzLgCu":
		DataStore.a=request.args['userId']
		DataStore.b=request.args['postId']
		DataStore.c=request.args['imagePath']
		

		userId = DataStore.a
		postId = DataStore.b
		imagePath = DataStore.c
		
		print(userId)
		print(postId)
		print(imagePath)


		tritonurl1 = 'http://91dd8dcc.eu.ngrok.io'
	#	tritonurl2 = 'http://127.0.0.1:9000/'
		path = "Desktop/triton/image/"
	#	serImgPath="/api/images/download/"
	#	serReqPath="/api/posts/recommended"
		serImgPath="/api/images/download/"
		serReqPath="/api/posts/recommended/multimedia-based"
		dbTable="sn"

		samePost=[]

		cursor = mysql.connection.cursor()
		cursor.execute("SELECT COUNT(*) FROM "+ dbTable+ " WHERE userId = %s AND postId= %s", (userId, postId))
		row = cursor.fetchone()
		samePost.append(row[0])	
		cursor.close()
		


		if samePost[0]==0:
			cursor = mysql.connection.cursor()
			cursor.execute("INSERT INTO "+dbTable+"(userID, postId, imagePath) VALUES (%s, %s, %s)", (userId, postId, imagePath))
			mysql.connection.commit()
			cursor.close()
			
		

		else:
			deleteImage = []
			cursor = mysql.connection.cursor()
			cursor.execute("SELECT imagePath FROM "+dbTable+" WHERE userId LIKE %s AND postId LIKE %s", (userId, postId))
			row = cursor.fetchone( )
			deleteImage.append(row[0])	
			os.remove(path+deleteImage[0])
			cursor.execute("UPDATE "+dbTable+" SET imagePath = %s WHERE userId = %s AND postId = %s", (imagePath, userId, postId))
			mysql.connection.commit()
			cursor.close()
		
			
		url = tritonurl1+serImgPath+imagePath
		urllib.request.urlretrieve(url, path+imagePath)
		
		files = [image for image in os.listdir(path) ]

		imageNumber = []

		cursor = mysql.connection.cursor()
		cursor.execute("SELECT COUNT(*) FROM "+dbTable)
		row = cursor.fetchone()
		imageNumber.append(row[0])	
		cursor.close()
		
		nb_closest_images = imageNumber[0]
		


	#////////////////**********/////////////////////************* TRITON Recommendation Engine ***********/////////////////************//////////////////////
		
		K.clear_session()
		
		importedImages = []

		vgg_model = vgg16.VGG16(weights='imagenet')
		
		feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
		
		feat_extractor.summary()
		
		for f in files:
			filename = f
			original = load_img(path+filename, target_size=(224, 224))
			numpy_image = img_to_array(original)
			image_batch = np.expand_dims(numpy_image, axis=0)
			importedImages.append(image_batch)
			
		images = np.vstack(importedImages)

		processed_imgs = preprocess_input(images.copy())
			
		imgs_features = feat_extractor.predict(processed_imgs)

		print("features successfully extracted!")
		
		cosSimilarities = cosine_similarity(imgs_features)

		cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)
		
		print(cos_similarities_df)
		
		closest_imgs = cos_similarities_df[imagePath].sort_values(ascending=False)[0:nb_closest_images+1].index
		
		score = cos_similarities_df[imagePath].sort_values(ascending=False)[0:nb_closest_images+1]



	#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		


		if imageNumber[0]>0:
		
			'''
			original = load_img(path+imagePath, target_size=(224, 224))
			plt.imshow(original)
			plt.show() 
			plt.pause(3000)
			
			for i in range(0,len(closest_imgs)):
				original = load_img(path+closest_imgs[i], target_size=(224, 224))
				plt.imshow(original)
				plt.show()
			'''


			imageName=[]
			imageScore=[]
			
			for name in range(0,len(closest_imgs)):
				imageName.append(closest_imgs[name])
				imageScore.append(score[name])
			
			print(imageName)
			print(imageScore)
			
			rcmndPost=[]
			
			for name in imageName:
				
				print(name) 
				
				cursor = mysql.connection.cursor()
				cursor.execute("SELECT postId FROM "+dbTable+" WHERE imagePath LIKE %s", [name])
				row = cursor.fetchone()
				rcmndPost.append(row[0])	
				cursor.close()
			
			print(rcmndPost)		
		
			postList =  ",".join( repr(post) for post in rcmndPost )		
			scoreList =   ",".join( repr(score) for score in imageScore )				
			
			imageNameList = ' '.join(imageName)
			print(imageNameList)
			
			params = {'token': '5fK2Zh50fDVXJxsCp7KZzYejopzLgCu', 'userId': userId, 'postIds': postList, 'score':scoreList}
		
		else:
			params = {'token': '5fK2Zh50fDVXJxsCp7KZzYejopzLgCu', 'userId': userId, 'postIds':'0', 'score':'0'}

		K.clear_session()

		response = requests.post(tritonurl1+serReqPath, params=params)
		print (response.text)


		return "Success"	
	
	else:
		return "Unsuccess"



@app.route('/post-content', methods=['POST'])
def postContent():


	userId=request.args['userId']
	postId=request.args['postId']
	text=request.args['content']
	getToken = request.args['token']
	
	
	print(userId)
	print(postId)
	print(text)
	print(getToken)

	if getToken == "5fK2Zh50fDVXJxsCp7KZzYejopzLgCu":

		translator = Translator()
		translation = translator.translate(text)
		print(translation.text)

		dbTable="post"

		keyword = translation.text
		texts = []
		samePost=[]

		tritonurl2="http://91dd8dcc.eu.ngrok.io"
		serReqPath="/api/posts/recommended/content-based"
		cursor = mysql.connection.cursor()
		cursor.execute("SELECT COUNT(*) FROM "+ dbTable)
		row = cursor.fetchone()
		samePost.append(row[0])	
		cursor.close()


		if samePost[0]==0:
			params = {'token': '5fK2Zh50fDVXJxsCp7KZzYejopzLgCu', 'userId': userId, 'postIds':0, 'score':0}
			response = requests.post(tritonurl2+serReqPath, params=params)
			print (response.text)

			cursor = mysql.connection.cursor()
			cursor.execute("INSERT INTO "+ dbTable+"(userId, postId, content, translation) VALUES (%s, %s, %s, %s)", (userId, postId, text, translation.text))
			mysql.connection.commit()
			cursor.close()  


			return "Success"	

		else:
			for i in range(0, samePost[0]):
				cursor = mysql.connection.cursor()
				cursor.execute("SELECT translation FROM "+ dbTable)
				row = cursor.fetchall()
				row=[j[0] for j in row]
				texts.append(row[i])	
				cursor.close()
			print(texts)
			postList={}
			j=0
			texts=list(texts)
			cursor = mysql.connection.cursor()
			cursor.execute("SELECT postId FROM "+ dbTable)
			row = cursor.fetchall()
			row=[i[0] for i in row]
			for x in row:

		 		#postList.append(x)
		 		postList[j]=x
		 		#print(postList[j])
		 		j=j+1
			cursor.close()

			for i in (postList.keys()) :  
				print(postList[i]) 

			texts = [jieba.lcut(text) for text in texts]

			dictionary = corpora.Dictionary(texts)

			feature_cnt = len(dictionary.token2id)

			corpus = [dictionary.doc2bow(text) for text in texts]
			tfidf = models.TfidfModel(corpus) 
			kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
			index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
			sim = index[tfidf[kw_vector]]


			scores={}
			f=0
			for i in range(len(sim)):
				scores[f] = sim[i]
		#		print(scores[f])
				f=f+1

			    #print('keyword is similar to text%d: %.2f' % (i + 1, sim[i]))

			
		#	a=dict((scores[key], value) for (key, value) in postList.items())

			

			list1 = []
			list2 = []
			for value in scores.values():
				list1.append(value)
		#	print(list1)

			for value in postList.values():
				list2.append(value)
		#	print(list2)

			d = dict(zip(list2, list1))
		#	d=sorted(dictionary.items(),reverse=True)
			d1 = {k: v for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)}
			
			slist1=[]
			slist2=[]
			slist1.append(int(postId))
			slist2.append(1.0)
			for key, value in d1.items():
				slist1.append(key)
				slist2.append(value)
			print(slist1)
			print(slist2)




			tt =  ",".join( repr(slist1) for slist1 in slist1 )		
			jj =   ",".join( repr(slist2) for slist2 in slist2 )	

			params = {'token': '5fK2Zh50fDVXJxsCp7KZzYejopzLgCu', 'userId': userId, 'postIds':tt , 'score':jj}

			response = requests.post(tritonurl2+serReqPath, params=params)
			print (response.text)

			cursor = mysql.connection.cursor()
			cursor.execute("INSERT INTO "+ dbTable+"(userId, postId, content, translation) VALUES (%s, %s, %s, %s)", (userId, postId, text, translation.text))
			mysql.connection.commit()
			cursor.close()  


			return "Success"	


	else:
		return "Unsuccessfull - Token is wrong"

@app.route('/user-details/face-detect', methods=['POST'])
def faceDetection():


	getToken=request.args['token']

	if getToken == "5fK2Zh50fDVXJxsCp7KZzYejopzLgCu":
	
#		postId=request.args['postId']
		imagePath=request.args['imagePath']


		tritonurl1 = 'http://dcc96bc9.ngrok.io'
		serImgPath="/api/images/download/"



		def extract_face(filename, required_size=(160, 160)):
			image = Image.open(filename)
			image = image.convert('RGB')
			pixels = asarray(image)
			detector = MTCNN()
			results = detector.detect_faces(pixels)
			list1=[]
			for result in results:
				list1.append(result['box'])
			return list1


		'''
		def extract_face1(filename, required_size=(160, 160)):
			image = Image.open(filename)
			image = image.convert('RGB')
			pixels = asarray(image)
			detector = MTCNN()
			results = detector.detect_faces(pixels)
			i=0
			j=[]
			for result in results:	
				x1, y1, width, height = result['box']
				x1, y1 = abs(x1), abs(y1)
				x2, y2 = x1 + width, y1 + height
				face = pixels[y1:y2, x1:x2]
				image = Image.fromarray(face)
				image = image.resize(required_size)
				face_array = asarray(image)
				j.append(face_array)
				i+=1
			return j
		'''

		a=0
		urllib.request.urlretrieve(tritonurl1+serImgPath+imagePath,'Desktop/triton/draft/img/'+imagePath)


		'''
		f=extract_face1('Desktop/triton/draft/img/'+imagePath)
		for t in range(0,len(f)):
			pixels = f[t]
			pixels = Image.fromarray(pixels)
			pixels.save('Desktop/triton/draft/img/ttt'+str(a)+'.jpg')
			a+=1
		
		'''
		



		folder = 'Desktop/triton/draft/img/'+imagePath
		face = extract_face(folder)
		a=",".join( repr(face) for face in face )	
		print(a)

		
		params = {'token': '5fK2Zh50fDVXJxsCp7KZzYejopzLgCu', 'faces': a}

		response=requests.post('http://127.0.0.1:9000/',params=params)
		print(response)
		
		return "Success"

	else:
		return "Unsuccessfull"

@app.route('/profile-image', methods=['POST'])
def faceIdentification():

	getToken=request.args['token']
	userId=request.args['userId']
	imagePath=request.args['imagePath']


	tritonurl1 = 'http://a95ca73b.ngrok.io'
	serImgPath="/api/images/download/"
	pathprofile = "Desktop/triton/imageprofile/"
	pathextracted = "Desktop/triton/imageextracted/"
	pathimage="Desktop/triton/image/"
	dbTable="sn"

	def extract_face1(filename, required_size=(160, 160)):
		image = Image.open(filename)
		image = image.convert('RGB')
		pixels = asarray(image)
		detector = MTCNN()
		results = detector.detect_faces(pixels)
		i=0
		j=[]
		for result in results:	
			x1, y1, width, height = result['box']
			x1, y1 = abs(x1), abs(y1)
			x2, y2 = x1 + width, y1 + height
			face = pixels[y1:y2, x1:x2]
			image = Image.fromarray(face)
			image = image.resize(required_size)
			face_array = asarray(image)
			j.append(face_array)
			i+=1
		return j

	urllib.request.urlretrieve(tritonurl1+serImgPath+imagePath,pathprofile+imagePath)
	
	postimages = [image for image in os.listdir(pathimage) ]
	rcmndPost = []

	imageName=[]
	imageScore=[]

	for f in postimages:
		
		a=0

		extrProfileImage=extract_face1(pathprofile+imagePath)
		for t in range(0,len(extrProfileImage)):
			pixels = extrProfileImage[t]
			pixels = Image.fromarray(pixels)
			pixels.save(pathextracted+'img'+str(a)+'.jpg')
			a+=1
		
		extrPostImage=extract_face1(pathimage+f)
		for t in range(0,len(extrPostImage)):
			pixels = extrPostImage[t]
			pixels = Image.fromarray(pixels)
			pixels.save(pathextracted+'img'+str(a)+'.jpg')
			a+=1 
	

#////////////////**********/////////////////////************* TRITON Recommendation Engine ***********/////////////////************//////////////////////
		
		K.clear_session()
		importedImages = []

		vgg_model = vgg16.VGG16(weights='imagenet')
		feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
		feat_extractor.summary()
		

		extractedFiles=[image for image in os.listdir(pathextracted) ]
		
		i=0
		
		for img in extractedFiles:
			filename = img
			original = load_img(pathextracted+filename, target_size=(224, 224))
			numpy_image = img_to_array(original)
			image_batch = np.expand_dims(numpy_image, axis=0)
			importedImages.append(image_batch)
			i+=1
			
		images = np.vstack(importedImages)

		processed_imgs = preprocess_input(images.copy())
			
		imgs_features = feat_extractor.predict(processed_imgs)

		print("features successfully extracted!")
		
		cosSimilarities = cosine_similarity(imgs_features)

		cos_similarities_df = pd.DataFrame(cosSimilarities, columns=extractedFiles, index=extractedFiles)
		
		print(cos_similarities_df)
		
		closest_imgs = cos_similarities_df['img0.jpg'].sort_values(ascending=False)[1:i].index
		
		score = cos_similarities_df['img0.jpg'].sort_values(ascending=False)[1:i]

		
		
			
		for name in range(0,len(closest_imgs)):
			imageName.append(closest_imgs[name])
			imageScore.append(score[name])




		print(imageName)
		print(imageScore)


	

#		shutil.rmtree(pathprofile)
	 	
		 
		 
		 
		shutil.rmtree("Desktop/triton/imageextracted")
#		os.makedirs(pathprofile)
		os.mkdir("Desktop/triton/imageextracted") 


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	shutil.rmtree("Desktop/triton/imageprofile")
	
	os.mkdir("Desktop/triton/imageprofile")

	d = dict(zip(imageName, imageScore))
	d = dict((k, v) for k, v in d.items() if v >= 0.75)

	print(d)
		

	if any(d)==True:
		cursor = mysql.connection.cursor()
		cursor.execute("SELECT postId FROM "+dbTable+" WHERE imagePath LIKE %s", [f])
		row = cursor.fetchone()
		rcmndPost.append(row[0])	
		cursor.close()


	postList =  ",".join( repr(post) for post in rcmndPost )	

	postList = str(postList)
	
	
	K.clear_session()



	params = {'token': '5fK2Zh50fDVXJxsCp7KZzYejopzLgCu', 'userId':userId, 'postIds': postList}

	response=requests.post(tritonurl1+"api/posts/recommended/face-based",params=params)
	print(response)
	
	return "Success"




@app.route('/friend-suggest', methods=['POST'])
def friendsuggestion():
	
	tritonurl2=""
	serReqPath=""


	dbTable1="frtaste"
	db="tritonre"
	getToken=request.args['token']
	userId=request.args['userId']
	age=request.args['age']
	sex=request.args['sex']
	belief=request.args['belief']
	interest=request.args['interest']
	user="root"
	pass1 = ''
	host="localhost"
	port=3306

	url = 'mysql://{0}:{1}@{2}:{3}/{4}'.format(user, pass1, host, port,db)

	samePost=[]

	cursor = mysql.connection.cursor()
	cursor.execute("SELECT COUNT(*) FROM "+ dbTable1+ " WHERE userId = %s", [userId])
	row = cursor.fetchone()
	samePost.append(row[0])	
	cursor.close()


	if samePost[0]==0:
		cursor = mysql.connection.cursor()
		cursor.execute("INSERT INTO "+dbTable1+"(userId, age, sex, belief, interest) VALUES (%s, %s, %s, %s, %s)", (userId, age, sex, belief, interest))
		mysql.connection.commit()
		cursor.close()

	
	cursor = mysql.connection.cursor()
	cursor.execute("SELECT userId FROM "+dbTable1)
	listuser = [item[0] for item in cursor.fetchall()]
	cursor.close()
	
	print(listuser)

	SQL_Query = pd.read_sql_query("SELECT  age, sex, belief, interest FROM "+dbTable1, url)
	df = pd.DataFrame(SQL_Query,   columns=['age','sex','belief','interest'])
	df.index=listuser
	print (df)
	
	data=cosine_similarity(df)
	

	cos_similarities_df = pd.DataFrame(data, columns=listuser, index=listuser)
		
	print(cos_similarities_df)


#	closest_friends = cos_similarities_df[userId].sort_values(ascending=False).index
#	score = cos_similarities_df[userId].sort_values(ascending=False)

	closest_friends = list(cos_similarities_df[int(userId)].index)
	score = cos_similarities_df[int(userId)].values.tolist()
	
	print(closest_friends)
	print(score)
	
	d = dict(zip(closest_friends, score))
		
	d1 = {k: v for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)}
			
	slist1=[]
	slist2=[]
	
	for key, value in d1.items():
		slist1.append(key)
		slist2.append(value)
	#print(slist1)
	#print(slist2)




	tt =  ",".join( repr(slist1) for slist1 in slist1 )		
	jj =   ",".join( repr(slist2) for slist2 in slist2 )	

	'''
	params = {'token': '5fK2Zh50fDVXJxsCp7KZzYejopzLgCu','userId':userId 'userIds': jj}

	response = requests.post(tritonurl2+serReqPath, params=params)
	print (response.text)
	'''
	print(tt)
	print(jj)
	return tt

if __name__ == "__main__":
	app.run(port = port, debug=True)
	
	
