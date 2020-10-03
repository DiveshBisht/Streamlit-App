import streamlit as st
import cv2
import numpy as np 
from keras.models import load_model 
import os
from PIL import Image
import emoji
import joblib
import dill
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity



def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext

def preprocessText(title,description): 
    code = str(re.findall(r'<code>(.*?)</code>', description, flags=re.DOTALL))
    description=re.sub('<code>(.*?)</code>', '', description, flags=re.MULTILINE|re.DOTALL)
    description=striphtml(description.encode('utf-8'))
    title=title.encode('utf-8')
 
    #Adding title three time to the data to increase its weightage
    question=str(title)+" "+str(title)+" "+str(title)+" "+description
    question=re.sub(r'[^A-Za-z0-9#+.\-]+',' ',question)
    words=word_tokenize(str(question.lower()))
    
    #Removing all single letter and stopwords from question except for the letter 'c'
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
    return tuple([question])

def recommend_movies(predictions_df, userID, movies_df, ratings_df):
    
    #get and sort the user's predictions
    user_row_number = userID - 1 
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    #get the user's data and merge in the movie information
    user_data = ratings_df[ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False))
    
    #recommend the 5 highest rated predicted movies that the user hasn't seen yet
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',left_on = 'MovieID',right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).iloc[:5, :-1])

    return user_full, recommendations

def main():
    """ Streamlit App for Data Science"""
    st.info("Data Science Projects by Divesh Bisht")

    topics = ["Machine Learning", "Deep Learning", "Transfer Learning", "Recommender System"]
    choice = st.sidebar.selectbox("Choose Topic",topics)

    if choice=="Machine Learning":

        html_temp = """
        <div style="background-color:teal; padding:10px">
        <h1 style="color:white; text-align:center;">Stack Overflow Tag Predictor</h1>
        </div>
        <br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        ques_vectorizer = dill.load(open("ques_vectorizer.pkl", "rb"))
        tag_vectorizer = dill.load(open("label_vectorizer.pkl", "rb"))

        all_ml_models = ["LR","SVM"]
        model_choice = st.selectbox("Choose ML Model", all_ml_models)
        
        title_text = st.text_area("Enter Title of the Question", "Type here")
        desc_text = st.text_area("Enter Description of the Question", "Type here")

        if st.button("Analyze"):
            processed = preprocessText(title_text,desc_text)
            vectorized_ques = ques_vectorizer.transform(processed)

            if model_choice=='LR':
                LR_classifier = joblib.load("model/LR_model.pkl")
                prediction = LR_classifier.predict(vectorized_ques)
                
            if model_choice=='SVM':
                SVM_classifier = joblib.load("model/SVM_model.pkl")
                prediction = SVM_classifier.predict(vectorized_ques)

            st.text_area("Processed Text",processed[0])

            tag = tag_vectorizer.inverse_transform(prediction)
            label=""
            if len(tag[0])>1:
                for t in tag[0]:
                    label+=t+"  "
            else:
                label= tag[0][0]
            st.success("Predicted Tag  :  {}".format(label))



    elif choice=="Deep Learning":
        
        html_temp = """
        <div style="background-color:salmon; padding:10px">
        <h1 style="color:white; text-align:center;">Covid-19 Mask Detector</h1>
        </div>
        <br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        model = load_model('model/mask_model')
        face_clf=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        st.set_option('deprecation.showfileUploaderEncoding', False)
        
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image, height=400, width=400)

        if st.button("Analyze"):
            labels_dict={0:"MASK",1:"NO MASK"}
            color_dict={0:(0,255,0),1:(255,0,0)}
            img = np.array(our_image)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_clf.detectMultiScale(gray,1.3,5) 

            for (x,y,w,h) in faces:
                face_img=gray[y:y+w,x:x+h]
                resized=cv2.resize(face_img,(100,100))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,100,100,1))
            
                result=model.predict(reshaped)
                label=np.argmax(result,axis=1)[0]
      
                cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],4)
                cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            st.text("Processed Image")
            st.image(img, height=400, width=400)

            if label==0:
                st.success("Mask Detected : You are a Covid Warrior")
            else:
                st.success("No Mask Detected : You are a Covid Defaulter")



    elif choice=="Transfer Learning":
        
        html_temp = """
        <div style="background-color:goldenrod; padding:10px">
        <h1 style="color:white; text-align:center;">Facial Emotion Detector</h1>
        </div>
        <br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        model = load_model('model/fine_tuned_model.h5')
        face_clf=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        st.set_option('deprecation.showfileUploaderEncoding', False)

        if image_file is not None:
            my_image = Image.open(image_file)
            st.text("Original Image")
            st.image(my_image, height=400, width=400)

        if st.button("Analyze"):
            img=np.array(my_image)
            gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_clf.detectMultiScale(gray_img, 1.32, 5)

            for (x,y,w,h) in faces_detected:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=4)
                cv2.rectangle(img,(x,y-40),(x+w,y),(0,0,255),-1)
                face_img=gray_img[y:y+w,x:x+h]
                face_img=cv2.resize(face_img,(48,48))
                face_img=face_img/255.0
                face_img=face_img.reshape(1,48, 48)
                face_img=np.repeat(face_img[..., np.newaxis], 3, -1)

                prediction = model.predict(face_img)
                max_index = np.argmax(prediction[0])
                emotions = ('ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL')
                emojis = (':angry:', ':unamused:', ':fearful:', ':smile:', ':pensive:', ':hushed:', ':neutral_face:')
                predicted_emotion = emotions[max_index]
                custom_emoji = emojis[max_index]

                cv2.putText(img, predicted_emotion, (int(x),int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            st.text("Processed Image")
            st.image(img, height=400, width=400)

            st.success("Emotion Detected : " + str(predicted_emotion.title()) + "  " + emoji.emojize(custom_emoji,use_aliases=True))



    elif choice=="Recommender System":
       
        html_temp = """
        <div style="background-color:darkgreen; padding:10px">
        <h1 style="color:white; text-align:center;">Movie Recommendation System</h1>
        </div>
        <br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        movies_df = pd.read_csv('movies.csv')
        movies_df.columns = ['MovieID', 'Title', 'Genres']
        movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
        ratings_df = pd.read_csv('ratings.csv')
        ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        
        R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
        R = R_df.to_numpy()
        user_ratings_mean = np.mean(R, axis = 1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(R_demeaned, k = 100)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

        techniques = ["Colaborative Filtering (Matrix Factorisation)","Movie-Movie Similarity"]
        technique = st.selectbox("Choose Technique", techniques)

        if technique=='Colaborative Filtering (Matrix Factorisation)':
                userID = int(st.text_input("Enter User ID", 300))

                if st.button("Analyze"):
                    already_rated, predicted = recommend_movies(preds_df, userID, movies_df, ratings_df)
                    already_rated = already_rated.drop(['UserID', 'MovieID', 'Rating', 'Timestamp'], axis = 1).reset_index(drop=True).head()
                    predicted = predicted.drop(['MovieID'], axis = 1).reset_index(drop=True)
                    predicted.index +=1
                    already_rated.index +=1


                    html = '''<style>
                            table {  border-collapse: collapse; width: 100%;}
                            th, td { text-align: center; padding: 5px; }
                            th { background-color:#33FF6E; color: black; }
                            </style>'''
                                                

                    st.text("Top 5 Watched Movies")
                    already_html = html +  already_rated.to_html()
                    st.markdown(already_html, unsafe_allow_html=True)

                    st.markdown("<br>",unsafe_allow_html=True)

                    st.text("Top 5 Recommended Movies")
                    predicted_html = html + predicted.to_html()
                    st.markdown(predicted_html, unsafe_allow_html=True)

                
        elif technique=='Movie-Movie Similarity':
                movieID = int(st.text_input("Enter Movie ID", 364))

                if st.button("Analyze"):
                    sparse_matrix = sparse.csr_matrix((ratings_df.Rating.values, (ratings_df.UserID.values,ratings_df.MovieID.values)))
                    m_m_sim_sparse = sparse.load_npz("m_m_sim_sparse.npz")
                    movie_df = movies_df.set_index('MovieID')

                    movie = str(movie_df.loc[movieID].values[0])
                    total_ratings = sparse_matrix[:,movieID].getnnz()
                    avg_rating = sum(sparse_matrix[:,movieID].toarray())/total_ratings

                    st.markdown('Movie name :  **' + movie+'**')
                    st.markdown('Total ratings from users   :  **' + str(total_ratings) +'**')
                    st.markdown('Average rating out of 5     :  **' + str(round(avg_rating[0],1)) + '**')

                    st.markdown("<br>",unsafe_allow_html=True)
                    
                    similarities = m_m_sim_sparse[movieID].toarray().ravel()
                    sim_indices = similarities.argsort()[::-1][1:]
                    similar_df = movies_df.set_index('MovieID')
                    top10_df = similar_df.loc[sim_indices[:10]]   
                    top10_df = top10_df.reset_index()
                    top10_df.index+=1

                    html = '''<style>
                            table {  border-collapse: collapse;}
                            th, td { text-align: center; padding: 5px; }
                            th { background-color:#52FF33; color: black; }
                            </style>'''
                                                

                    st.text("Top 10 Recommended Movies")
                    top10_html = html +  top10_df.to_html()
                    st.markdown(top10_html, unsafe_allow_html=True)


if __name__=='__main__':
    main()