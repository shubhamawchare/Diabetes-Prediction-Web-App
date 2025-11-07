# imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cleaning up our data
df = pd.read_csv("diabetes.csv")
df_temp = df.drop(columns=['Outcome','Pregnancies','Insulin','SkinThickness'])
df_temp = df_temp.replace(0,np.nan)
df = pd.concat([df['Pregnancies'],df['Insulin'],df['SkinThickness'],df_temp,df['Outcome']],axis=1)
df = df.dropna().reset_index(drop=True)

df = df.rename(columns={'BloodPressure': 'Diastolic Blood Pressure','DiabetesPedigreeFunction': 'Diabetes Pedigree Function'})

# Setting up the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualizations", "ML Model 1", "ML Model 2"])

# Homepage
if page == "Home":
    st.title("Welcome to the Diabetes App!")
    st.write("Use the sidebar to navigate to different sections.")
    st.write("Here's a quick look at the data:")
    st.dataframe(df.head())
    st.dataframe(df.describe())

    with st.expander("Read more about this dataset!"):
        st.write("""This is data collected from a group of Pima Indian women, aged 21 and above. 
        Zeros indicate missing values, except in the case of outcome, where they indicate a non-diabetic person. 
        There are 768 rows of data, and 8 features, not including Outcome.""")

# Data Viz Page
elif page == "Data Visualizations":
    st.title("Let's take a look, shall we?")
    st.write("Choose a figure you're curious about!")

    # Plots we already created!
    if st.button("Display Age Histogram"):
        st.title("Age Histogram")
        fig, ax = plt.subplots()
        ax.hist(df["Age"], bins=20)
        st.pyplot(fig)
    if st.button("Display Glucose vs. BMI Scatterplot"):
        st.title("Glucose vs BMI")
        fig, ax = plt.subplots()
        ax.scatter(df["Glucose"], df["BMI"])
        st.pyplot(fig)
    if st.button("Display Correlation Heatmap"):
        corr = df.corr()
        st.title("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data=corr, ax=ax,cmap='coolwarm', annot=True)
        st.pyplot(fig)

        # Correlation Heatmap Explanation
        with st.expander("What is this plot showing?"):
            st.write("""This is a correlation heatmap. Warmer colors indicate positive correlations and cooler colors indicate negative correlations. 
            The closer the absolute value of the correlation is to 1, the more strongly correlated.""")
        
    # Buttons for distribution by outomce
    st.write("Choose a characteristic to see its distribution by outcome.")
    for feature in df.columns:
        if feature == 'Outcome':
            continue
        if st.button(feature):
            fig, ax = plt.subplots()
            sns.kdeplot(data = df[df['Outcome']==1], x = feature, label = 'Diabetic', ax=ax);
            sns.kdeplot(data = df[df['Outcome']==0], x = feature, label = 'Non-diabetic',ax=ax);
            ax.set_title(f'{feature} Distribution by Outcome')
            ax.legend()
            st.pyplot(fig)
        
    
# Logistic Regression Page
elif page == "ML Model 1":
    if 'scaler' not in st.session_state:
        scaler = None
    # imports
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    st.title("Logistic Regression Model")
    # Explain the model
    with st.expander("How does this model work?"):
            st.write("""This is a logistic regression model, often used for classification tasks, such as 'spam' vs. 'not spam' or our scenario, 
            'diabetic' vs. 'non-diabetic'. It multiplies each value for each feature in the information passed to it and sums all of them;  
            it then passes this sum through a sigmoid function, which squeezes this to ensure all values are between 0 and 1.  For larger sums, 
            the output is closer to one, and smaller sums produce an output closer to 0. If the final output is over a certain threshold, usually ~0.5, 
            the model returns '1.' Otherwise, it returns '0.'
            """)

    # Choosing features
    features = st.multiselect("Choose features:", df.columns[:-1], default=["Glucose", "BMI"])
    target = "Outcome"
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

    # Training the model
    if st.button("Train model"):
        X = df[features]
        y = df[target]

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # Doing the scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test,y_pred)

        st.success(f"Accuracy: {acc:.2f}")
        st.write("Confusion matrix:")
        st.write(cm)

        with st.expander("What is this showing?"):
            st.write("""This is a confusion matrix, a way to get a quick look at how our model performed on the testing set. The top indicates the 
            actual correct answer - 0 or 1. The first row includes all the model's false predictions; for example, the top left square indicates how 
            many false cases were assigned 'false' by the model. Similarly, the botton row indicates all the model's true predictions, so the bottom 
            left square indicates how many negative cases were assigned 'positive' by the model.""")

       
        st.write("Classification Report:")
        st.text(report)  
        with st.expander('What is this showing?'):
            st.write("""The first number, 0.81, tells us the portion of 0 predictions that actually were 0. The second number, 0.80, tells us the
            portion of true 0s, or negative outcomes, that the model predicted to be 0. Similarly, in the bottom row, the precision is the portion 
            of 1 predictions that actually were 1. Essentially, if the model predicted 1, there’s a 65% chance that you actually do have diabetes. 
            On the other hand, the recall is telling us that 67% of the true diabetics in the test set were actually caught by the 
            model - that is, 0.67 is the portion of true 1s that the model predicted to be 1.""") 

        #Saving the model
        st.session_state['model'] = model
        st.session_state['features'] = features
        st.session_state['scaler'] = scaler

    # Show form for prediction if model exists
    if 'model' in st.session_state:
        st.subheader("Make a Prediction")

        #Reload the model
        model = st.session_state['model']
        features = st.session_state['features']
        scaler = st.session_state['scaler']
        # Allow users to enter data
        with st.form("prediction_form"):
            feature_dict = {}
            for feature in features:
                feature_dict[feature] = st.number_input(f"Enter your {feature}:", value=0.0)
            submitted = st.form_submit_button("Predict!")
            if submitted:
                input_df = pd.DataFrame([feature_dict])
                prediction = model.predict(scaler.transform(input_df))
                st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
                

elif page == "ML Model 2":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    st.title("Decision Tree Classifier")
    with st.expander("How does this model work?"):
            st.write("""A decision tree creates a series of splits, or branches, using layers of condtions such as 'Age < 30' and 'Age > 30' to classify data. When making a prediction, it simply follows the path of the tree, given a certain piece of data, and outputs the result of the last branch it reaches, which in our case is either a 0 or a 1.""")
            
    # Allow user to select features
    features = st.multiselect("Choose features:", df.columns[:-1], default=["Glucose", "BMI"])
    target = "Outcome"
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

    # Train model
    if st.button("Train model"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        model1 = DecisionTreeClassifier()
        model1.fit(X_train, y_train)

        preds = model1.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        y_pred = model1.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test,y_pred)

        st.success(f"Accuracy: {acc:.2f}")
        st.write("Confusion matrix:")
        st.write(cm)

        with st.expander("What is this showing?"):
            st.write("""This is a confusion matrix, a way to get a quick look at how our model performed on the testing set. The top indicates the 
            actual correct answer - 0 or 1. The first row includes all the model's false predictions; for example, the top left square indicates how 
            many false cases were assigned 'false' by the model. Similarly, the botton row indicates all the model's true predictions, so the bottom 
            left square indicates how many negative cases were assigned 'positive' by the model.""")

       
        st.write("Classification Report:")
        st.text(report)  
        with st.expander('What is this showing?'):
            st.write("""The first number, 0.81, tells us the portion of 0 predictions that actually were 0. The second number, 0.80, tells us the
            portion of true 0s, or negative outcomes, that the model predicted to be 0. Similarly, in the bottom row, the precision is the portion 
            of 1 predictions that actually were 1. Essentially, if the model predicted 1, there’s a 65% chance that you actually do have diabetes. 
            On the other hand, the recall is telling us that 67% of the true diabetics in the test set were actually caught by the 
            model - that is, 0.67 is the portion of true 1s that the model predicted to be 1.""") 
        

        # Save model
        st.session_state['model1'] = model1
        st.session_state['features'] = features

    # Show form for prediction if model exists
    if 'model1' in st.session_state:
        st.subheader("Make a Prediction")

        # Reload model
        model = st.session_state['model1']
        features = st.session_state['features']

        # Allow user to input data
        with st.form("prediction_form"):
            feature_dict = {}
            for feature in features:
                feature_dict[feature] = st.number_input(f"Enter your {feature}:", value=0.0)
            submitted = st.form_submit_button("Predict!")

            # Make a prediction
            if submitted:
                input_df = pd.DataFrame([feature_dict])
                prediction = model.predict(input_df)
                st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")

