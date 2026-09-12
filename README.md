Titanic Survival Prediction

This project involves analyzing passenger data from the Titanic and building a machine learning model that can predict if a passenger survived.

I began by looking at the dataset to learn about the passengers and find what factors might influence survival.
After making the data clean and ready, I used a Logistic Regression model to make predictions.I also made a Streamlit app so people can use the model through a web interface.

Dataset

The data comes from Kaggle's Titanic: Machine Learning from Disaster competition.

Dataset Link: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

The dataset has information about passengers like their age, gender, class, fare, family details, where they embarked, and whether they survived.

Exploratory Data Analysis

I started with data exploration to understand the dataset better and find what patterns are related to survival.

During this phase, I looked at:

* Structure of the dataset and some basic stats
* Missing values
* How many people survived
* How many were in each class
* Survival by class
* Gender distribution and survival
* Age and fare distributions
* Outliers using boxplots and the IQR method

I used Matplotlib and Seaborn to make visualizations.

Data Preprocessing

Before training the model, I cleaned and prepared the data.

The key steps I did were:

* Filled missing age values using the median
* Filled missing embarkation values using the most common value
* Removed the Cabin column as it had many missing values
* Changed the Sex column to numbers
* Used one-hot encoding for Embarked
* Removed unnecessary columns like PassengerId, Name, and Ticket

Model Training

For the prediction task, I used Logistic Regression because the target is a binary outcome—survived or not survived.
I split the dataset into training and testing data using an 80:20 ratio.
The model used features like class, gender, age, family information, fare, and embarkation port.

Model Evaluation

After training the model, I used several metrics to evaluate performance on the test data:

* Accuracy Score
* Confusion Matrix
* Classification Report
* ROC Curve
* ROC-AUC Score

These measurements helped me see how well the model was working and how accurate it was at predicting survival.

Streamlit Application

I also developed a Streamlit app to make it easier for people to use the model.

The app lets users enter details like:

* Class
* Gender
* Age
* Number of siblings or spouses
* Number of parents or children
* Fare
* Embarkation port

After entering the details and clicking Predict, the app shows what the model predicts for survival.

I also added a custom background image and some style to make the app look better.

Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

Project Structure

```
Titanic-Survival-Predication/
│
├── Titanic Survival Analysis and Prediction.py
├── app.py
├── Titanic Survival Prediction Project DataScience.pdf
├── bg.png
└── titanic_model.pkl
```

How to Run the Project

Clone the repository:

```bash
git clone https://github.com/shivam-kr11/Titanic-Survival-Predication.git
```

Go to the project directory:

```bash
cd Titanic-Survival-Predication
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser.
Enter passenger details and click Predict to get the result.

Make sure `titanic_model.pkl` and `bg.png` are in the project directory.

What I Learned

This project helped me understand the complete basic machine learning workflow, from data exploration and cleaning to model training and evaluation.

I also got hands-on experience with handling missing values, encoding categorical variables, detecting outliers, selecting useful features, and evaluating a classification model.

Building the Streamlit app helped me see how a trained model can be connected to a simple user interface.

Future Improvements

Some improvements I want to make in the future include comparing different machine learning algorithms, improving feature engineering, doing hyperparameter tuning, adding prediction probabilities, and deploying the application online.

Author

Shivam Kumar

GitHub: [shivam-kr11](https://github.com/shivam-kr11)
