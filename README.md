## Project: Phishing Link Detection System
# Description:
Developed a machine learning-based system to detect phishing URLs with high accuracy, using a Kaggle dataset and built in Python and Django.

# Key Responsibilities:

● Collected and processed data from Kaggle's Phishing URL dataset, which included various features like URL length, domain age, hyperlink ratios, and other key indicators of phishing.

● Applied feature engineering to extract over 40 unique features, including domain-specific attributes and URL structure analysis.

● Trained and fine-tuned machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting, and SVM) to classify URLs as safe or phishing, using accuracy and precision as key metrics.

● Built and deployed a web application in Django with an admin interface for adding new data and monitoring model performance.

● Integrated external APIs to retrieve real-time domain information (e.g., domain age, Google index, page rank) to enhance model accuracy and reliability.

# Technologies Used:
Python, Scikit-Learn, Django, BeautifulSoup, REST APIs, Pandas, Numpy, Kaggle dataset

# Project Outcome:
Achieved 84% accuracy on test data and deployed an interactive web-based detection system that flags potentially harmful URLs in real-time.

Note :
index 0: Train  data accuracy
index 1: Test data accuracy
'Logistic Regression': [0.7320283679505495, 0.7468385998385958], 
'Random Forest': [1.0, 0.8355034220149401], 
'Gradient Boosting': [0.992000846804494, 0.8763359059119408], 
'SVM': [0.8990106909067369, 0.8471698459855117], 
'KNN': [0.8615146603028027, 0.8110039316767397], 
'XGBClassifier': [0.9950005292528088, 0.8763359059119408]

