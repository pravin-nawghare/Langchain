from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language

text = '''
# Telecom Customer Churn Prediction – End-to-End ML Pipeline
An **end-to-end machine** learning project to predict customer churn for a telecom company. The pipeline connects to a MongoDB database for data ingestion, preprocesses the data, trains classification models focused on minimizing false positives, and deploys the best model in AWS. A FastAPI-based web app provides a user interface, and GitHub Actions powers the CI/CD pipeline.
    
## Pipeline Details
### Data Collection
Customer usage and subscription data is pulled from MongoDB.
### Preprocessing
Cleans missing and inconsistent values.
Applies encoding, normalization, and feature engineering.
Splits data into train/test sets using stratified sampling.
### Model Training
Multiple classifiers tested (e.g., Logistic Regression, Random Forest).
Hyperparameter tuning applied as needed.
### Model Evaluation
Primary Metric: Recall (due to class imbalance).
Goal: Reduce false positives to avoid misclassifying loyal customers.
Best-performing model is selected and saved.
### Model Deployment
Final model is serialized and uploaded to AWS S3.
Ready for inference via API or frontend.

## CI/CD with GitHub Actions
Automated workflows handle:
Code linting & style checks
Unit tests
Model training and evaluation
Deployment to cloud (AWS)

## Frontend (FastAPI)
A simple FastAPI web interface allows telecom customer data to be submitted for real-time churn prediction.
To run locally:
uvicorn webapp.main:app --reload

## Getting Started
- Clone the Repo
- Install Dependencies
- Start the Web App

### Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.
### License
This project is licensed under the MIT License.
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 3,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)

# output - see the image at path (TextSplitter\DocumentStructureBased\Recursive text splitter on markdown text with 3 chunks.png)
# and (TextSplitter\DocumentStructureBased\Recursive text splitter on markdown text with 6 chunks.png)