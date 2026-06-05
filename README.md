Ensure you have python 3.10 or 3.11 installed

## Create a virtual environment:

python -m venv venv  
.\venv\Scripts\activate

## Install libraries:

pip install -r requirements.txt

## Adding documents:

Create a "Documents" folder within the root directory and add pdf files here.

## Running the program:

python ingest_documents.py  
python query.py "\<insert your query here\>"

## API endpoint:

python retrieve_api.py "\<insert your query here\>"
