**Install dependencies:**

pip install -r requirements.txt

**Project Structure**

project/

│

├── recommender.py                                  (Core logic for data loading, model creation, and recommendation)

├── train.ipynb                                     (Model training notebook (saves trained weights))

├── main.py                                         (Flask backend: web app routes, request handling, connects everything)

├── requirements.txt                                (Python dependencies)

│

├── templates/                                     (Contains HTML templates)

│   └── index.html

│

└── static/                                            (Contains static assets like CSS)

    └── style.css
