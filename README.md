# Seagull Eye: Beach Recommender System

The aim of this project was to build an app that would allow beachgoers to find uncrowded serene beaches, especially during the hot summer days when thousands of people flock to the world-renowned Sydney beaches.

# Demo 

**Landing Page**

![landing](/data/landing_page.gif)

**Real Time Detection**

![real_time](/data/real_time.gif)

**Preloaded videos**

![preloaded](/data/preloaded_videos.gif)

## Installation and running the app
- Clone the repo:
  ```sh
  git clone https://github.com/naeer/beach_recommender.git
  ```

- Go to the root directory of the app
- Create a virtual environment named `venv`
  -  ```sh
     python -m venv venv
     ```
- Activate the virtual environment
  - For MacOS users
    ```sh
    source venv/bin/activate
    ```
  - For Windows users
    ```sh
    venv\Scripts\activate
    ```
- Run the following following commands (upgrade pip and install the requirements):
  ```sh
  pip3 install --upgrade pip
  pip3 install -r requirements.txt
  ```     
- Go to the directory: `mysite`
  ```sh
  cd mysite
  ```
- Run the following commands:
  ```sh
  python3 manage.py makemigrations
  python3 manage.py migrate
  ```
- Run the following command to start the app:
  ```sh
  python3 manage.py runserver
  ```
- Navigate to the following url in your web browser
  ```sh
  http://127.0.0.1:8000/
  ```
