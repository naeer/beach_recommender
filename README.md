# Beach Recommender System


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
