<a id="readme-top"></a>
<!--
*** ReadMe template from Othneil Drew @othneildrew on GitHub
-->
<!-- [![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/alvagarciia/ml-dl-carprices">
    <img src="media/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Car Price Predictor</h3>

  <p align="center">
    Car market price predictor web app based on ML models
    <br />
    <br />
    <a href="https://github.com/alvagarciia/ml-dl-carprices/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/alvagarciia/ml-dl-carprices/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Car Prices Showcase ML](./media/carprice-ML-demo.gif)
![Car Prices Showcase DL](./media/carprice-DL-demo.gif)

This web application predicts the market price of used cars based on various features such as mileage, brand, model, fuel type, and more. Built using Python and Streamlit, the app leverages machine learning models to provide pricing estimates, assisting both buyers and sellers in making informed decisions.

My personal objective with this project was to put in practice everything I learnt for working with Machine Learning and Deep Learning models by using it to create a piece of software. In addition, I aimed to experiment creating this app using Streamlit, which I hadn't used prior to this. 

Both models were trained from scratch using a Kaggle [dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data) uploaded by Taeef Najib, made up of a collection of automotive information extracted from the popular automotive marketplace website, https://www.cars.com. This dataset comprises 4,009 data points, each representing a unique vehicle listing, and includes nine distinct features providing valuable insights into the world of automobiles.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Features

- User input form for entering car specifications
- Pre-trained ML and DL models for price prediction
- Built-in preprocessing (encoding, imputation, etc.)
- Works from serialized models using `.joblib` and `.keras` files

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

[![Pandas][pandas]](https://github.com/pandas-dev/pandas)
[![NumPy][numpy]](https://github.com/numpy/numpy)
[![Scikit-Learn][scikit-learn]](https://github.com/scikit-learn/scikit-learn)
[![Tensorflow][tensorflow]](https://github.com/tensorflow/tensorflow)
[![Streamlit][streamlit]](https://github.com/streamlit/streamlit)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

* Python 3.10+
  ```sh
  pyenv install python-3.12.5
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/alvagarciia/ml-dl-carprices.git
   cd ml-dl-carprices
   ```
2. Set up a virtual environment
   Using `venv` (comes with Python):
   ```sh
   python -m venv venv
   ```
   Or using `conda` (if preferred):
   ```sh
   conda create -n car-price python=3.10
   conda activate car-price
   ```
3. Activate environment
   **Using `venv`:**
   ```sh
   # Windows
    venv\Scripts\activate

    # Linux / macOS
    source venv/bin/activate
   ```
4. Install dependencies
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   **Standard usage:**
    ```bash
   streamlit run app.py
   ```

    **If you're using WSL:**
    ```bash
   streamlit run app.py --server.headless true --server.enableCORS false --server.address=0.0.0.0
   ```
   This allows you to access the app from your Windows browser when running Streamlit inside WSL.
   
6. Open the url provided in terminal, usually `localhost:8501`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Once ran on Streamlit, users have the option of entering their own car data or using a pre-defined test case. For each option, they also have the option of getting a prediction using the Machine Learning model or the Deep Learning model. 

After selecting either of the options, the respective model should display the prediction on the website.

The repo also contains the scripts used to train each model inside the `scripts` folder, so that anyone can take a look at how these models were trained. Moreover, the `models` folder contains the trained files for both the ML and DL models.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alvaro Garcia - [LinkedIn @alvaro-garciam](https://www.linkedin.com/in/alvaro-garciam) - alvaroedgamu@gmail.com

Project Link: [https://github.com/alvagarciia/ml-dl-carprices](https://github.com/alvagarciia/ml-dl-carprices)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- 
https://www.markdownguide.org/basic-syntax/#reference-style-links 
https://simpleicons.org/
-->
[license-s
hield]: https://img.shields.io/github/license/alvagarciia/ml-dl-carprices.svg?style=for-the-badge
[license-url]: https://github.com/alvagarciia/ml-dl-carprices/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/alvaro-garciam


[pandas]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[numpy]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[scikit-learn]: https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white
[tensorflow]: https://img.shields.io/badge/Tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[streamlit]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white