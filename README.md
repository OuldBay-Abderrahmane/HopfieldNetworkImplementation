
<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">Implementation of Hopfield Networks in collaboration with Richard Chiu and Margot Vogel 
</h3>

  <p align="center">
    This project is an implementation of Hopefield networks or recurrent artificial neural network. Hopefield networks serve as content-addressable ("associative") memory systems with binary threshold nodes, or with continuous variables. using differents models such as <a href="https://en.wikipedia.org/wiki/Hopfield_network#Hebbian_learning_rule_for_Hopfield_networks" >Hebbian</a> and <a href="https://en.wikipedia.org/wiki/Hopfield_network#Storkey_learning_rule">Storkey</a> models to be able study biological systems. 
  </p>
</div>

<!-- ABOUT THE PROJECT -->

<h3 align="center" style="font-size: 1.5rem;">About the project</h3>

<!-- ABOUT THE PROJECT -->

<p>The current state of the project is able to execute Hebbs and Storkey models to retrieve initial state of a matrix/pattern that we perturbed before hand and create a videoof it. We also created a full suite of tests to make our project more robust. We also made a visualization of the modelss execution with the perturbation of a checkboard to see the checkborad returning to it's original state.</p>

<div align="center">
  <h4 style="font-weight:10px">Synchronous updates on Hebbian model (Energy plot and video)</h4>
  <div> 
  <img src="scripts/plots/hebbian_sync_energy_classes.png" style="max-width:400px;" /> <!-- Hebbian energy synchronous -->
  </div>


https://user-images.githubusercontent.com/71345328/144859014-c67ed657-c909-4168-ad05-8bc3d9d64f73.mp4


</div>
<div align="center">
  <h4 style="font-weight:bolder">Asynchronous updates on Hebbian model (Energy plot and video)</h4>
  <div> 
  <img src="scripts/plots/hebbian_async_energy_classes.png" style="max-width:40%;" /> <!-- Hebbian energy synchronous -->
 </div>


https://user-images.githubusercontent.com/71345328/144859023-e80ccc49-f762-4fb4-82ac-70bad3416fc9.mp4


</div>  
<div align="center">
  <h4 style="font-weight:bolder">Synchronous updates on Storkey model (Energy plot and video)</h4>
  <div> 
  <img src="scripts/plots/str_sync_energy_classes.png" style="max-width:400px;" /> <!-- Storkey energy synchronous -->


https://user-images.githubusercontent.com/71345328/144859046-a859d346-5ed2-4766-bc65-c3df10d8b381.mp4


  </div>
</div>  

<div align="center">
  <h4 style="font-weight:bolder">Synchronous updates on Storkey model (Energy plot and video)</h4>
  <div> 
  <img src="scripts/plots/str_async_energy_classes.png" style="max-width:400px;" /> <!-- Storkey energy asynchronous -->
  </div>


https://user-images.githubusercontent.com/71345328/144860171-ecd582da-c0df-41a5-86b8-c597dc74e692.mp4



</div>  

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Python](https://www.python.org)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

You will need to first make sure pip is installed and upgraded
* pip
  ```sh
  python -m pip install --upgrade pip
  ```
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/EPFL-BIO-210/BIO-210-team-28.git
   ```
2. Install PIP packages
   ```sh
   pip install numpy matplotlib pandas tables ffmpeg scikit-learn skimage 
   ```
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Our project is still in it's early tsaged but it will soon be able too handle complex biological data. 

<p align="right">(<a href="#top">back to top</a>)</p>


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

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

>>>>>>> master
