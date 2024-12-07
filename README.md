# COMP5130_FINAL
<!-- Project Title -->
<br />
<div align="center">
  <h3 align="center">Adversarial Attack on Graph Structured Data</h3>

  <p align="center">
    Runs various tests related to this paper. Random Sampling with number of nodes to attack, 
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project


* Getting started
  This application can be run by installing the packages with the following.
  ```sh
  pip install -r requirements.txt
  ```

  Then go to the Experiment_Code directory to run the program, run:
  ```sh
  python3 main.py
  ```

* Functionality explained
  This program has the ability to run 4 different programs depending on a user inputting 1, 2, 3, or 4.
  * (1) Will allow the user to input an integer greater than 0 and then the program will run the random sampling attack on the citeseer database and attack that many nodes.
  * (2) Allows the user to play sudoku with a board that comes with
  * (3) Will run a test evaluating runtime on 49151 boards using arc consistency and forward checking with corresponding histograms.
  * (4) Runs the MCV test suite and displays resulting graphs
