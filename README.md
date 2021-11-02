# Egg_SummingGame
Citation for core folder


@misc{kharitonov:etal:2021,
  author = "Kharitonov, Eugene  and Dess{\`i}, Roberto and Chaabouni, Rahma  and Bouchacourt, Diane  and Baroni, Marco",
  title = "{EGG}: a toolkit for research on {E}mergence of lan{G}uage in {G}ames",
  howpublished = {\url{https://github.com/facebookresearch/EGG}},
  year = {2021}
}

Contents of notebooks folder:
 - data generation script used to sample and create train and test sets
 - performance analysis script used to analyse loss and accuracy for agents with size (N,v)
 - symbol analysis script used to analyse sender message and receiver output 

data_reader.py - Includes implementation of data reader that creates feature matrix and target

architecture.py - includes implementation of the architecture of the Sender and Receiver agents

play.py - end-to-end implementation of single-symbol game
