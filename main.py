"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
from evolver import Evolver
from tqdm import tqdm
import logging
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True


# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='/content/IA/My Drive/IA/carie_class/log_evolve_1.txt'
)

def train_genomes(genomes, config):
    """Train each genome.

    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(config)
        pbar.update(1)
    
    pbar.close()

def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.

    Args:
        networks (list): List of networks/genomes

    Returns:
        float: The average accuracy of a population of networks/genomes.

    """
    total_accuracy = 0

    for genome in genomes:
        total_accuracy += genome.accuracy

    return total_accuracy / len(genomes)

def generate(generations, population, all_possible_genes, config):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")
    evolver = Evolver(all_possible_genes)
    genomes = evolver.create_population(population)

    # Evolve the generation.
    for i in range( generations ):

        logging.info("***Now in generation %d of %d***" % (i, generations))

        print_genomes(genomes)
        
        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, config)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(genomes)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80) #-----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:5])

    #save_path = saver.save(sess, '/output/model.ckpt')
    #print("Model saved in file: %s" % save_path)

def print_genomes(genomes):
    """Print a list of genomes.

    Args:
        genomes (list): The population of networks/genomes

    """
    logging.info('-'*80)

    for genome in genomes:
        genome.print_genome()

class Config() :
    epochs = 10000
    n_classes = 2
    batch_size = 32
    input_shape = (256, 256, 1)
    
    datadir = '../aug_2'
    labels_filename = 'labels.txt'
        
def main():
    """Evolve a genome."""
    population = 30 # Number of networks/genomes in each generation.
    #we only need to train the new ones....
    config = Config()
        
 
    generations = 100 # Number of times to evolve the population.
    all_possible_genes = {
        'cnn_nb_layers' : [1, 2, 3, 4, 5, 6, 7, 8],
        'cnn_nb_neurons': [11, 13, 16, 32, 64, 31, 23],
        'ann_nb_layers' : [1, 2, 3, 4, 5, 6, 7, 8],
        'ann_nb_neurons': [11, 13, 16, 32, 64, 31, 23],
        'cnn_activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
        'ann_activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
        'optimizer'     : ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
     }
 
    # replace nb_neurons with 1 unique value for each layer
    # 6th value reserved for dense layer
    cnn_nb_neurons = all_possible_genes['ann_nb_neurons']
    ann_nb_neurons = all_possible_genes['ann_nb_neurons']
    for i in range(0, len(cnn_nb_neurons) ):
        all_possible_genes['cnn_nb_neurons_' + str(i)] = cnn_nb_neurons
        
    for i in range(0, len(ann_nb_neurons) ):
        all_possible_genes['ann_nb_neurons_' + str(i)] = ann_nb_neurons
        
    print( all_possible_genes )
    # remove old value from dict
    all_possible_genes.pop('cnn_nb_neurons')
    all_possible_genes.pop('ann_nb_neurons')
            
    print("***Evolving for %d generations with population size = %d***" % (generations, population))

    generate(generations, population, all_possible_genes, config)

if __name__ == '__main__':
    main()
