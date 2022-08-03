from operator import sub
import algorithms as alg
import visualization as vis
import numpy as np
import random
from math import log, sqrt
import pandas as pds
from matplotlib import pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu

def experiment(size, num_patterns, weight_rule, num_perturb, num_trials=10, max_iter=100):
    '''
    This function run an experiment (pattern retrieving function of a spacific algorithm) to be able to return a dictionnay 
    with the informations regarding the experiment and percentage of pattern retrieval for further performance analysis.

    Parameters
    --  --  --  --  --
    size : int 
        Lenght of each pattern.
    num_patterns : int
        Number of pattterns.
    weight_rule : str
        Name of the algorithm to use for the experiment
    num_perturb : float
        Percentage of the pattern to perturb
    num_trials : int
        Number of trials in the experiment 
    max_iter : int
        Number of maximum interations allowed for each algorithm to retreive the original pattern 

    Returns
    --  --  -- -
    dict of the match frac of the experiment with the other information regarding the experiment

    '''
    # Creation of the patterns
    if num_patterns == 0:
        patterns = alg.generate_patterns(1, size)
    else:
        patterns = alg.generate_patterns(num_patterns, size)

    # Creation of nets with a specific rule
    network = alg.HopfieldNetwork(patterns, weight_rule)
    number = 0
    
    # Do the experiment num_trials times to return the fraction of successful experiments (match frac)
    for i in range(num_trials):
        if (num_patterns == 0) or (num_patterns == 1):
            randint = 0
        else:
            randint = random.randint(0, num_patterns-1)
        perturbed = alg.perturb_pattern(
            patterns[randint], int(size * num_perturb))
        saver = alg.DataSaver()
        network.dynamics(perturbed, saver, max_iter)
        if saver.get_data():
            if (alg.pattern_match(patterns, saver.get_data()[-1]) == randint):
                number += 1
    return {"network size": size, "weight rule": weight_rule,
            "num patterns": num_patterns, "num perturb": num_perturb, "match frac": number/num_trials}


def capacity(sizes_list, rule, t):
    '''
    This function a list of pattern size and apply a capacity calculation to see 
    if it's near the theoritical capacity of a specific algorithm.

    Parameters
    --  --  --  --  --
    sizes_list : list of int
        One dimensional list which gives the pattern size to experiment with.
    rule : str
        Gives the algortihm to use.
    t : int
        Number of pattern to test of each size_list value 

    Returns
    --  --  -- -
    None

    '''
    
    total_data = []  # List of all the results
    capacity_data = {}  # experimental maximum capacity (match frac superior to 0.9)
    frac_data = []  # List of dictionnaries to plot the fraction of retreived patterns from n original patterns
    
    # create the experiment material depending on the algorithm to test
    if rule == "hebbian":
        t_rule = dict(map(lambda x: (x, np.linspace(.5 * (x/(2*log(x, 2))), 2 * (x/(2*log(x, 2))), t).astype(int)), sizes_list))
    else:
        t_rule = dict(map(lambda x: (x, np.linspace(.5 * (x/sqrt(2*log(x, 2))), 2 * (x/sqrt(2*log(x, 2))), t).astype(int)), sizes_list))

    for size in sizes_list:
        # Run experiment for each pattern in sizes_list
        results = list(map(lambda t:  experiment(
            size, t, rule, 0.2), t_rule[size]))
        total_data = total_data + results
        # Creation the plots
        plot_x, plot_y= [], []
        for result in results:
            plot_x.append(result["num patterns"])
            plot_y.append(result["match frac"])
            if result['match frac'] >= 0.9:
                capacity_data[size] = result["num patterns"]
        # Save of the data of the plots
        frac_data.append((size, [plot_x, plot_y]))
        
    # Creation of the graphs for each size and run of the function example_plot   
    fig, axs = plt.subplots(5, 2, sharey=True, tight_layout=True, figsize=(10, 9))
    for i in range(2):
        for j in range(5):
            if i == 0:
                example_plot(axs[j][i], frac_data[j], "capacity for size "+ str(frac_data[j][0]))
            if i == 1:
                example_plot(axs[j][i], frac_data[j+5], "capacity for size " + str(frac_data[j+5][0]))
    
    # Formatting for the plot dispaly
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False,
                    bottom=False, left=False, right=False)
    plt.ylabel("Fraction of retrieved patterns")
    plt.xlabel("Number of pattern for each size ")
    plt.tight_layout()
    
    # Save file of the designated directory
    fig.savefig("experiment-results/capacity-"+rule+".png")

    # Store a pandas file with the results for all simulated network sizes 
    
    total_data = pds.DataFrame(total_data)
    total_data.to_hdf("experiment-results/report_capacity-"+rule+".hdf5", key='df')
    total_data = pds.read_hdf("experiment-results/report_capacity-"+rule+".hdf5")
    print("DataFrame read from the HDF5 file through pandas:")
    print(total_data.to_markdown("experiment-results/report_capacity-"+rule+".md"))
    
    successful_data = pds.DataFrame([capacity_data])
    successful_data.to_hdf("experiment-results/report_successful_capacity-"+rule+".hdf5", key='df')
    successful_data = pds.read_hdf("experiment-results/report_successful_capacity-"+rule+".hdf5")
    print("DataFrame read from the HDF5 file through pandas:")
    print(successful_data.to_markdown("experiment-results/report_successful_capacity-"+rule+".md"))

    return None




def robustness(sizes, rule, step, t, init_perturb):
    '''
    This function tests the robustness to perturbations of the model. It shows at which point 
    does the system stop converging to the initial pattern.   
    
    Parameters
    --  --  --  --  --
    size : list
        List of different sizes of the Network
    rule : string
        Set which rule is used
    step : float
        Step of increase of the perturbation 
    t : int 
        number of patterns in the Network
    init_perturb :
        initial number of perturbation 
    
    
    Returns
    --  --  -- -
    None
    '''
    # Creation of the plot
    plot_x = np.arange(0, 1.05, 0.05)
    plot_y = []
    fig, axs = plt.subplots(5, 2, sharex=True, sharey=True,
                            tight_layout=True, figsize=(10, 9))
    j = 0
    k = 0
    
    # Run experiment for each pattern in sizes_list
    data = []
    data_successful = {}
    for size in sizes:
        data_for_size = []
        for i in range(plot_x.shape[0]):
            result = experiment(size, t, rule, init_perturb + step*i)
            data_for_size.append(result["match frac"])
            data.append(result)
            if (result["match frac"] >= 0.9):
                data_successful[result["network size"]] = step*i
        # Save of the data of the plots
        plot_y.append(data_for_size)
        axs[j][k].plot(plot_x, data_for_size)
        # Creation of the graphs for each size and run of the function example_plot 
        example_plot(axs[j][k], [plot_x, data_for_size],
                     "Robustness for size " + str(size), True)
        if j == 4:
            j = 0
            k = 1
        else:
            j += 1
    fig.add_subplot(111, frameon=False)

    # Formatting for the plot display
    plt.tick_params(labelcolor='none', which='both', top=False,
                    bottom=False, left=False, right=False)
    plt.xlabel("Number of perturbation (%)")
    plt.ylabel("Fraction of retrieved patterns")
    plt.tight_layout()
    # Number of perturbation (%)
    # Save file of the designated directory
    fig.savefig("experiment-results/robustness-"+rule+".png")
    
    # Store a pandas file with the results for all simulated network sizes

    total_data = pds.DataFrame(data)
    total_data.to_hdf("experiment-results/report_robustness-"+rule+".hdf5", key='df')
    total_data = pds.read_hdf("experiment-results/report_robustness-"+rule+".hdf5")
    print("DataFrame read from the HDF5 file through pandas:")
    print(total_data.to_markdown("experiment-results/report_robustness-"+rule+".md"))
    
    successful_data = pds.DataFrame([data_successful])
    successful_data.to_hdf("experiment-results/report_successful_robustness-"+rule+".hdf5", key='df')
    successful_data = pds.read_hdf("experiment-results/report_successful_robustness-"+rule+".hdf5")
    print("DataFrame read from the HDF5 file through pandas:")
    print(successful_data.to_markdown("experiment-results/report_successful_robustness-"+rule+".md"))
    
    return None


def example_plot(ax, data, title, robustness=False, fontsize=12, hide_labels=False):
    '''
    This function creates a plot from the given data
    
    Parameters
    --  --  --  --  --
    ax : axes.Axes
        a single Axes object
    data : list
        Data on the plot
    title : string
        Title of the plot

    Returns
    --  --  -- --
    None
    '''
    if robustness == True:
        # Create plot adapted for robustness
        ax.plot(data[0], data[1])  
    else:
        #Create plot adapted for capacity 
        ax.plot(data[1][0], data[1][1])
    ax.locator_params(nbins=3)
    ax.set_title(title, fontsize=fontsize)
    
    return None


def image_retrieval():
    '''
    This function allows to recall a 100 x 100 pixel image from a perturbed pattern. The function binarizes the image and stores
    it in a Hopfield network. It demonstrates that the model can find the complete image from an
    incomplete subsets of an image.
    
    Returns
    --  --  -- -
    None
    '''
    import os
    import skimage
    from skimage import io, data, color
    filename = os.path.join(skimage.data_dir, 'snail.png')
    # Import the image
    snail = io.imread('snail.png')
    snail = color.rgb2gray(snail)
    
    # Thresholding the image to create a binary image from a grayscale imagev
    thresh = threshold_otsu(snail)
    binary = snail > thresh
    binary = binary.flatten().astype('int32')
    binary[binary == 0] = -1
    
    # Creating the Hopfield network
    patterns = alg.generate_patterns(50, 10000)
    rand_int = random.randrange(50)
    patterns[rand_int] = binary
   
    # Perturb the image
    perturbed = alg.perturb_pattern(patterns[rand_int], 2000)  # Perturbations
    
    # Generate a video of the model recalling the originial image
    vis.video_visualization(patterns, perturbed)

    return None
