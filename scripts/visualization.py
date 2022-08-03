import numpy as np
import algorithms as alg
from matplotlib import pyplot as plt
from matplotlib import animation


def init_checkboard():
    """
    This function generates a video of the evolution of the system which is saves in outer-path.
    Parameters
    ----------
    None
    Returns
    -------
    checkerboard :  numpy.array
    Returns a checkboard of size 2500
    """
    x_axis = np.ones((10, 5))
    axis_bis = x_axis[::2]
    axis_bis[:] = np.ones(1)*-1

    y_axis = np.array(-x_axis)
    return np.outer(x_axis, y_axis)


def create_energy_plot(dynamics_history, weights, file):
    """
    This function generates a plot of the energy evolution of the system which in energy file.
    Parameters
    ----------
    dynamics_history : list
        list of the save states of dimmension (50,50)
    weights : ndarray
        ndarray of size (2500, 2500) specific to each algorithm
    file : string
        path where the plot is saved
    Returns
    -------
    None.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    energies = list(map(lambda x: alg.energy(x, weights), dynamics_history))
    ax.plot(np.arange(len(dynamics_history)), energies)
    ax.set_xlabel('Convergence step')
    ax.set_ylabel('Energy')
    ax.set_title('Energy of a perturbed pattern in our Hopefield network')
    fig.set_facecolor('lightsteelblue')
    fig.savefig(file)

    return None


def save_video(state_list, out_path):
    """
    This function generates a video of the evolution of the system which is saves in outer-path.
    Parameters
    ----------
    state_list : list
        list of the save states of dimmension (50,50)
    out_path : string
        path where the video is saved
    Returns
    -------
    None.
    """
    try:
        # creating the axis
        fig, ax = plt.subplots()
        # ims = [[ax.imshow(state, animated=True, cmap="binary")]
        #       for state in state_list]
        ims = list(
            map(lambda x: [ax.imshow(x, animated=True, cmap="binary")], state_list))
        # creating the animation
        anim = animation.ArtistAnimation(
            fig, ims, interval=500)
        FFwriter = animation.FFMpegWriter()
        anim.save(out_path, writer=FFwriter)

        return None

    except AttributeError:
        print("Error : save_video(state_list, out_path) - visualization.py - The parameters aren't of the correct type be sure to put a list of np.array of the correct size for state_list and a string for out_path.")
    except IndexError:
        print("Error : save_video(state_list, out_path)  - visualization.py -  The numpy array in the list are not of the correct dimension. Be sure to use an array of size 2D.")


def hebbian_algorithm_visualization(patterns, perturbed):
    """
    Generate Hebbian weights from the patterns and apply the hebbian to retreive the original pattern from the perturbed one.
    Create 2 plots and 2 videos to show the process with the synchronous an asynchronous methods.
    Parameters
    --  --  --  --  --
    patterns : numpy.array
        Input array with shape (i, j). j should be the size of each individual
        pattern, and i the number of patterns.
    perturbed : numpy.array
        One dimensionnal.
    Returns
    --  --  -- -
    None.
    """
    try:
        Hebbian_Hopfield = alg.HopfieldNetwork(patterns)
        data_async = alg.DataSaver()
        Hebbian_Hopfield.dynamics_async(
            perturbed, data_async, 200000, 10000, 1000)
        data_async.save_video("videos/hebbian_async.mp4", 100)
        data_async.plot_energy().savefig("plots/hebbian_async_energy.png")

        data_sync = alg.DataSaver()
        Hebbian_Hopfield.dynamics(perturbed, data_sync)
        data_sync.save_video("videos/hebbian_sync.mp4", 100)
        data_sync.plot_energy().savefig("plots/hebbian_sync_energy.png")
        return None

    except AttributeError:
        print("Error : hebbian_algorithm_visualization(patterns, perturbed) - visualization.py - The parameters aren't of the correct type be sure to put a numpy.array of the correct size for both.")
    except ValueError:
        print("Error : hebbian_algorithm_visualization(patterns, perturbed) - visualization.py - The dimensions of the ndarray are incorrect.")


def storkey_algorithm_visualization(patterns, perturbed):
    """
    Generate Storkey weights from the patterns and apply the storkey to retreive the original pattern from the perturbed one.
    Create 2 plots and 2 videos to show the process with the synchronous an asynchronous methods.
    Parameters
    --  --  --  --  --
    patterns : numpy.array
        Input array with shape (i, j). j should be the size of each individual
        pattern, and i the number of patterns.
    perturbed : numpy.array
        One dimensionnal.
    Returns
    --  --  -- -
    None.
    """
    try:
        Storkey_Hopfield = alg.HopfieldNetwork(patterns, rule="storkey")

        data_async_storkey = alg.DataSaver()
        Storkey_Hopfield.dynamics_async(
            perturbed, data_async_storkey, 200000, 10000, 1000)
        data_async_storkey.save_video("videos/str_async_classes.mp4")
        data_async_storkey.plot_energy().savefig("plots/str_async_energy.png")
        data_sync_storkey = alg.DataSaver()
        Storkey_Hopfield.dynamics(perturbed, data_sync_storkey)
        data_sync_storkey.save_video("videos/str_sync_classes.mp4")
        data_sync_storkey.plot_energy().savefig("plots/str_sync_energy.png")
        return None

    except AttributeError:
        print("Error : storkey_algorithm_visualization(patterns, perturbed) - visualization.py - The parameters aren't of the correct type be sure to put a numpy.array of the correct size for both.")
    except ValueError:
        print("Error : storkey_algorithm_visualization(patterns, perturbed) - visualization.py - The dimensions of the ndarray are incorrect.")

        
def video_visualization(patterns, perturbed):
    """
    Generate Hebbian weights from the patterns and apply the hebbian to retreive the original image from the perturbed one.
    Parameters
    --  --  --  --  --
    patterns : numpy.array
        Input array with shape (i, j). j should be the size of each individual
        pattern, and i the number of patterns.
    perturbed : numpy.array
        One dimensionnal.
    Returns
    --  --  -- -
    None.
    """
    try:
        Hebbian_Hopfield = alg.HopfieldNetwork(patterns)
        data_sync = alg.DataSaver()
        Hebbian_Hopfield.dynamics(perturbed, data_sync)
        data_sync.save_video("videos/snail.mp4", 100)

        return None

    except AttributeError:
        print("Error : hebbian_algorithm_visualization(patterns, perturbed) - visualization.py - The parameters aren't of the correct type be sure to put a numpy.array of the correct size for both.")
    except ValueError:
        print("Error : hebbian_algorithm_visualization(patterns, perturbed) - visualization.py - The dimensions of the ndarray are incorrect.")      
