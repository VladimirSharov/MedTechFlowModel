import math
from scipy import stats
import matplotlib.pyplot as plt
import simpy
import pandas as pd
import numpy as np
from numpy.random import default_rng
pd.options.mode.chained_assignment = None  # default='warn'
import os
from datetime import datetime

# initialization module
NUMBER_PREPARATION_UNITS = 4
NUMBER_OPERATION_UNITS = 1
NUMBER_RECOVERY_UNITS = 5

SIM_TIME = 1000

# Random generator
RNG = default_rng(1234)

DEBUG = False


def patient_arrival(env, pr, ot, rr, process_times):
    """
    process to generate the patients with an exponential arrival process

    Args:
        env (simpy Environment): working environment
        pr (simpy Ressource): preparation rooms ressource
        ot (simpy Ressource): operation theatre ressource
        rr (simpy Ressource): recovery rooms ressourc
    """
    # IDs for patients
    next_patient_id = 0
    while True:

        # Wait for the patient
        yield env.timeout(process_times[0])
        next_patient_id += 1
        if DEBUG:
            print('%3d arrives at %.2f' % (next_patient_id, env.now))

        # pass as parameters $pr, $ot, $rr which are the ressources,
        # and $normal_times, wich is the array containing the times. This parameter
        # can depend on the type of patient.
        env.process(prepare(env, pr, ot, rr, next_patient_id,
                    process_times))


def prepare(env, pr, ot, rr, patient_number, process_times):
    """
    preparation rooms process

    Args:
        env (simpy Environment): working environment
        pr (simpy Ressource): preparation rooms ressource
        ot (simpy Ressource): operation theatre ressource
        rr (simpy Ressource): recovery rooms ressource
        patient_number (int): patient number
        time_of_arrival (float): data used to compute the metrics
        process_times (list): list of processing times to individualize patient care
    """
    with pr.request() as req:
        if DEBUG:
            print('%3d enters the preparation queue at %.2f' %
                  (patient_number, env.now))

        queue_in_preparation = env.now
        # array just for the arrival times
        length = len(pr.queue)
        tme_in_queue_preparation.append(queue_in_preparation)
        len_in_queue_preparation.append(length)

        yield req
        if DEBUG:
            print('%3d starts preparation at %.2f' % (patient_number, env.now))

        queue_out_preparation = env.now
        length = len(pr.queue)
        tme_in_queue_preparation.append(queue_out_preparation)
        len_in_queue_preparation.append(length)

        yield env.timeout(process_times[1])
        if DEBUG:
            print("%3d preparation duration was %.2f" %
                  (patient_number, process_times[1]))

        env.process(operate(env, ot, rr,
                    patient_number, process_times))


def operate(env, ot, rr, patient_number, process_times):
    """
    operation room process

    Args:
        env (simpy Environment): working environment
        ot (simpy Ressource): operation theatre ressource
        rr (simpy Ressource): recovery rooms ressource
        patient_number (int): patient number
        time_of_arrival (float): data used to compute the metrics
        process_times (list): list of processing times to individualize patient care
    """
    with ot.request() as req1, rr.request() as req2:
        if DEBUG:
            print('%3d enters the operation queue at %.2f' %
                  (patient_number, env.now))

        # the patient must wait until the operation room is free AND a recovery room is free
        yield req1 & req2
        if DEBUG:
            print('%3d starts operation at %.2f' %
                  (patient_number, env.now))

        operation_time = RNG.exponential(scale=20, size=1)

        yield env.timeout(operation_time)
        if DEBUG:
            print("%3d operation duration was %.2f" %
                  (patient_number, operation_time))

        env.process(recover(env, rr,
                    patient_number, process_times))


def recover(env, rr, patient_number, process_times):
    """
    patient recovery rooms process

    Args:
        env (simpy Environment): working environment
        rr (simpy Ressource): recovery rooms ressource
        patient_number (int): patient number
        time_of_arrival (float): data used to compute the metrics
        process_times (list): list of processing times to individualize patient care
    """
    if DEBUG:
        print('%3d enters the recovery queue at %.2f' %
              (patient_number, env.now))

    yield env.timeout(process_times[2])
    if DEBUG:
        print("%3d recovery duration was %.2f" %
              (patient_number, process_times[2]))


def avg_line(df_length):
    """
    Finds the time average number of patients in the waiting line

    Returns:
        float: average
    """
    # use the next row to figure out how long the queue was
    df_length['delta_time'] = df_length['time'].shift(-1)-df_length['time']
    # drop the last row because it would have an infinite delta time
    df_length = df_length[0:-1]
    avg = np.average(df_length['len'], weights=df_length['delta_time'])
    return avg


def std_line(df_length):
    """
    Return the weighted standard deviation.
    """
    df_length['delta_time'] = df_length['time'].shift(-1)-df_length['time']
    average = avg_line(df_length)
    # Fast and numerically precise:
    variance = np.average(
        (df_length['len']-average)**2, weights=df_length['delta_time'])
    return math.sqrt(variance)


def calc_batches(df_length, number_batchs, time_in_batch, run=None):
    """
    Divides a long run into several little ones, acting as independant samples. As we are in a non-terminating 
    simulation, we are here using the [Batch means method]{https://rossetti.github.io/RossettiArenaBook/ch5-BatchMeansMethod.html}.
    The idea is to remove the transcient time only once.

    Args:
        df_length (pandas Dataframe): pandas dataframe containing a dataframe with 'time' and 'len' columns
        number_batchs (int): number of batches.
        time_in_batch (int): time of batches.
        run (int): index of the run

    Returns:
        list of floats : analytics data - mean, std, upper bound and lower bound
    """
    # compute the limite time, with one more batch for the transcient effect
    time_batches = (number_batchs+1)*time_in_batch
    # truncate the dataframe, we don't need the end
    df_length_trunc = df_length.loc[df_length['time'] < time_batches]
    # eliminating transient effects (warm-up period), equivalent to one batch time
    df_length_trunc = df_length_trunc.loc[df_length['time'] > time_in_batch]
    matrix = []
    for i in range(number_batchs):
        # we selec the lines with the times in the batches
        matrix.append(df_length_trunc.loc[
            (df_length_trunc['time'] > (i+1*time_in_batch)) & (df_length_trunc['time'] < (i+2)*time_in_batch)])

    # dof means degree of freedom
    dof = number_batchs - 1
    confidence = 0.95
    t_crit = np.abs(stats.t.ppf((1-confidence)/2, dof))
    # means of each batches
    batch_means = [avg_line(df) for df in matrix]
    # computing the overall mean and std over the means
    average_batch_means = np.mean(batch_means, axis=0)
    standard_batch_means = np.std(batch_means, axis=0)
    # now we can find the confidence intervall over the means average
    inf = average_batch_means - standard_batch_means * \
        t_crit/np.sqrt(number_batchs)
    sup = average_batch_means + standard_batch_means * \
        t_crit/np.sqrt(number_batchs)
    inf = round(float(inf), 2)
    sup = round(float(sup), 2)
    print('')
    if (run == None):
        print('Simulation of a surgery facility')
    else:
        print('Simulation of a surgery facility number%3s' % (run + 1))
    print('')
    print('%3s batches of %3s time unit were used for calculations' %
          (number_batchs,  time_in_batch))
    print('The average length of the preparation queue for all the batches is %3s ' %
          average_batch_means)
    print('')
    print('The average length of the preparation queue belongs to the interval %3s %3s' % (inf, sup))

    return average_batch_means, standard_batch_means, inf, sup


def print_output(df):
    """
    Plotting the dataframe with the results using Pandas

    Args:
        df (pandas DataFrame): dataframe containing the results we want to display
    """
    fig, ax = plt.subplots(1, 1)
    ax.axis('tight')
    ax.axis('off')
    output_table = ax.table(cellText=df.values,
                            colLabels=df.columns, rowLabels=df.index,
                            rowColours=["skyblue"]*len(df.index), colColours=["cyan"]*len(df.columns),
                            cellLoc='center', loc="center")
    ax.set_title("Output data for %i independent batches for different configurations" % (number_batchs),
                 fontsize=18, y=0.8, pad=4)
    output_table.auto_set_font_size(False)
    output_table.set_fontsize(8)
    plt.subplots_adjust(left=0.4, bottom=0.2)

    # Create the 'figures' directory if it doesn't exist
    output_directory = 'figures'
    os.makedirs(output_directory, exist_ok=True)

    # Add a timestamp to the file name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f'output_perf_measures_{timestamp}.png'
    file_path = os.path.join(output_directory, file_name)
    
    plt.savefig(file_path)
    print(f"Results saved to: {file_path}")
    plt.show()



if __name__ == '__main__':

    # arrays to keep track
    tme_in_queue_preparation, len_in_queue_preparation = [], []

    number_batchs = 10
    time_in_batch = 1000

    # preparing the dataframe to recieve the data
    col_labels = ["Mean", "Std. Dev.", "Lower bound", "Upper Bound"]
    row_labels = ["Design 1",
                  "Design 2",
                  "Design 3",
                  "Design 4",
                  "Design 5",
                  "Design 6",
                  "Design 7",
                  "Design 8"
                  ]

    df = pd.DataFrame(columns=col_labels, index=row_labels)

    designs = [
        [RNG.exponential(scale=25, size=1), RNG.exponential(
            scale=40, size=1), RNG.exponential(scale=40, size=1)],
        [RNG.exponential(scale=25, size=1), RNG.uniform(
            low=30, high=40), RNG.uniform(low=30, high=50)],
        [RNG.uniform(low=20, high=30), RNG.exponential(
            scale=40, size=1), RNG.exponential(scale=40, size=1)],
        [RNG.uniform(low=20, high=30), RNG.uniform(
            low=30, high=40), RNG.uniform(low=30, high=50)],
        [RNG.exponential(scale=22.5, size=1), RNG.exponential(
            scale=40, size=1), RNG.exponential(scale=40, size=1)],
        [RNG.exponential(scale=22.5, size=1), RNG.uniform(
            low=30, high=40), RNG.uniform(low=30, high=50)],
        [RNG.uniform(low=20, high=25), RNG.exponential(
            scale=40, size=1), RNG.exponential(scale=40, size=1)],
        [RNG.uniform(low=20, high=25), RNG.uniform(
            low=30, high=40), RNG.uniform(low=30, high=50)]
    ]

    # iterating over the different configurations

    for i, design in enumerate(designs):
        print("\n Experiment ", (i+1))
        # set up the environment
        env = simpy.Environment()
        # defining resources
        pr = simpy.Resource(env, capacity=NUMBER_PREPARATION_UNITS)
        ot = simpy.Resource(env, capacity=NUMBER_OPERATION_UNITS)
        rr = simpy.Resource(env, capacity=NUMBER_OPERATION_UNITS)

        env.process(patient_arrival(env, pr, ot, rr, process_times=design))

        # run the simulation, proceeding with a granularity of 1 time unit, decomposing between batches
        # the goal is to compute the time when the recovery rooms and the operation theatre are full
        prep_queues = []
        sample_length = 50
        sample_count = 10
        interval = 25
        sample_lists = []
        for b in range(number_batchs):
            print("Correlations for sample ", b+1)
            prep_queue = []
            sample_list = []
            for t in range((b*time_in_batch)+1, (b+1)*time_in_batch):
                env.run(until=t)
                prep_queue.append(len(pr.queue))
            # creating 10 samples for each batches
            for count in range(sample_count):
                sample_list = sample_list + \
                    [prep_queue[count*interval+count*sample_length:count *
                                interval+(count+1)*sample_length]]
            # adding the 10 samples to the list for each batch
            sample_lists.append(sample_list)
            prep_queues.append(prep_queue)

            # compute the covariance
            print(np.cov(sample_list))

        dof = number_batchs - 1
        confidence = 0.95
        t_crit = np.abs(stats.t.ppf((1-confidence)/2, dof))

        mean = np.mean(prep_queues)
        print(mean)
        std = np.std(prep_queues)

        inf = mean - \
            std*t_crit/np.sqrt(number_batchs)
        sup = mean + \
            std*t_crit/np.sqrt(number_batchs)

        # complete dataframe for mean of probablity to block the operation room
        df.iloc[i, 0] = round(mean, 2)

        # for standart deviation
        df.iloc[i, 1] = round(std, 2)

        # for sup and inf
        df.iloc[i, 2] = round(inf, 2)
        df.iloc[i, 3] = round(sup, 2)

    print(df)
    print_output(df)